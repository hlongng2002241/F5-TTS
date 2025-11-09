"""
Standalone trainer for Duration Predictor.

This script trains the duration predictor independently from the main TTS model,
using precomputed text-mel alignments to learn character-to-mel-frame durations.

Based on F5TTS-stabilized approach:
- Loss: MSE in log-space
- Ground truth: sum(alignment_matrix, axis=mel_frames)
- Training: AdamW with linear warmup + decay
"""

import argparse
from pathlib import Path
from collections import OrderedDict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from f5_tts.model.dataset import DynamicBatchSampler, load_dataset_v2, collate_fn
from f5_tts.model.duration_predictor import DurationPredictor
from f5_tts.model.utils import list_str_to_idx, get_tokenizer


class DurationPredictorTrainer:
    """Trainer for duration predictor model."""

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset=None,
        vocab_char_map=None,
        save_dir="ckpts/duration_predictor",
        batch_size_frames=52000,
        max_samples=64,
        learning_rate=1e-4,
        num_warmup_steps=1000,
        grad_clip_norm=1.0,
        save_per_steps=5000,
        log_per_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.vocab_char_map = vocab_char_map
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.grad_clip_norm = grad_clip_norm
        self.save_per_steps = save_per_steps
        self.log_per_steps = log_per_steps
        self.batch_size_frames = batch_size_frames

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Train data loader with dynamic batching
        train_sampler = torch.utils.data.SequentialSampler(train_dataset)
        train_batch_sampler = DynamicBatchSampler(
            train_sampler,
            frames_threshold=batch_size_frames,
            max_samples=max_samples,
            random_seed=42,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Test data loader (if test dataset provided)
        if test_dataset is not None:
            test_sampler = torch.utils.data.SequentialSampler(test_dataset)
            test_batch_sampler = DynamicBatchSampler(
                test_sampler,
                frames_threshold=batch_size_frames,
                max_samples=max_samples,
                random_seed=42,
            )
            self.test_loader = DataLoader(
                test_dataset,
                batch_sampler=test_batch_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
            )
        else:
            self.test_loader = None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))

        # Tracking
        self.global_step = 0
        self.epoch = 0

    def setup_scheduler(self, num_training_steps):
        """Setup learning rate scheduler with warmup and decay."""
        warmup_steps = min(self.num_warmup_steps, num_training_steps // 10)
        decay_steps = num_training_steps - warmup_steps

        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)

        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps])

    def compute_duration_loss(self, batch, return_details=False):
        text = batch["text"]
        text_lengths = batch["text_lengths"]
        attn = batch["mel_attn"].to(self.device)  # [batch, text_len, mel_len]

        max_text_len = text_lengths.max().item()

        # Tokenize text
        text_tokens = list_str_to_idx(text, self.vocab_char_map, padding_value=-1).to(self.device)  # [batch, text_len]

        # Create text mask
        text_lengths_device = text_lengths.to(self.device)
        range_tensor = torch.arange(max_text_len, device=self.device).unsqueeze(0)  # [1, text_len]
        text_mask = (range_tensor < text_lengths_device.unsqueeze(1)).int()  # [batch, text_len]

        # Ground truth durations: sum alignment over mel dimension
        w_gt = attn.sum(dim=2)  # [batch, text_len] - frames per character
        logw_gt = torch.log(w_gt + 1e-6) * text_mask.float()  # Log-space with epsilon

        # Predict durations
        logw_pred = self.model(text_tokens, text_mask)  # [batch, 1, text_len]
        logw_pred = logw_pred.squeeze(1)  # [batch, text_len]

        # MSE loss in log-space (masked)
        squared_error = (logw_pred - logw_gt) ** 2 * text_mask.float()
        loss = squared_error.sum() / text_mask.sum()

        # Additional metrics for logging
        with torch.no_grad():
            # Average predicted duration vs ground truth
            w_pred = torch.exp(logw_pred)  # Convert to linear space
            pred_duration_avg = w_pred * text_mask.float()
            gt_duration_avg = w_gt * text_mask.float()
            pred_mean = pred_duration_avg.sum() / text_mask.sum()
            gt_mean = gt_duration_avg.sum() / text_mask.sum()

        result = {
            "loss": loss,
            "pred_duration_mean": pred_mean.item(),
            "gt_duration_mean": gt_mean.item(),
        }

        # Return additional details for evaluation if requested
        if return_details:
            result["w_gt"] = w_gt
            result["w_pred"] = w_pred
            result["text_mask"] = text_mask

        return result

    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss_dict = self.compute_duration_loss(batch)
        loss = loss_dict["loss"]

        # Backward and update
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        return loss_dict

    def evaluate(self):
        """Run evaluation on test set."""
        if self.test_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_pred_dur = 0.0
        total_gt_dur = 0.0
        total_accurate = 0  # Within ±2 frames
        total_samples = 0
        num_batches = 0

        accuracy_threshold = 2.0  # Within ±2 frames is considered accurate

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                # Compute loss and get detailed outputs for accuracy calculation
                loss_dict = self.compute_duration_loss(batch, return_details=True)

                # Accumulate metrics
                total_loss += loss_dict["loss"].item()
                total_pred_dur += loss_dict["pred_duration_mean"]
                total_gt_dur += loss_dict["gt_duration_mean"]

                # Use returned tensors for accuracy calculation
                w_gt = loss_dict["w_gt"]
                w_pred = loss_dict["w_pred"]
                text_mask = loss_dict["text_mask"]

                # Count accurate predictions (within threshold)
                error = torch.abs(w_pred - w_gt) * text_mask.float()
                accurate = (error <= accuracy_threshold).float() * text_mask.float()
                total_accurate += accurate.sum().item()
                total_samples += text_mask.sum().item()

                num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_pred_dur = total_pred_dur / num_batches
        avg_gt_dur = total_gt_dur / num_batches
        accuracy = total_accurate / total_samples if total_samples > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "pred_duration_mean": avg_pred_dur,
            "gt_duration_mean": avg_gt_dur,
            "accuracy": accuracy,
        }

        # Log to TensorBoard
        self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
        self.writer.add_scalar("eval/pred_duration_mean", avg_pred_dur, self.global_step)
        self.writer.add_scalar("eval/gt_duration_mean", avg_gt_dur, self.global_step)
        self.writer.add_scalar("eval/accuracy", accuracy, self.global_step)

        return metrics

    def save_checkpoint(self, step, loss):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"duration_predictor_step_{step}.pt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                "step": step,
                "loss": loss,
                "vocab_char_map": self.vocab_char_map,
            },
            checkpoint_path,
        )

        print(f"Saved checkpoint to {checkpoint_path}")

    def train(self, num_epoch=50):
        """Main training loop."""
        print(f"Starting training for {num_epoch} epochs")
        print(f"Batch size: {self.train_loader.batch_sampler.frames_threshold} frames")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")

        # Estimate total steps for scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = num_epoch * steps_per_epoch
        self.setup_scheduler(total_steps)

        # Training loop
        running_loss = 0.0
        running_pred_dur = 0.0
        running_gt_dur = 0.0

        for epoch in range(1, num_epoch + 1):
            self.train_loader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epoch}")

            for batch in progress_bar:
                batch_size = len(batch["text"])
                loss_dict = self.train_step(batch)

                running_loss += loss_dict["loss"].item()
                running_pred_dur += loss_dict["pred_duration_mean"]
                running_gt_dur += loss_dict["gt_duration_mean"]
                lr = self.optimizer.param_groups[0]["lr"]

                self.global_step += 1

                progress_bar.set_postfix(
                    OrderedDict(
                        {
                            "batch_size": batch_size,
                            "loss": f"{running_loss:.4f}",
                            "pred_dur": f"{running_pred_dur:.2f}",
                            "gt_dur": f"{running_gt_dur:.2f}",
                            "lr": f"{lr:.2e}",
                        }
                    )
                )

                # Logging
                if self.global_step % self.log_per_steps == 0:
                    avg_loss = running_loss / self.log_per_steps
                    avg_pred = running_pred_dur / self.log_per_steps
                    avg_gt = running_gt_dur / self.log_per_steps

                    # Log to TensorBoard
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/pred_duration_mean", avg_pred, self.global_step)
                    self.writer.add_scalar("train/gt_duration_mean", avg_gt, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

                    running_loss = 0.0
                    running_pred_dur = 0.0
                    running_gt_dur = 0.0

                # Save checkpoint
                if self.global_step % self.save_per_steps == 0:
                    self.save_checkpoint(self.global_step, loss_dict["loss"].item())

            progress_bar.close()

            # Run evaluation at end of epoch
            if self.test_loader is not None:
                print(f"\nRunning evaluation for epoch {epoch}...")
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(
                        f"Eval - Loss: {eval_metrics['loss']:.4f}, "
                        f"Pred Duration: {eval_metrics['pred_duration_mean']:.2f}, "
                        f"GT Duration: {eval_metrics['gt_duration_mean']:.2f}, "
                        f"Accuracy: {eval_metrics['accuracy']:.2%}"
                    )

            self.save_checkpoint(self.global_step, loss_dict["loss"].item())
            
        # Final checkpoint
        self.save_checkpoint(self.global_step, loss_dict["loss"].item())

        print(f"\nTraining completed! Final loss: {loss_dict['loss'].item():.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Duration Predictor")

    # Data
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.txt")

    # Model
    parser.add_argument("--text_dim", type=int, default=512, help="Text embedding dimension")
    parser.add_argument("--filter_channels", type=int, default=32, help="Filter channels in conv layers")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for conv layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")

    # Training
    parser.add_argument("--batch_size_frames", type=int, default=52000, help="Batch size in frames")
    parser.add_argument("--max_samples", type=int, default=64, help="Max samples per batch")
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of training epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="ckpts/duration_predictor", help="Save directory")
    parser.add_argument("--save_per_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--log_per_steps", type=int, default=100, help="Log every N steps")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load vocab
    print(f"Loading vocabulary from {args.vocab_path}")
    vocab_char_map, vocab_size = get_tokenizer(args.vocab_path, "custom")

    # Load dataset
    print(f"Loading dataset from {args.train_path}")
    train_dataset = load_dataset_v2(args.train_path)
    print(f"Loading dataset from {args.test_path}")
    test_dataset = load_dataset_v2(args.test_path)

    # Initialize model
    print("Initializing duration predictor...")
    model = DurationPredictor(
        text_num_embeds=vocab_size,
        text_dim=args.text_dim,
        filter_channels=args.filter_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )

    # Initialize trainer
    trainer = DurationPredictorTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        vocab_char_map=vocab_char_map,
        save_dir=args.save_dir,
        batch_size_frames=args.batch_size_frames,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        save_per_steps=args.save_per_steps,
        log_per_steps=args.log_per_steps,
        device=args.device,
    )

    # Train
    trainer.train(num_epoch=args.num_epoch)


if __name__ == "__main__":
    main()
