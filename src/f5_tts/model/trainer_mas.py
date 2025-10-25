from __future__ import annotations

import gc
import math
import os
import logging

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


_logger = logging.getLogger(__name__)


# trainer


class TrainerMAS:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        verbose=False,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        logging_step: int = 10,
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        # MAS-specific parameters
        use_mas: bool = False,
        mas_warmup_steps_alpha: int = 150000,
        mas_warmup_steps_temperature: int = 100000,
        duration_loss_weight: float = 0.1,
        lr_mas_components: float = 1e-4,
        lr_v0v1_components: float = 1e-5,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.verbose = verbose
        if self.verbose:
            assert checkpoint_path is not None
            from quick_utils.common.logger import setup_logger

            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            setup_logger(file=os.path.join(checkpoint_path, "train.log"), stdout=False)

        self.logger = logger
        self.logging_step = logging_step
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            assert checkpoint_path is not None
            self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, "tensorboard"))

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print("Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e")

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        # MAS-specific settings
        self.use_mas = use_mas
        self.mas_warmup_steps_alpha = mas_warmup_steps_alpha
        self.mas_warmup_steps_temperature = mas_warmup_steps_temperature
        self.duration_loss_weight = duration_loss_weight
        self.lr_mas_components = lr_mas_components
        self.lr_v0v1_components = lr_v0v1_components

        # Validation: trainer_mas.py requires MAS to be enabled
        assert self.use_mas, "trainer_mas.py requires use_mas=True"
        assert self.duration_predictor is not None, "MAS training requires duration_predictor"
        assert hasattr(model, "use_mas") and model.use_mas, "Model must have use_mas=True for MAS training"

        # Setup optimizer with parameter groups for MAS
        # Split parameters into V0/V1 components and MAS components
        mas_param_names = {"similarity_proj", "duration_predictor", "mel_feature_proj"}

        v0v1_params = []
        mas_params = {}

        for name, param in model.named_parameters():
            if any(mas_name in name for mas_name in mas_param_names):
                assert name not in mas_params
                mas_params[name] = param
            else:
                v0v1_params.append(param)

        param_groups = [
            {"params": v0v1_params, "lr": self.lr_v0v1_components},
            {"params": list(mas_params.values()), "lr": self.lr_mas_components},
        ]

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(param_groups)
        else:
            self.optimizer = AdamW(param_groups)

        if self.is_main:
            print(
                f"MAS Training: Using separate learning rates - V0/V1: {self.lr_v0v1_components}, MAS: {self.lr_mas_components}"
            )
            print(f"  - V0/V1 parameters: {len(v0v1_params)}")
            print(f"  - MAS parameters: {len(mas_params)} | {mas_params.keys()}")

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def evaluate(self, test_dataloader, global_update):
        """Evaluate model on test dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(
            test_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
            position=2,
        )
        with torch.no_grad():
            for batch in pbar:
                text_inputs = batch["text"]
                mel_spec = batch["mel"].permute(0, 2, 1)
                mel_lengths = batch["mel_lengths"]

                # Handle different return formats (standard vs MAS)
                # During evaluation, we still need to compute MAS loss for duration predictor
                model_output = self.model(
                    mel_spec,
                    text=text_inputs,
                    lens=mel_lengths,
                    noise_scheduler=self.noise_scheduler,
                )
                # For MAS training, output is (loss, cond, pred, text_embed, attn, dur_loss)
                loss = model_output[0]

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix(batch_size=len(text_inputs))

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Gather losses from all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.accelerator.device)
        avg_loss_tensor = self.accelerator.gather(avg_loss_tensor).mean()

        # All processes get the gathered average
        avg_loss = avg_loss_tensor.item()

        if self.accelerator.is_local_main_process:
            print(f"\nTest Loss: {avg_loss:.4f}")

            # Log to wandb/tensorboard
            self.accelerator.log({"test_loss": avg_loss}, step=global_update)
            if self.logger == "tensorboard":
                self.writer.add_scalar("test_loss", avg_loss, global_update)

        self.model.train()

        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure all processes finish evaluation before continuing
        self.accelerator.wait_for_everyone()

        return avg_loss

    def _load_state_dict_with_mas_compatibility(self, model, checkpoint_state_dict, is_ema=False):
        """
        Load state dict with MAS compatibility.

        Handles loading V0/V1 checkpoints (without MAS components) into MAS-enabled models.
        Uses strict=True for matching parameters while gracefully skipping MAS components
        that don't exist in the checkpoint.

        Args:
            model: The model to load into
            checkpoint_state_dict: State dict from checkpoint
            is_ema: Whether loading EMA model (keys have "ema_model." prefix)
        """
        # MAS component keys that may be missing in V0/V1 checkpoints
        mas_key_suffixes = [
            "transformer.text_embed.similarity_proj.weight",
            "transformer.text_embed.similarity_proj.bias",
            "transformer.text_embed.mas_alpha",
            "transformer.text_embed.mas_temperature",
            "mel_feature_proj.weight",
            "mel_feature_proj.bias",
            "duration_predictor.conv_1.weight",
            "duration_predictor.conv_1.bias",
            "duration_predictor.norm_1.gamma",
            "duration_predictor.norm_1.beta",
            "duration_predictor.conv_2.weight",
            "duration_predictor.conv_2.bias",
            "duration_predictor.norm_2.gamma",
            "duration_predictor.norm_2.beta",
            "duration_predictor.proj.weight",
            "duration_predictor.proj.bias",
        ]

        # Add "ema_model." prefix if loading EMA model
        if is_ema:
            mas_keys = {f"ema_model.{k}" for k in mas_key_suffixes}
        else:
            mas_keys = set(mas_key_suffixes)

        model_state_dict = model.state_dict()
        checkpoint_keys = set(checkpoint_state_dict.keys())
        model_keys = set(model_state_dict.keys())

        # Check if checkpoint is missing MAS components
        missing_mas_keys = mas_keys & model_keys - checkpoint_keys
        print("missing_mas_keys =", missing_mas_keys)

        if missing_mas_keys:
            # Loading V0/V1 checkpoint into MAS model
            model_type = "EMA model" if is_ema else "model"
            print("\n" + "=" * 70)
            print(f"Loading V0/V1 checkpoint (no MAS) into MAS-enabled {model_type}")
            print("=" * 70)
            print(f"MAS components will use initialized values:")
            print(f"  - similarity_proj: identity matrix")
            print(f"  - duration_predictor: random initialization")
            print(f"  - mel_feature_proj: random initialization")
            print("=" * 70 + "\n")

            # Load only non-MAS parameters with strict=True
            checkpoint_filtered = {k: v for k, v in checkpoint_state_dict.items() if k not in mas_keys}
            model_filtered_keys = model_keys - mas_keys

            # Verify no unexpected missing/extra keys (excluding MAS)
            missing_keys = model_filtered_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys - mas_keys

            if missing_keys:
                raise RuntimeError(f"Missing keys in checkpoint (non-MAS): {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")

            # Load with strict=False (only because of MAS keys)
            model.load_state_dict(checkpoint_filtered, strict=False)

        else:
            # Checkpoint has MAS components or model doesn't use MAS - normal loading
            model.load_state_dict(checkpoint_state_dict, strict=True)

    def save_checkpoint(self, update, last=False, epoch=None):
        self.log(f"save_checkpoint called, waiting for everyone...", flush=True)
        self.accelerator.wait_for_everyone()
        self.log(f"wait_for_everyone completed in save_checkpoint", flush=True)

        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            elif epoch is not None:
                # Save epoch checkpoint
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_epoch={epoch}_update={update}.pt")
                print(f"Saved epoch {epoch} checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

        # Ensure all processes finish checkpoint saving before continuing
        self.log(f"save_checkpoint done, final wait_for_everyone...", flush=True)
        self.accelerator.wait_for_everyone()
        self.log(f"save_checkpoint completely finished", flush=True)

    def load_checkpoint(self, resume_from_checkpoint: str = None) -> int:
        latest_checkpoint = None
        if resume_from_checkpoint is not None:
            latest_checkpoint = resume_from_checkpoint
        elif (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if latest_checkpoint is not None:
            pass
        elif "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
            latest_checkpoint = os.path.join(self.checkpoint_path, latest_checkpoint)
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

            latest_checkpoint = os.path.join(self.checkpoint_path, latest_checkpoint)

        print("Loading checkpoint at", latest_checkpoint)

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(latest_checkpoint, device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(latest_checkpoint, weights_only=True, map_location="cpu")
        else:
            raise NotImplementedError(latest_checkpoint)

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            # Handle loading V0/V1 checkpoints (no MAS) into MAS-enabled EMA model
            self._load_state_dict_with_mas_compatibility(self.ema_model, checkpoint["ema_model_state_dict"], is_ema=True)

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            # Handle loading V0/V1 checkpoints (no MAS) into MAS-enabled models
            self._load_state_dict_with_mas_compatibility(
                self.accelerator.unwrap_model(self.model), checkpoint["model_state_dict"]
            )
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loading optimizer from checkpoint")
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Loading scheduler from checkpoint")
            update = checkpoint["update"]

        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            # Handle loading V0/V1 checkpoints (no MAS) into MAS-enabled models
            self._load_state_dict_with_mas_compatibility(
                self.accelerator.unwrap_model(self.model), checkpoint["model_state_dict"]
            )
            update = 0

        del checkpoint
        gc.collect()
        assert isinstance(update, int)
        return update

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset = None,
        eval_first=False,
        resume_from_checkpoint: str = None,
        restart=False,
        num_workers=16,
        resumable_with_seed: int = None,
    ):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
                device="cpu",
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            # Create batches on all data, then distribute across GPUs
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
                num_replicas=self.accelerator.num_processes,  # Number of GPUs
                rank=self.accelerator.process_index,  # Current GPU rank
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        # Create test dataloader (same configuration as train dataloader)
        test_dataloader = None
        if test_dataset is not None:
            if self.batch_size_type == "sample":
                test_dataloader = DataLoader(
                    test_dataset,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_size=self.batch_size_per_gpu,
                    shuffle=False,
                )
            elif self.batch_size_type == "frame":
                self.accelerator.even_batches = False
                # Create batches on all data for test set, then distribute across GPUs
                test_sampler = SequentialSampler(test_dataset)
                test_batch_sampler = DynamicBatchSampler(
                    test_sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=None,  # No shuffling for test set
                    drop_residual=False,
                    num_replicas=self.accelerator.num_processes,  # Number of GPUs
                    rank=self.accelerator.process_index,  # Current GPU rank
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_sampler=test_batch_sampler,
                )
            test_dataloader = self.accelerator.prepare(test_dataloader)

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint(resume_from_checkpoint)
        if restart:
            start_update = 0
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        if eval_first:
            self.evaluate(test_dataloader, global_update)

        for epoch in tqdm(list(range(skipped_epoch, self.epochs)), desc="Training"):
            # Synchronize all processes at the start of each epoch
            self.accelerator.wait_for_everyone()
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for DynamicBatchSampler if it exists
            if self.batch_size_type == "frame":
                if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                    train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
                position=1,
            )

            for batch_idx, batch in enumerate(current_dataloader):
                if batch_idx % 10 == 0:
                    self.log(f"Batch {batch_idx}, Update {global_update}", flush=True)

                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # Update MAS parameters (alpha and temperature) for gradual mixing
                    from f5_tts.model.utils import update_mas_alpha, update_mas_temperature

                    mas_alpha = update_mas_alpha(
                        self.accelerator.unwrap_model(self.model), global_update, warmup_steps=self.mas_warmup_steps_alpha
                    )
                    mas_temperature = update_mas_temperature(
                        self.accelerator.unwrap_model(self.model),
                        global_update,
                        warmup_steps=self.mas_warmup_steps_temperature,
                    )

                    # Forward pass - always returns extended output for MAS training
                    model_output = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
                        noise_scheduler=self.noise_scheduler,
                        returns_text_tokens=True,
                    )

                    # Unpack extended output: (loss, cond, pred, text, text_embed, attn, dur_loss)
                    loss, cond, pred, text_tokens, text_embed, attn, dur_loss = model_output

                    # Combine flow matching loss and duration loss
                    total_loss = loss + self.duration_loss_weight * dur_loss

                    self.accelerator.backward(total_loss)

                    # Compute gradient norm before clipping
                    grad_norm = 0.0
                    if self.accelerator.sync_gradients:
                        # Compute total gradient norm across all parameters
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm if self.max_grad_norm > 0 else float("inf")
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update),
                        loss=loss.item(),
                        dur_loss=dur_loss.item(),
                        batch_size=len(text_inputs),
                        lr=self.scheduler.get_last_lr()[0],
                    )

                if self.accelerator.is_local_main_process and global_update % self.logging_step == 0:
                    log_dict = {
                        "loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "mas_alpha": mas_alpha,
                        "mas_temperature": mas_temperature,
                        "duration_loss": dur_loss.item(),
                        "total_loss": total_loss.item(),
                        "grad_norm": grad_norm,
                    }

                    self.accelerator.log(log_dict, step=global_update)

                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
                        self.writer.add_scalar("mas_alpha", mas_alpha, global_update)
                        self.writer.add_scalar("mas_temperature", mas_temperature, global_update)
                        self.writer.add_scalar(
                            "duration_loss", dur_loss.item() if torch.is_tensor(dur_loss) else dur_loss, global_update
                        )
                        self.writer.add_scalar("total_loss", total_loss.item(), global_update)
                        self.writer.add_scalar("grad_norm", grad_norm, global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.log(f"Saving last checkpoint at update {global_update}", flush=True)
                    self.save_checkpoint(global_update, last=True)
                    self.log(f"Finished saving last checkpoint", flush=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.log(f"Saving checkpoint at update {global_update}", flush=True)
                    self.save_checkpoint(global_update)
                    self.log(f"Finished saving checkpoint", flush=True)

                    # Evaluate after saving checkpoint
                    if test_dataloader is not None:
                        self.evaluate(test_dataloader, global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).cpu()
                            ref_mel_spec = batch["mel"][0].unsqueeze(0).cpu()
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec)
                                ref_audio = vocoder.decode(ref_mel_spec)
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0)
                                ref_audio = vocoder(ref_mel_spec).squeeze(0)

                        torchaudio.save(f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate)
                        torchaudio.save(f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate)
                        self.model.train()

            # Save checkpoint and evaluate at the end of each epoch
            # Note: Don't add wait_for_everyone() here with even_batches=False
            # as GPUs may have different batch counts and will deadlock
            self.log(f"Exited training loop for epoch {epoch}, update {global_update}", flush=True)
            progress_bar.close()

            self.log(f"Epoch {epoch} completed. About to save checkpoint...", flush=True)

            self.save_checkpoint(global_update, epoch=epoch)
            self.log(f"Finished saving epoch checkpoint", flush=True)

            if test_dataloader is not None:
                if self.accelerator.is_local_main_process:
                    print(f"\n{'='*60}")
                    print(f"Running evaluation...")
                    print(f"{'='*60}")
                self.evaluate(test_dataloader, global_update)

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()

    def log(self, *value, end="\n", sep=" ", flush=False, file=None):
        if self.verbose:
            # print(f"[GPU {self.accelerator.process_index}]", *value, end=end, sep=sep, flush=flush, file=file)
            _logger.info(sep.join([f"[GPU {self.accelerator.process_index}]"] + [str(v) for v in value]))
