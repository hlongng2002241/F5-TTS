# CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python src/f5_tts/train/train.py \
taskset -c 0-15 accelerate launch --config_file ckpts/accelerate.yaml src/f5_tts/train/train.py \
    --config-name=F5TTS_Base_vi \
    \
    datasets.train_path=data/vi/train.jsonl \
    datasets.test_path=data/vi/test.jsonl \
    datasets.batch_size_per_gpu=15000 \
    datasets.batch_size_type=frame \
    datasets.max_samples=80 \
    datasets.num_workers=16 \
    \
    optim.epochs=200 \
    optim.learning_rate=5e-5 \
    optim.eval_first=true \
    \
    ckpts.log_samples=true \
    ckpts.logger=tensorboard \
    ckpts.save_per_updates=50000 \
    ckpts.keep_last_n_checkpoints=5 \
    ckpts.last_per_updates=1000 \
    ckpts.save_dir=ckpts/f5tts_vi_ft \
    ckpts.resume_from_checkpoint=ckpts/f5tts_vi/model_last.pt \
    \
    model.tokenizer=custom \
    model.tokenizer_path=ckpts/vi_vocab.txt