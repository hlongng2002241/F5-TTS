taskset -c 0-15 accelerate launch --config_file ckpts/accelerate.yaml src/f5_tts/train/train.py \
    --config-name=F5TTS_Base_mas \
    \
    datasets.train_path=data/vi/train.jsonl \
    datasets.test_path=data/vi/test.jsonl \
    datasets.batch_size_per_gpu=11000 \
    datasets.batch_size_type=frame \
    datasets.max_samples=64 \
    datasets.num_workers=8 \
    \
    optim.epochs=100 \
    optim.num_warmup_updates=20000 \
    optim.eval_first=true \
    optim.restart=true \
    \
    mas.lr_mas_components=1e-4 \
    mas.lr_v0v1_components=1e-5 \
    \
    ckpts.log_samples=false \
    ckpts.logger=tensorboard \
    ckpts.save_per_updates=20000 \
    ckpts.keep_last_n_checkpoints=-1 \
    ckpts.last_per_updates=1000 \
    ckpts.save_dir=ckpts/f5tts_vi_mas_ft \
    ckpts.resume_from_checkpoint=ckpts/f5tts_vi/model_last_no_optim.pt \
    \
    model.tokenizer=custom \
    model.tokenizer_path=ckpts/f5tts_vi/vocab.txt