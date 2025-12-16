taskset -c 12-31 accelerate launch --config_file ckpts/multi_gpu.yaml src/f5_tts/train/train.py \
    --config-name=F5TTS_v1_Base_vi \
    \
    datasets.train_path=data/vi/train_with_alignments.data \
    datasets.test_path=data/vi/test_with_alignments.data \
    datasets.synthesize_path=data/vi/synthesize_with_attn.json \
    datasets.batch_size_per_gpu=50000 \
    datasets.batch_size_type=frame \
    datasets.max_samples=180 \
    datasets.num_workers=8 \
    \
    optim.epochs=25 \
    optim.learning_rate=4e-5 \
    optim.num_warmup_updates=4000 \
    optim.eval_first=true \
    optim.restart=true \
    \
    ckpts.log_samples_per_updates=250 \
    ckpts.logger=tensorboard \
    ckpts.save_per_updates=5000000 \
    ckpts.keep_last_n_checkpoints=-1 \
    ckpts.last_per_updates=500 \
    ckpts.save_dir=ckpts/f5tts_vi_v1_base_ft_ali_full \
    ckpts.resume_from_checkpoint=ckpts/f5tts_vi_v1_base/model_last_ema.safetensors \
    \
    model.tokenizer=custom \
    model.tokenizer_path=ckpts/f5tts_vi_v1_base/vocab.txt \
    model.arch.mel_attn_alpha=0.0
    