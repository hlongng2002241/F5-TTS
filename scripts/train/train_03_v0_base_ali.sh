taskset -c 12-31 accelerate launch --config_file ckpts/accelerate.yaml src/f5_tts/train/train.py \
    --config-name=F5TTS_Base_vi \
    \
    datasets.train_path=data/vi/test_with_alignments.data \
    datasets.test_path=data/vi/test_with_alignments.data \
    datasets.batch_size_per_gpu=11000 \
    datasets.batch_size_type=frame \
    datasets.max_samples=64 \
    datasets.num_workers=8 \
    \
    optim.epochs=20 \
    optim.learning_rate=3e-5 \
    optim.num_warmup_updates=20000 \
    optim.eval_first=false \
    optim.restart=true \
    \
    ckpts.log_samples=true \
    ckpts.logger=tensorboard \
    ckpts.save_per_updates=20000 \
    ckpts.keep_last_n_checkpoints=-1 \
    ckpts.last_per_updates=1000 \
    ckpts.save_dir=ckpts/f5tts_vi_ft_ali_r2 \
    ckpts.resume_from_checkpoint=ckpts/f5tts_vi/model_last.pt \
    \
    model.tokenizer=custom \
    model.tokenizer_path=ckpts/f5tts_vi/vocab.txt
    