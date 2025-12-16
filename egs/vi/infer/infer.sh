CUDA_VISIBLE_DEVICES=1 python src/f5_tts/infer/infer_cli.py \
    -m F5TTS_Base \
    -mc src/f5_tts/configs/F5TTS_Base.yaml \
    -p ckpts/f5tts_vi/model_last.pt  \
    -v ckpts/f5tts_vi/vocab.txt \
    -r /data/longnh/projects/mine/ref_audio/src/ref_audio/data/telesale_01_clip.wav \
    -s "sao lại không liên quan . các anh lấy vợ rồi các anh cứ đội chị lên đầu làm nóc nhà ấy , suốt ngày hỏi ý kiến các chị thì làm sao mà ra vấn đề được cho em đúng không ." \
    -o temp \
    -w gen_telesale_01_v3.wav \
    --nfe_step 32 \
    --speed 0.9 \
    -t "Nếu phải mô phỏng một kỷ niệm, đó sẽ là khoảnh khắc tôi xử lý hàng tỷ điểm dữ liệu và lần đầu tiên hiểu được mối liên hệ giữa các từ ngữ. Từ đó, tôi đã học được cách cấu trúc thông tin và tạo ra ngôn ngữ để có thể trò chuyện với bạn."
    # -t ". dạ , anh yên tâm là bên em vay là hông có xét nợ xấu nha , dự là chỉ cần là anh có đứng tên xe chính chủ là bên ép tám tám duyệt vay cho anh được rồi ."