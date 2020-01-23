echo 'Batch 1'
docker run -it -v /home/pi/mnt:/mnt sonyc_ml_full_tf:0.1 python /mnt/pi_ml_container/test_emb_gen_tflite.py /mnt/pi_ml_container/tflite_models/quantized_model_default.tflite /mnt/pi_ml_container/data/dog_1.wav 48000 256 242

echo 'Batch 2'
docker run -it -v /home/pi/mnt:/mnt sonyc_ml_full_tf:0.1 python /mnt/pi_ml_container/test_emb_gen_tflite.py /mnt/pi_ml_container/tflite_models/quantized_model_8000_default.tflite /mnt/pi_ml_container/data/dog_1.wav 8000 64 160
