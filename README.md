how to run:

preprocessing_video

1. extract video ke png
   py extract_compressed_video.py --input_dir --output_dir

2. extract png ke landmark
   py detect_landmarks.py --input_dir --output_dir

cropmouth
py preprocessing/crop_mouths.py --dataset CelebDF

evaluate model
py evaluate.py --dataset CelebDF --weights_forgery ./models/weights/lipforensics_ff.pth
