## Dependencies

- Python 3.9  
- Pytorch 2.7.1 + cu126  
- Check [requirements.txt](./requirements.txt) for other dependencies.

## Data Preparation
1. You can download the images from the original source and place them in ln_data folder:
- RefCOCO/RefCOCO+/RefCOCOg
- Referit
Finally, the 'ln_data' folder will have the following structure:
```
|-- ln_data
   |-- other/images/mscoco/images/train2014/
   |-- referit
```
2. Download [data](https://drive.google.com/drive/folders/1g_fGobBKY00clHxb4fbIRcgecSO-Gf66?usp=sharing) labels here
3. Download [yolov8l.pt](https://github.com/ultralytics/ultralytics) and place it in 'saved_models' folder

## Training and Evaluation
1. Training
```
python train_yolo_v8.py --gpu '0' \
--nb_epoch 100 \
--batch_size 32 \
--size 512 \
--dataset referit/unc/unc+/gred_umd \
--savename output
```
2. Evaluation
```
python train_yolo_v8.py --gpu '0' \
--dataset referit/unc/unc+/gred_umd \
--resume xxx \
--savename test_output
--test
```
## Acknowledgement
This project is based on and inspired by:
1. [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
2. [A Fast and Accurate One-Stage Approach to Visual Grounding](https://github.com/zyang-ur/onestage_grounding)

We thank the authors for their great work!
