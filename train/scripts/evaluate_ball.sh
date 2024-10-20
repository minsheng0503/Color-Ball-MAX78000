#!/bin/sh
python train.py --deterministic --print-freq 1 --batch-size 2 --model ai85tinierssd --use-bias --dataset BALL_74_74 --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_svhn.yaml --qat-policy policies/qat_policy_eggcounter.yaml --evaluate -8 --exp-load-weights-from /data3/cms/max78000/ai8x-synthesis/trained/ai85-ball-qat8-q.pth.tar --validation-split 0 --save-sample 1