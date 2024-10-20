#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix perfect_ball_v1 --checkpoint-file /data3/cms/max78000/ai8x-synthesis/trained/ai85-ball-qat8-q.pth.tar --config-file /data3/cms/max78000/ai8x-synthesis/networks/ball.yaml --sample-input /data3/cms/max78000/ai8x-synthesis/tests/sample_ball_74_74.npy --overlap-data --mlator --overwrite $COMMON_ARGS "$@"

