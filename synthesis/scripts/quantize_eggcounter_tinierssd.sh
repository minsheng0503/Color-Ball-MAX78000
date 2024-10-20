#!/bin/sh
python quantize.py /data3/cms/max78000/ai8x-synthesis/trained/ai85-ball-qat8.pth.tar /data3/cms/max78000/ai8x-synthesis/trained/ai85-ball-qat8-q.pth.tar --device MAX78000 -v
