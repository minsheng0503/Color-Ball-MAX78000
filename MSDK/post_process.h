/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
 * (now owned by Analog Devices, Inc.),
 * Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved. This software
 * is proprietary to Analog Devices, Inc. and its licensors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define SQUARE(x) ((x) * (x))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define IMG_SCALE 3

#define IMAGE_SIZE_X 74
#define IMAGE_SIZE_Y 74

#define CAMERA_SIZE_X (IMG_SCALE * IMAGE_SIZE_X)
#define CAMERA_SIZE_Y (IMG_SCALE * IMAGE_SIZE_Y)

#define TFT_X_OFFSET 50

#define TFT_W 320
#define TFT_H 240
#define NUM_ARS 2
#define NUM_SCALES 2
#define NUM_CLASSES 6

#define LOC_DIM 4 //(x, y, w, h) or (x1, y1, x2, y2)

#define NUM_PRIORS_PER_AR 425
#define NUM_PRIORS NUM_PRIORS_PER_AR *NUM_ARS

#define MAX_PRIORS 100
#define MIN_CLASS_SCORE 19660 // ~0.4*65536
#define MAX_ALLOWED_OVERLAP 0.1 //170

void get_priors(void);
void nms(void);
void get_cxcy(float *cxcy, int prior_idx);
void gcxgcy_to_cxcy(float *cxcy, int prior_idx, float *priors_cxcy);
void cxcy_to_xy(float *xy, float *cxcy);
void localize_objects(void);
