/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef DISTRUB_UTIL_CUH_
#define DISTRUB_UTIL_CUH_

#include "hl_base.h"

/*
 * Functionality: randomly rotate, scale and sample a minibatch of images
                  and their label maps
 * images:            (numImages, imgPixels, 3)
 * targets:           (numImages, imgPixels, 3)
 *
 * created by Wei Xu. Converted to paddle by Jiang Wang.
 */
void hl_conv_random_disturb(const real* images, int imgSize, int tgtSize,
                            int channels, int numImages, real scaleRatio,
                            real rotateAngle, int samplingRate,
                            real* gpu_r_angle, real* gpu_s_ratio,
                            int* gpu_center_r, int* gpu_center_c,
                            int paddingValue, bool isTrain, real* targets);

void hl_conv_random_disturb_with_params(const real* images, int imgSize,
                                        int tgtSize, int channels,
                                        int numImages, int samplingRate,
                                        const real* gpuRotationAngle,
                                        const real* gpuScaleRatio,
                                        const int* gpuCenterR,
                                        const int* gpuCenterC,
                                        int paddingValue, real* targets);

void hl_generate_disturb_params(real*& gpuAngle, real*& gpuScaleRatio,
                                int*& gpuCenterR, int*& gpuCenterC,
                                int numImages, int imgSize,
                                real rotateAngle, real scaleRatio,
                                int samplingRate, bool isTrain);

#endif /* DISTURB_UTIL_CUH_ */
