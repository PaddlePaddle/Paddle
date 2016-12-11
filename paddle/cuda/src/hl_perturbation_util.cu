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


#include <cmath>
#include <stdlib.h>
#include "hl_cuda.h"
#include "hl_time.h"
#include "hl_base.h"
#include "hl_perturbation_util.cuh"

#define _USE_MATH_DEFINES

/*
 * Get the original coordinate for a pixel in a transformed image.
 * x, y: coordiate in the transformed image.
 * tgtCenter: the center coordiate of the transformed image.
 * imgSCenter: the center coordinate of the source image.
 * centerX, centerY: translation.
 * sourceX, sourceY: output coordinates in the original image.
 */
__device__ void getTranformCoord(int x, int y, real theta, real scale,
                                 real tgtCenter, real imgCenter,
                                 real centerR, real centerC,
                                 int* sourceX, int* sourceY) {
  real H[4] = {cosf(-theta), -sinf(-theta), sinf(-theta), cosf(-theta)};

  // compute coornidates in the rotated and scaled image
  real x_new = x - tgtCenter + centerC;
  real y_new = y - tgtCenter + centerR;

  // compute coornidates in the original image
  x_new -= imgCenter;
  y_new -= imgCenter;
  real xx = H[0] * x_new + H[1] * y_new;
  real yy = H[2] * x_new + H[3] * y_new;
  *sourceX = __float2int_rn(xx / scale + imgCenter);
  *sourceY = __float2int_rn(yy / scale + imgCenter);
}

/*
 * imgs:            (numImages, imgPixels)
 * target:          (numImages * samplingRate, tgtPixels)
 * the channels of one pixel are stored continuously in memory.
 *
 * created by Wei Xu (genome), converted by Jiang Wang
 */

__global__ void kSamplingPatches(const real* imgs, real* targets,
                                 int imgSize, int tgtSize, const int channels,
                                 int samplingRate, const real* thetas,
                                 const real* scales, const int* centerRs,
                                 const int* centerCs, const real padValue,
                                 const int numImages) {
  const int caseIdx = blockIdx.x * 4 + threadIdx.x;
  const int pxIdx = blockIdx.y * 128 + threadIdx.y;
  const int imgPixels = imgSize * imgSize;
  const int tgtPixels = tgtSize * tgtSize;
  const int numPatches = numImages * samplingRate;

  real tgtCenter = (tgtSize - 1) / 2;
  real imgCenter = (imgSize - 1) / 2;

  if (pxIdx < tgtPixels && caseIdx < numPatches) {
    const int imgIdx = caseIdx / samplingRate;

    // transform coordiates
    const int pxX = pxIdx % tgtSize;
    const int pxY = pxIdx / tgtSize;

    int srcPxX, srcPxY;
    getTranformCoord(pxX, pxY, thetas[imgIdx], scales[imgIdx], tgtCenter,
                     imgCenter, centerCs[caseIdx], centerRs[caseIdx], &srcPxX,
                     &srcPxY);

    imgs += (imgIdx * imgPixels + srcPxY * imgSize + srcPxX) * channels;
    targets += (caseIdx * tgtPixels + pxIdx) * channels;
    if (srcPxX >= 0 && srcPxX < imgSize && srcPxY >= 0 && srcPxY < imgSize) {
      for (int j = 0; j < channels; j++) targets[j] = imgs[j];
    } else {
      for (int j = 0; j < channels; j++) targets[j] = padValue;
    }
  }
}

/*
 * Functionality: generate the disturb (rotation and scaling) and
 *                sampling location sequence
 *
 * created by Wei Xu
 */
void hl_generate_disturb_params(real*& gpuAngle, real*& gpuScaleRatio,
                                int*& gpuCenterR, int*& gpuCenterC,
                                int numImages, int imgSize, real rotateAngle,
                                real scaleRatio, int samplingRate,
                                bool isTrain) {
  // The number of output samples.
  int numPatches = numImages * samplingRate;

  // create CPU perturbation parameters.
  real* r_angle = new real[numImages];
  real* s_ratio = new real[numImages];
  int* center_r = new int[numPatches];
  int* center_c = new int[numPatches];

  // generate the random disturbance sequence and the sampling locations
  if (isTrain) {  // random sampling for training
    // generate rotation ans scaling parameters
    // TODO(yuyang18): Since it will initialize random seed here, we can use
    // rand_r instead of rand to make this method thread safe.
    srand(getCurrentTimeStick());
    for (int i = 0; i < numImages; i++) {
      r_angle[i] =
          (rotateAngle * M_PI / 180.0) * (rand() / (RAND_MAX + 1.0)  // NOLINT
                                          - 0.5);
      s_ratio[i] =
          1 + (rand() / (RAND_MAX + 1.0) - 0.5) * scaleRatio;  // NOLINT
    }

    int imgCenter = (imgSize - 1) / 2;

    // generate sampling location parameters
    for (int i = 0; i < numImages; i++) {
      int j = 0;
      srand((unsigned)time(NULL));
      while (j < samplingRate) {
        int pxX =
            (int)(real(imgSize - 1) * rand() / (RAND_MAX + 1.0));  // NOLINT
        int pxY =
            (int)(real(imgSize - 1) * rand() / (RAND_MAX + 1.0));  // NOLINT

        const real H[4] = {cos(-r_angle[i]), -sin(-r_angle[i]),
                           sin(-r_angle[i]), cos(-r_angle[i])};
        real x = pxX - imgCenter;
        real y = pxY - imgCenter;
        real xx = H[0] * x + H[1] * y;
        real yy = H[2] * x + H[3] * y;

        real srcPxX = xx / s_ratio[i] + imgCenter;
        real srcPxY = yy / s_ratio[i] + imgCenter;

        if (srcPxX >= 0 && srcPxX <= imgSize - 1 && srcPxY >= 0 &&
            srcPxY <= imgSize - 1) {
          center_r[i * samplingRate + j] = pxY;
          center_c[i * samplingRate + j] = pxX;
          j++;
        }
      }
    }
  } else {  // central crop for testing
    for (int i = 0; i < numImages; i++) {
      r_angle[i] = 0.0;
      s_ratio[i] = 1.0;

      for (int j = 0; j < samplingRate; j++) {
        center_r[i * samplingRate + j] = (imgSize - 1) / 2;
        center_c[i * samplingRate + j] = (imgSize - 1) / 2;
      }
    }
  }

  // copy disturbance sequence to gpu
  hl_memcpy_host2device(gpuAngle, r_angle, sizeof(real) * numImages);
  hl_memcpy_host2device(gpuScaleRatio, s_ratio, sizeof(real) * numImages);

  delete[] r_angle;
  delete[] s_ratio;

  // copy sampling location sequence to gpu
  hl_memcpy_host2device(gpuCenterR, center_r, sizeof(int) * numPatches);
  hl_memcpy_host2device(gpuCenterC, center_c, sizeof(int) * numPatches);

  delete[] center_r;
  delete[] center_c;
}

void hl_conv_random_disturb_with_params(const real* images, int imgSize,
                                        int tgtSize, int channels,
                                        int numImages, int samplingRate,
                                        const real* gpuRotationAngle,
                                        const real* gpuScaleRatio,
                                        const int* gpuCenterR,
                                        const int* gpuCenterC,
                                        int paddingValue,
                                        real* target) {
  // The number of output samples.
  int numPatches = numImages * samplingRate;
  // The memory size of one output patch.
  int targetSize = tgtSize * tgtSize;

  dim3 threadsPerBlock(4, 128);
  dim3 numBlocks(DIVUP(numPatches, 4), DIVUP(targetSize, 128));

  kSamplingPatches <<<numBlocks, threadsPerBlock>>>
      (images, target, imgSize, tgtSize, channels, samplingRate,
      gpuRotationAngle, gpuScaleRatio, gpuCenterR, gpuCenterC,
      paddingValue, numImages);

  hl_device_synchronize();
}

void hl_conv_random_disturb(const real* images, int imgSize,
                            int tgtSize, int channels, int numImages,
                            real scaleRatio, real rotateAngle,
                            int samplingRate, real* gpu_r_angle,
                            real* gpu_s_ratio, int* gpu_center_r,
                            int* gpu_center_c, int paddingValue,
                            bool isTrain, real* targets) {
  // generate the random disturbance sequence and the sampling locations
  hl_generate_disturb_params(gpu_r_angle, gpu_s_ratio, gpu_center_r,
                  gpu_center_c, numImages, imgSize, rotateAngle,
                  scaleRatio, samplingRate, isTrain);

  hl_conv_random_disturb_with_params(
                  images, imgSize, tgtSize, channels, numImages,
                  samplingRate, gpu_r_angle, gpu_s_ratio,
                  gpu_center_r, gpu_center_r, paddingValue,
                  targets);
}
