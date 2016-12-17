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

#ifndef PADDLE_ONLY_CPU

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "hl_cuda.h"
#include "hl_perturbation_util.cuh"

using namespace std;  // NOLINT

#define _USE_MATH_DEFINES

const int NUM_IMAGES = 2;
const int SAMPLING_RATE = 2;
const int IMG_SIZE = 41;
const int TGT_SIZE = 21;
const int CHANNELS = 3;

class PerturbationTest : public testing::Test {
protected:
  virtual void SetUp() { generateTestImages(gpuImages_); }

  virtual void TearDown() {}

  void allocateMem(real*& gpuAngle,
                   real*& gpuScale,
                   int*& gpuCenterR,
                   int*& gpuCenterC) {
    gpuAngle = (real*)hl_malloc_device(sizeof(real) * NUM_IMAGES);
    gpuScale = (real*)hl_malloc_device(sizeof(real) * NUM_IMAGES);
    gpuCenterR =
        (int*)hl_malloc_device(sizeof(int) * NUM_IMAGES * SAMPLING_RATE);
    gpuCenterC =
        (int*)hl_malloc_device(sizeof(int) * NUM_IMAGES * SAMPLING_RATE);
  }

  // Generate translation parameters for testing.
  void generateTranslationParams(int*& gpuCenterR,
                                 int*& gpuCenterC,
                                 int imgSize) {
    int cpuCenterR[NUM_IMAGES * SAMPLING_RATE];
    int cpuCenterC[NUM_IMAGES * SAMPLING_RATE];
    for (int i = 0; i < NUM_IMAGES * SAMPLING_RATE; ++i) {
      cpuCenterR[i] = (imgSize - 1) / 2;
      cpuCenterC[i] = (imgSize - 1) / 2 - 1;
    }

    gpuCenterR =
        (int*)hl_malloc_device(sizeof(int) * NUM_IMAGES * SAMPLING_RATE);
    hl_memcpy_host2device(
        gpuCenterR, cpuCenterR, sizeof(int) * NUM_IMAGES * SAMPLING_RATE);

    gpuCenterC =
        (int*)hl_malloc_device(sizeof(int) * NUM_IMAGES * SAMPLING_RATE);
    hl_memcpy_host2device(
        gpuCenterC, cpuCenterC, sizeof(int) * NUM_IMAGES * SAMPLING_RATE);
  }

  // Generate rotation parameters for testing.
  void generateRotationParams(real*& gpuAngle) {
    real cpuAngle[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; ++i) {
      cpuAngle[i] = 90.0 * M_PI / 180.0;
    }
    gpuAngle = (real*)hl_malloc_device(sizeof(real) * NUM_IMAGES);
    hl_memcpy_host2device(gpuAngle, cpuAngle, sizeof(real) * NUM_IMAGES);
  }

  void generateScaleParams(real*& gpuScale) {
    real cpuScale[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; ++i) {
      cpuScale[i] = static_cast<real>(TGT_SIZE - 2) / TGT_SIZE;
    }
    gpuScale = (real*)hl_malloc_device(sizeof(real) * NUM_IMAGES);
    hl_memcpy_host2device(gpuScale, cpuScale, sizeof(real) * NUM_IMAGES);
  }

  // Generate the test images, only the center regions are set to 1.
  // The other parts are set to 0.
  void generateTestImages(real*& gpuImages) {
    const int IMAGE_MEM_SIZE = NUM_IMAGES * IMG_SIZE * IMG_SIZE * CHANNELS;
    real cpuImages[IMAGE_MEM_SIZE];
    // Set the middle of each image to 1.
    real* ptr = cpuImages;
    for (int i = 0; i < NUM_IMAGES; ++i) {
      for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; ++c) {
          for (int ch = 0; ch < CHANNELS; ++ch) {
            if (r >= IMG_SIZE / 4 && r < IMG_SIZE - IMG_SIZE / 4 &&
                c >= IMG_SIZE / 4 && c < IMG_SIZE - IMG_SIZE / 4) {
              *ptr = 1.0;
            } else {
              *ptr = 0.0;
            }
            ++ptr;
          }
        }
      }
    }
    gpuImages = (real*)hl_malloc_device(sizeof(real) * IMAGE_MEM_SIZE);
    hl_memcpy_host2device(gpuImages, cpuImages, sizeof(real) * IMAGE_MEM_SIZE);
  }

  real* gpuImages_;
};

// Random perturbation. Only to make sure the code does not break.
TEST_F(PerturbationTest, random_perturb) {
  real *gpuAngle, *gpuScaleRatio;
  int *gpuCenterR, *gpuCenterC;
  allocateMem(gpuAngle, gpuScaleRatio, gpuCenterR, gpuCenterC);

  real* targets = NULL;
  const int TARGET_MEM_SIZE =
      NUM_IMAGES * SAMPLING_RATE * TGT_SIZE * TGT_SIZE * CHANNELS;
  targets = (real*)hl_malloc_device(sizeof(real) * TARGET_MEM_SIZE);
  hl_conv_random_disturb(gpuImages_,
                         IMG_SIZE,
                         TGT_SIZE,
                         CHANNELS,
                         NUM_IMAGES,
                         1.0,
                         1.0,
                         SAMPLING_RATE,
                         gpuAngle,
                         gpuScaleRatio,
                         gpuCenterR,
                         gpuCenterC,
                         2,
                         true,
                         targets);
  real cpuTargets[TARGET_MEM_SIZE];
  hl_memcpy_device2host(cpuTargets, targets, sizeof(real) * TARGET_MEM_SIZE);
}

TEST_F(PerturbationTest, identity_perturb) {
  real *gpuAngle, *gpuScaleRatio;
  int *gpuCenterR, *gpuCenterC;
  allocateMem(gpuAngle, gpuScaleRatio, gpuCenterR, gpuCenterC);

  real* targets = NULL;
  const int TARGET_MEM_SIZE =
      NUM_IMAGES * SAMPLING_RATE * TGT_SIZE * TGT_SIZE * CHANNELS;
  targets = (real*)hl_malloc_device(sizeof(real) * TARGET_MEM_SIZE);
  hl_conv_random_disturb(gpuImages_,
                         IMG_SIZE,
                         TGT_SIZE,
                         CHANNELS,
                         NUM_IMAGES,
                         1.0,
                         1.0,
                         SAMPLING_RATE,
                         gpuAngle,
                         gpuScaleRatio,
                         gpuCenterR,
                         gpuCenterC,
                         2,
                         false,
                         targets);
  real cpuTargets[TARGET_MEM_SIZE];
  hl_memcpy_device2host(cpuTargets, targets, sizeof(real) * TARGET_MEM_SIZE);
  for (int i = 0; i < TARGET_MEM_SIZE; ++i) {
    EXPECT_FLOAT_EQ(1.0, cpuTargets[i]);
  }
}

TEST_F(PerturbationTest, translation_test) {
  real *gpuAngle, *gpuScaleRatio;
  int *gpuCenterR, *gpuCenterC;
  allocateMem(gpuAngle, gpuScaleRatio, gpuCenterR, gpuCenterC);
  hl_generate_disturb_params(gpuAngle,
                             gpuScaleRatio,
                             gpuCenterR,
                             gpuCenterC,
                             NUM_IMAGES,
                             IMG_SIZE,
                             0.0,
                             0.0,
                             SAMPLING_RATE,
                             false);
  generateTranslationParams(gpuCenterR, gpuCenterC, IMG_SIZE);

  real* targets = NULL;
  const int TARGET_MEM_SIZE =
      NUM_IMAGES * SAMPLING_RATE * TGT_SIZE * TGT_SIZE * CHANNELS;
  targets = (real*)hl_malloc_device(sizeof(real) * TARGET_MEM_SIZE);
  hl_conv_random_disturb_with_params(gpuImages_,
                                     IMG_SIZE,
                                     TGT_SIZE,
                                     CHANNELS,
                                     NUM_IMAGES,
                                     SAMPLING_RATE,
                                     gpuAngle,
                                     gpuScaleRatio,
                                     gpuCenterR,
                                     gpuCenterC,
                                     2,
                                     targets);

  real cpuTargets[TARGET_MEM_SIZE];
  hl_memcpy_device2host(cpuTargets, targets, sizeof(real) * TARGET_MEM_SIZE);
  for (int i = 0; i < SAMPLING_RATE * NUM_IMAGES; ++i) {
    for (int p = 0; p < TGT_SIZE * TGT_SIZE * CHANNELS; ++p) {
      const int offset = i * TGT_SIZE * TGT_SIZE * CHANNELS + p;
      if (p < TGT_SIZE * CHANNELS) {
        EXPECT_FLOAT_EQ(0.0, cpuTargets[offset]);
      } else {
        EXPECT_FLOAT_EQ(1.0, cpuTargets[offset]);
      }
    }
  }
}

TEST_F(PerturbationTest, rotation_test) {
  real *gpuAngle, *gpuScaleRatio;
  int *gpuCenterR, *gpuCenterC;
  allocateMem(gpuAngle, gpuScaleRatio, gpuCenterR, gpuCenterC);
  hl_generate_disturb_params(gpuAngle,
                             gpuScaleRatio,
                             gpuCenterR,
                             gpuCenterC,
                             NUM_IMAGES,
                             IMG_SIZE,
                             0.0,
                             0.0,
                             SAMPLING_RATE,
                             false);
  generateRotationParams(gpuAngle);

  real* targets = NULL;
  const int TARGET_MEM_SIZE =
      NUM_IMAGES * SAMPLING_RATE * TGT_SIZE * TGT_SIZE * CHANNELS;
  targets = (real*)hl_malloc_device(sizeof(real) * TARGET_MEM_SIZE);
  hl_conv_random_disturb_with_params(gpuImages_,
                                     IMG_SIZE,
                                     TGT_SIZE,
                                     CHANNELS,
                                     NUM_IMAGES,
                                     SAMPLING_RATE,
                                     gpuAngle,
                                     gpuScaleRatio,
                                     gpuCenterR,
                                     gpuCenterC,
                                     2,
                                     targets);

  real cpuTargets[TARGET_MEM_SIZE];
  hl_memcpy_device2host(cpuTargets, targets, sizeof(real) * TARGET_MEM_SIZE);
  for (int i = 0; i < TARGET_MEM_SIZE; ++i) {
    EXPECT_FLOAT_EQ(1.0, cpuTargets[i]);
  }
}

TEST_F(PerturbationTest, scale_test) {
  real *gpuAngle, *gpuScaleRatio;
  int *gpuCenterR, *gpuCenterC;
  allocateMem(gpuAngle, gpuScaleRatio, gpuCenterR, gpuCenterC);
  hl_generate_disturb_params(gpuAngle,
                             gpuScaleRatio,
                             gpuCenterR,
                             gpuCenterC,
                             NUM_IMAGES,
                             IMG_SIZE,
                             0.0,
                             0.0,
                             SAMPLING_RATE,
                             false);
  generateScaleParams(gpuScaleRatio);

  real* targets = NULL;
  const int TARGET_MEM_SIZE =
      NUM_IMAGES * SAMPLING_RATE * TGT_SIZE * TGT_SIZE * CHANNELS;
  targets = (real*)hl_malloc_device(sizeof(real) * TARGET_MEM_SIZE);
  hl_conv_random_disturb_with_params(gpuImages_,
                                     IMG_SIZE,
                                     TGT_SIZE,
                                     CHANNELS,
                                     NUM_IMAGES,
                                     SAMPLING_RATE,
                                     gpuAngle,
                                     gpuScaleRatio,
                                     gpuCenterR,
                                     gpuCenterC,
                                     2,
                                     targets);

  real cpuTargets[TARGET_MEM_SIZE];
  hl_memcpy_device2host(cpuTargets, targets, sizeof(real) * TARGET_MEM_SIZE);
  for (int i = 0; i < SAMPLING_RATE * NUM_IMAGES; ++i) {
    for (int p = 0; p < TGT_SIZE * TGT_SIZE * CHANNELS; ++p) {
      const int offset = i * TGT_SIZE * TGT_SIZE * CHANNELS + p;
      int c = (p / CHANNELS) % TGT_SIZE;
      int r = (p / CHANNELS) / TGT_SIZE;
      if (r == 0 || r == TGT_SIZE - 1 || c == 0 || c == TGT_SIZE - 1) {
        EXPECT_FLOAT_EQ(0.0, cpuTargets[offset]);
      } else {
        EXPECT_FLOAT_EQ(1.0, cpuTargets[offset]);
      }
    }
  }
}

#endif
