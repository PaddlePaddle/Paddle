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

#include "MathUtils.h"
#include <algorithm>
#include "Vector.h"
#include "paddle/utils/Logging.h"

namespace paddle {

/*if csc, major is cols and minor is rows, else
 * major is rows and minor is cols, according to
 * major value to initialize minor value"
 */
void sparseRand(
    int* major, int* minor, int nnz, int majorLen, int minorMax, bool useGpu) {
  CHECK(size_t(nnz) > size_t(1));
  int* cpuMajor;
  int* cpuMinor;
  CpuIVector cpuMinorVec(nnz);
  CpuIVector cpuMajorVec(majorLen);
  if (useGpu) {
    cpuMajor = cpuMajorVec.getData();
    cpuMinor = cpuMinorVec.getData();
  } else {
    cpuMajor = major;
    cpuMinor = minor;
  }

  /*major value init*/
  for (int i = 0; i < majorLen - 1; i++) {
    cpuMajor[i] = 1.0 * i * nnz / (majorLen - 1);
  }
  cpuMajor[majorLen - 1] = nnz;

  /*minor value init according to major value*/
  std::vector<char> used(minorMax, 0);
  for (int i = 0; i < majorLen - 1; i++) {
    CHECK_LE(cpuMajor[i + 1] - cpuMajor[i], minorMax);
    used.assign(minorMax, 0);
    for (int j = cpuMajor[i]; j < cpuMajor[i + 1]; j++) {
      int idx = ::rand() % minorMax;
      while (used[idx]) {
        idx = ::rand() % minorMax;
      }
      cpuMinor[j] = idx;
      used[idx] = 1;
    }
    std::sort(cpuMinor + cpuMajor[i],
              cpuMinor + cpuMajor[i + 1],
              [](int a, int b) { return a < b; });
  }
  /*memcpy result to gpu*/
  if (useGpu) {
    hl_memcpy_host2device(major, cpuMajor, sizeof(int) * majorLen);
    hl_memcpy_host2device(minor, cpuMinor, sizeof(int) * nnz);
  }
}

int outputSize(
    int imageSize, int filterSize, int padding, int stride, bool caffeMode) {
  int outputSize;
  if (!caffeMode) {
    outputSize =
        (imageSize - filterSize + 2 * padding + stride - 1) / stride + 1;
  } else {
    outputSize = (imageSize - filterSize + 2 * padding) / stride + 1;
  }
  CHECK_GE(outputSize, 1);
  return outputSize;
}

int imageSize(
    int outputSize, int filterSize, int padding, int stride, bool caffeMode) {
  int imageSize;
  if (!caffeMode) {
    imageSize =
        (outputSize - 1) * stride + filterSize - 2 * padding - stride + 1;
  } else {
    imageSize = (outputSize - 1) * stride + filterSize - 2 * padding;
  }
  CHECK_GE(imageSize, 1);
  return imageSize;
}

}  // namespace paddle
