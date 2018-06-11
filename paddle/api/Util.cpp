/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PaddleAPI.h"

#include "paddle/parameter/Parameter.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Util.h"

#include <algorithm>
#include <iostream>
#include <iterator>

void initPaddle(int argc, char** argv) {
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
}

FloatArray::FloatArray(const float* b, const size_t l)
    : buf(b), length(l), needFree(false) {}

IntArray::IntArray(const int* b, const size_t l, bool f)
    : buf(b), length(l), needFree(f) {}

IntWithFloatArray::IntWithFloatArray(const float* v,
                                     const int* i,
                                     size_t l,
                                     bool f)
    : valBuf(v), idxBuf(i), length(l), needFree(f) {}

bool isUsingGpu() { return FLAGS_use_gpu; }

void setUseGpu(bool useGpu) { FLAGS_use_gpu = useGpu; }

bool isGpuVersion() {
#ifndef PADDLE_WITH_CUDA
  return false;
#else
  return true;
#endif
}

int getTrainerCount() { return FLAGS_trainer_count; }

static_assert(NUM_PARAMETER_TYPES == paddle::NUM_PARAMETER_TYPES,
              "The Parameter Type should be same in core/api and core/common");
