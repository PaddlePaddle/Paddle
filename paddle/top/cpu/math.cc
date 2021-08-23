//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/top/cpu/math.h"

namespace pt {}  // namespace pt

// Register method 1:
// PT_REGISTER_STANDARD_KERNEL(sign, CPU, NCHW, FLOAT32,
// PT_KERNEL(pt::Sign<float>))
//   .Input(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32))
//   .Output(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32));
// PT_TOUCH_KERNEL_REGISTRAR(sign, CPU, NCHW, FLOAT32);

// Register method 2:
// PT_REGISTER_KERNEL_AUTO_SPECIALIZE(sign, CPU, NCHW, pt::Sign, float)
//   .Input(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32))
//   .Output(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32));
// PT_TOUCH_KERNEL_REGISTRAR(sign, CPU, NCHW, FLOAT32);

// Register method 3:
PT_REGISTER_KERNEL_2T(sign, CPU, NCHW, pt::Sign, float, double);
