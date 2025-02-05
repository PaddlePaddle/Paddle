/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"

namespace phi {
namespace funcs {
#define CUDNN_PER_ACTIVATION_THRESHOLD 10240
#define CUDNN_SPATIAL_THRESHOLD_TRAIN 880801
#define CUDNN_SPATIAL_THRESHOLD_EVAL 65535

inline void ExtractNCWHD(const phi::DDim &dims,
                         const DataLayout &data_layout,
                         int *N,
                         int *C,
                         int *H,
                         int *W,
                         int *D) {
  *N = dims[0];
  if (dims.size() == 2) {
    *C = dims[1];
    *H = 1;
    *W = 1;
    *D = 1;
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == DataLayout::kNCHW ? dims[4] : dims[3])
             : 1;
  }
}
}  // namespace funcs
}  // namespace phi
