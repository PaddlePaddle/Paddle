// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once
#include "paddle/cinn/runtime/cinn_runtime.h"

#ifdef CINN_WITH_DNNL
#include "dnnl.hpp"  // NOLINT
#endif

// define some C APIs
extern "C" {
void cinn_cpu_onednn_softmax_fp32(int batch,
                                  int channel,
                                  int h,
                                  int w,
                                  int axis,
                                  cinn_buffer_t* inputs,
                                  cinn_buffer_t* out);

void cinn_cpu_onednn_conv2d_nchw_fp32(int batch_size,
                                      int c_in,
                                      int input_h,
                                      int input_w,
                                      int c_out,
                                      int group,
                                      int filter_h,
                                      int filter_w,
                                      int pad_h,
                                      int pad_w,
                                      int stride_h,
                                      int stride_w,
                                      int dilation_h,
                                      int dilation_w,
                                      cinn_buffer_t* inputs,
                                      cinn_buffer_t* weights,
                                      cinn_buffer_t* out);

}  // extern "C"
