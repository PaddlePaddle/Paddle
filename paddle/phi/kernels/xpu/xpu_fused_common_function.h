// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
namespace phi {
struct XPUDropoutParam {
  float dropout_prob;
  bool is_upscale_in_train;
  bool is_test;
  bool fix_seed;
  const phi::DenseTensor *tensor_seed;
  int seed_val;

  XPUDropoutParam() {
    fix_seed = false;
    is_test = false;
    is_upscale_in_train = false;
    dropout_prob = 0.5;
    tensor_seed = nullptr;
    seed_val = 0;
  }

  void initXPUDropoutParam(float dropout_prob_,
                           bool is_upscale_in_train_,
                           bool is_test_,
                           bool fix_seed_,
                           const phi::DenseTensor *tensor_seed,
                           int seed_val_) {
    dropout_prob = dropout_prob_;
    is_upscale_in_train = is_upscale_in_train_;
    is_test = is_test_;
    fix_seed = fix_seed_;
    if (tensor_seed) {
      seed_val = *(tensor_seed->data<int>());
    } else {
      seed_val = fix_seed ? seed_val_ : 0;
    }
  }
};

/******************
 * check is l3
 ******************/

static bool is_in_l3(const void *addr) {
  int64_t addr_int = (int64_t)addr;
  int addr_int_high = addr_int >> 32;
  return (addr_int_high == 0);
}

/*************************
 * dropout
 *************************/

template <typename T>
void Dropout(xpu::Context *xpu_ctx,
             const T *x,
             T *mask,
             T *y,
             const XPUDropoutParam &param,
             int len) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int r = XPU_SUCCESS;
  if (param.dropout_prob == 0.0f) {
    r = xpu::copy(xpu_ctx,
                  reinterpret_cast<const XPUType *>(x),
                  reinterpret_cast<XPUType *>(y),
                  len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }
  if (!param.is_test) {
    if (param.dropout_prob == 1.0f) {
      r = xpu::constant(
          xpu_ctx, reinterpret_cast<XPUType *>(y), len, XPUType(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      r = xpu::constant(
          xpu_ctx, reinterpret_cast<XPUType *>(mask), len, XPUType(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    } else {
      r = xpu::dropout(xpu_ctx,
                       reinterpret_cast<const XPUType *>(x),
                       reinterpret_cast<XPUType *>(y),
                       reinterpret_cast<XPUType *>(mask),
                       param.seed_val,
                       len,
                       param.is_upscale_in_train,
                       param.dropout_prob);

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout");
    }
  } else {
    float scale = (param.is_upscale_in_train)
                      ? (1.0)
                      : (static_cast<float>(1.0f - param.dropout_prob));
    r = xpu::scale(xpu_ctx,
                   reinterpret_cast<const XPUType *>(x),
                   reinterpret_cast<XPUType *>(y),
                   len,
                   false,
                   scale,
                   0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}

template <typename T>
void DropoutGrad(xpu::Context *xpu_ctx,
                 const T *dy,
                 const T *mask,
                 T *dx,
                 const XPUDropoutParam &param,
                 int len) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (param.dropout_prob == 0.0f) {
    int r = xpu::copy(xpu_ctx,
                      reinterpret_cast<const XPUType *>(dy),
                      reinterpret_cast<XPUType *>(dx),
                      len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }
  if (!param.is_upscale_in_train) {
    int r = xpu::mul(xpu_ctx,
                     reinterpret_cast<const XPUType *>(dy),
                     reinterpret_cast<const XPUType *>(mask),
                     reinterpret_cast<XPUType *>(dx),
                     len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  } else {
    int r = xpu::dropout_grad(xpu_ctx,
                              reinterpret_cast<const XPUType *>(mask),
                              reinterpret_cast<const XPUType *>(dy),
                              reinterpret_cast<XPUType *>(dx),
                              param.dropout_prob,
                              len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_grad");
  }
}
}  // namespace phi
