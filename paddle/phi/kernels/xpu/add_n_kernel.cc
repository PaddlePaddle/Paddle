// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/add_n_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const TensorBase*>& x,
                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  size_t in_num = x.size();
  bool in_place = false;
  if (x.size() > 0 && x[0]->initialized() && DenseTensor::classof(x[0])) {
    if ((static_cast<const DenseTensor*>(x[0]))->Holder() == out->Holder()) {
      in_place = true;
    }
  }

  if (!in_place) {
    dev_ctx.template Alloc<T>(out);
  }
  std::vector<const XPUType*> ptrs;
  for (size_t i = 0; i < in_num; ++i) {
    PADDLE_ENFORCE_EQ(DenseTensor::classof(x[i]),
                      true,
                      errors::InvalidArgument("XPU only support DensorTensor"));

    auto& in_t = *(static_cast<const DenseTensor*>(x[i]));
    if (in_t.numel() == 0) {
      continue;
    }
    ptrs.push_back(reinterpret_cast<const XPUType*>(in_t.data<T>()));
  }
  int r = xpu::sum(dev_ctx.x_context(),
                   ptrs,
                   reinterpret_cast<XPUType*>(out->data<T>()),
                   out->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sum");
}

template <typename T, typename Context>
void AddNArrayKernel(const Context& dev_ctx,
                     const std::vector<const TensorArray*>& x,
                     TensorArray* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  for (auto& ele : *out) {
    dev_ctx.template Alloc<T>(&ele);
  }
  bool in_place = true;
  if (x.size() > 0 && x[0]->size() == out->size()) {
    for (size_t i = 0; i < out->size(); i++) {
      if (x[0]->at(i).IsInitialized() &&
          out->at(i).data() != x[0]->at(i).data()) {
        in_place = false;
        break;
      }
    }
  } else {
    in_place = false;
  }

  for (size_t i = in_place ? 1 : 0; i < x.size(); ++i) {
    auto* in_array = x.at(i);

    for (size_t j = 0; j < in_array->size(); ++j) {
      if (in_array->at(j).IsInitialized() && (in_array->at(j).numel() != 0)) {
        if (j >= out->size()) {
          out->resize(j + 1);
        }
        if (!out->at(j).IsInitialized() || (out->at(j).numel() == 0)) {
          Copy<Context>(dev_ctx,
                        in_array->at(j),
                        in_array->at(j).place(),
                        false,
                        &out->at(j));
          out->at(j).set_lod(in_array->at(j).lod());
        } else {
          PADDLE_ENFORCE_EQ(
              out->at(j).lod(),
              in_array->at(j).lod(),
              phi::errors::InvalidArgument(
                  "The lod message between inputs[%d] and"
                  " outputs[%d] must be same, but now is not same.",
                  j,
                  j));

          std::vector<const XPUType*> ptrs;
          ptrs.push_back(
              reinterpret_cast<const XPUType*>(in_array->at(j).data<T>()));
          ptrs.push_back(
              reinterpret_cast<const XPUType*>(out->at(j).data<T>()));

          // int sum(Context* ctx, const std::vector<const T*>& x_list, T*
          // y, int len);
          int r = xpu::sum(dev_ctx.x_context(),
                           ptrs,
                           reinterpret_cast<XPUType*>(out->at(j).data<T>()),
                           out->at(j).numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "sum");
        }
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    add_n, XPU, ALL_LAYOUT, phi::AddNKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(add_n_array,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddNArrayKernel,
                   float,
                   phi::dtype::float16) {}
