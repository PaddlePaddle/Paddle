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
#include "paddle/phi/kernels/squeeze_kernel.h"

#include <set>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void SqueezeStridedKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          const IntArray& axes_arr,
                          DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> output_dims;
  std::vector<int64_t> output_stride;

  std::set<int64_t> axes_set;

  auto input_dims = input.dims();
  auto input_stride = input.strides();

  if (input.Holder() == out->Holder() && input.meta() == out->meta()) {
    output_dims = common::vectorize<int64_t>(out->dims());
    if (axes.empty()) {
      for (int i = input_stride.size() - 1; i > 0; --i) {
        if (input_stride[i] != input_stride[i - 1]) {
          output_stride.insert(output_stride.begin(), input_stride[i]);
        }
      }
      if (output_dims.size() > output_stride.size()) {
        output_stride.insert(output_stride.begin(), input_stride[0]);
      }
    } else {
      for (auto& item : axes) {
        item = item < 0 ? item + input_stride.size() : item;
        if (item != 0 && input_stride[static_cast<int>(item)] ==
                             input_stride[static_cast<int>(item) - 1]) {
          axes_set.insert(item);
        }
      }
      for (int i = 0; i < input_stride.size(); i++) {
        if (axes_set.count(i) == 0) {
          output_stride.push_back(input_stride[i]);
        }
      }
      if (output_dims.size() < output_stride.size()) {
        output_stride.erase(output_stride.begin());
      }
    }

    auto meta = out->meta();
    meta.offset = input.offset();
    meta.strides =
        DDim(output_stride.data(), static_cast<int>(output_stride.size()));
    out->set_meta(meta);
    return;
  }

  if (axes.empty()) {
    for (int i = 0; i < input_dims.size(); i++) {
      if (input_dims[i] != 1) {
        output_dims.push_back(input_dims[i]);
        output_stride.push_back(input_stride[i]);
      }
    }
  } else {
    for (auto item : axes) {
      auto axis = item < 0 ? item + input_dims.size() : item;
      if (input_dims[static_cast<int>(axis)] == 1) {
        axes_set.insert(axis);
      }
    }

    for (int i = 0; i < input_dims.size(); i++) {
      if (axes_set.count(i) == 0) {
        output_dims.push_back(input_dims[i]);
        output_stride.push_back(input_stride[i]);
      }
    }
  }

  auto meta = out->meta();
  auto tmp_dim = DDim(output_dims.data(), static_cast<int>(output_dims.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("Unsqueeze kernel stride compute diff, infer
  //       shape"
  //                          "is %s, but compute is %s.",
  //                          meta.dims,
  //                          tmp_dim));
  // }
  meta.dims = tmp_dim;
  meta.strides =
      DDim(output_stride.data(), static_cast<int>(output_stride.size()));
  meta.offset = input.offset();
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

template <typename Context>
void SqueezeWithXShapeStridedKernel(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const IntArray& axes,
                                    DenseTensor* out,
                                    DenseTensor* xshape UNUSED) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  SqueezeStridedKernel<Context>(dev_ctx, x, axes, out);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(squeeze,
                                         STRIDED,
                                         phi::SqueezeStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(squeeze_with_xshape,
                                         STRIDED,
                                         phi::SqueezeWithXShapeStridedKernel) {}
