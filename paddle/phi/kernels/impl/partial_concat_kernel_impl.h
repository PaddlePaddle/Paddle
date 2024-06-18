// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/partial_concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"

namespace phi {

template <typename T, typename Context>
void PartialConcatKernel(const Context& dev_ctx,
                         const std::vector<const DenseTensor*>& x,
                         int start_index,
                         int length,
                         DenseTensor* out) {
  auto ins = x;
  PADDLE_ENFORCE_EQ(ins[0] != nullptr,
                    true,
                    phi::errors::InvalidArgument(
                        "The input of partial concat should not be null."));

  auto input_dim = ins[0]->dims();
  PADDLE_ENFORCE_EQ(input_dim.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "Only supports 2-D array with batch size in the 1st "
                        "dimension and data in the 2nd."));
  auto in_size = input_dim[1];

  // may be negative
  start_index = ComputeStartIndex(start_index, in_size);

  auto partial_len = length;
  if (partial_len < 0) {
    partial_len = in_size - start_index;
  }

  int batch = input_dim[0];
  int out_size = partial_len * ins.size();
  out->Resize({batch, out_size});
  T* out_data = dev_ctx.template Alloc<T>(out);

  for (size_t i = 0; i < ins.size(); ++i) {
    for (int j = 0; j < batch; ++j) {
      const T* in_data = ins[i]->data<T>();
      memcpy(out_data + out_size * j + partial_len * i,
             in_data + in_size * j + start_index,
             partial_len * sizeof(T));
    }
  }
}

template <typename T, typename Context>
void PartialConcatGradientOpKernel(const Context& dev_ctx,
                                   const std::vector<const DenseTensor*>& x,
                                   const DenseTensor& out_grad,
                                   int start_index,
                                   int length,
                                   std::vector<DenseTensor*> x_grad) {
  auto ins = x;
  auto outs = x_grad;

  PADDLE_ENFORCE_EQ(ins[0] != nullptr,
                    true,
                    phi::errors::InvalidArgument(
                        "The input of partial concat should not be null."));
  // all parameters
  auto batch_size = ins[0]->dims()[0];
  auto in_size = ins[0]->dims()[1];
  // may be negative
  start_index = ComputeStartIndex(start_index, in_size);
  auto partial_len = length;
  if (partial_len < 0) partial_len = in_size - start_index;

  auto in_num = ins.size();
  auto grad_batch_len = partial_len * in_num;
  auto all_length = grad_batch_len * batch_size;

  // initialize
  auto& place = *dev_ctx.eigen_device();
  for (size_t i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
    auto dxt = phi::EigenVector<T>::Flatten(*outs[i]);
    dxt.device(place) = dxt.constant(static_cast<T>(0));
  }

  auto* out_grad_t = out_grad.data<T>();
  for (size_t id = 0; id < all_length; id += partial_len) {
    int bs_id = id / grad_batch_len;
    int bs_index = id % grad_batch_len;
    int var_id = bs_index / partial_len;
    auto* out_t = outs[var_id]->data<T>();
    memcpy(out_t + bs_id * in_size + start_index,
           out_grad_t + id,
           partial_len * sizeof(T));
  }
}
}  // namespace phi
