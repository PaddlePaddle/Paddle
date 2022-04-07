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

#include "paddle/phi/kernels/mode_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/mode.h"

namespace phi {

template <typename T, typename Context>
void ModeKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int axis,
                bool keepdim,
                DenseTensor* out,
                DenseTensor* indices) {
  // get the input dims
  const auto& in_dims = x.dims();
  // calcluate the real axis
  if (axis < 0) axis += in_dims.size();

  auto out_dims = out->dims();

  const T* input_data = x.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  if (axis == in_dims.size() - 1) {
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];
    funcs::GetModebySort<T>(
        dev_ctx, &x, input_width, input_height, output_data, indices_data);
  } else {
    std::vector<int> trans_axis;
    for (int i = 0; i < axis; i++) {
      trans_axis.emplace_back(i);
    }
    trans_axis.emplace_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans_axis.emplace_back(i);
    }
    trans_axis.emplace_back(axis);

    if (!keepdim) {
      std::vector<int> tmp_out_shape;
      for (int i = 0; i < axis; i++) {
        tmp_out_shape.emplace_back(in_dims[i]);
      }
      tmp_out_shape.emplace_back(1);
      for (int i = axis + 1; i < in_dims.size(); i++) {
        tmp_out_shape.emplace_back(in_dims[i]);
      }
      DDim tmp_out_dim = phi::make_ddim(tmp_out_shape);
      out->Resize(tmp_out_dim);
      indices->Resize(tmp_out_dim);
    }

    DDim trans_shape(in_dims);
    DDim trans_out_shape(in_dims);
    for (int i = 0; i < trans_axis.size(); i++) {
      trans_shape[i] = in_dims[trans_axis[i]];
      trans_out_shape[i] = in_dims[trans_axis[i]];
    }
    trans_out_shape[in_dims.size() - 1] = 1;

    // second step, tranpose the input
    DenseTensor trans_input;
    trans_input.Resize(trans_shape);
    dev_ctx.template Alloc<T>(&trans_input);

    int ndims = trans_axis.size();
    funcs::TransCompute<Context, T>(
        ndims, dev_ctx, x, &trans_input, trans_axis);
    DenseTensor trans_ind;
    trans_ind.Resize(trans_out_shape);
    int64_t* trans_ind_data = dev_ctx.template Alloc<int64_t>(&trans_ind);

    DenseTensor trans_out;
    trans_out.Resize(trans_out_shape);
    T* trans_out_data = dev_ctx.template Alloc<T>(&trans_out);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_shape, 0, trans_shape.size() - 1));
    const int64_t input_width = trans_shape[trans_shape.size() - 1];
    funcs::GetModebySort<T>(dev_ctx,
                            &trans_input,
                            input_width,
                            input_height,
                            trans_out_data,
                            trans_ind_data);
    // last step, tranpose back the indices and output
    funcs::TransCompute<Context, int64_t>(
        ndims, dev_ctx, trans_ind, indices, trans_axis);
    funcs::TransCompute<Context, T>(ndims, dev_ctx, trans_out, out, trans_axis);
    if (!keepdim) {
      out->Resize(out_dims);
      indices->Resize(out_dims);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    mode, GPU, ALL_LAYOUT, phi::ModeKernel, float, double, int32_t, int64_t) {}
