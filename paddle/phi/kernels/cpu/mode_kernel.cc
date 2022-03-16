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

#include "paddle/phi/backends/cpu/cpu_context.h"
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
  const auto& in_dims = x.dims();
  auto out_dims = out->dims();
  // axis < 0, cacluate the real axis
  if (axis < 0) axis += in_dims.size();

  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);
  // if axis is not the last dim, transpose it to the last dim, do the
  // calculation, then tranpose it back to original axis.
  if (axis == in_dims.size() - 1) {
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];
    funcs::GetMode<T, int64_t>(input_height,
                               input_width,
                               in_dims.size(),
                               &x,
                               output_data,
                               indices_data);
  } else {
    std::vector<int> trans_axis;
    for (int i = 0; i < axis; i++) {
      trans_axis.emplace_back(i);
    }
    trans_axis.push_back(in_dims.size() - 1);
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

    // get the trans input_dims, out_dims
    DDim trans_shape(in_dims);
    DDim trans_out_shape(in_dims);

    for (size_t i = 0; i < trans_axis.size(); i++) {
      trans_shape[i] = in_dims[trans_axis[i]];
      trans_out_shape[i] = in_dims[trans_axis[i]];
    }
    trans_out_shape[in_dims.size() - 1] = 1;

    DenseTensor trans_input;
    trans_input.Resize(trans_shape);
    dev_ctx.template Alloc<T>(&trans_input);
    int ndims = trans_axis.size();

    // transpose the input value
    funcs::TransCompute<CPUContext, T>(
        ndims, dev_ctx, x, &trans_input, trans_axis);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_shape, 0, trans_shape.size() - 1));
    const int64_t input_width = trans_shape[trans_shape.size() - 1];
    DenseTensor tmp_out;
    tmp_out.Resize(trans_out_shape);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);

    DenseTensor tmp_indices;
    tmp_indices.Resize(trans_out_shape);
    int64_t* t_ind = dev_ctx.template Alloc<int64_t>(&tmp_indices);

    funcs::GetMode<T, int64_t>(
        input_height, input_width, in_dims.size(), &trans_input, t_out, t_ind);
    // transpose back
    funcs::TransCompute<CPUContext, int64_t>(
        ndims, dev_ctx, tmp_indices, indices, trans_axis);
    funcs::TransCompute<CPUContext, T>(
        ndims, dev_ctx, tmp_out, out, trans_axis);
    if (!keepdim) {
      out->Resize(out_dims);
      indices->Resize(out_dims);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    mode, CPU, ALL_LAYOUT, phi::ModeKernel, float, double, int32_t, int64_t) {}
