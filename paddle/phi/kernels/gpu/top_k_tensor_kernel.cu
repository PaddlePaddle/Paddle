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

#include "paddle/phi/kernels/top_k_kernel.h"

#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace ops = paddle::operators;

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_MAXLENGTH_BASE(MaxLength, ...) \
  case (MaxLength): {                        \
    constexpr auto maxLength = (MaxLength);  \
    __VA_ARGS__;                             \
  } break

#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);   \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

#define FIXED_MAXLENGTH(...)              \
  FIXED_MAXLENGTH_BASE(1, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(2, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(3, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(4, ##__VA_ARGS__); \
  FIXED_MAXLENGTH_BASE(5, ##__VA_ARGS__)

template <typename T, typename Context>
void TopKTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& k_list,
                      int axis,
                      bool largest,
                      DenseTensor* out,
                      DenseTensor* indices) {
  const auto* input = &x;
  // get the input dims
  const auto& in_dims = input->dims();
  // calcluate the real axis
  if (axis < 0) axis += in_dims.size();
  const int64_t& input_width = in_dims[axis];

  DenseTensor k_largest_tensor;
  phi::DDim k_largest_dim = phi::make_ddim({1});
  k_largest_tensor.Resize(k_largest_dim);
  dev_ctx.template Alloc<int>(&k_largest_tensor);
  int* k_largest_data = k_largest_tensor.data<int>();

  ops::getMaxK<int, 256><<<1, 256, 0, dev_ctx.stream()>>>(
      k_list.data<int>(), k_largest_data, k_list.numel());

  DenseTensor k_largest_host;
  phi::CPUPlace cpu;
  phi::Copy(dev_ctx, k_largest_tensor, cpu, false, &k_largest_host);

  int k_largest = k_largest_host.data<int>()[0];
  if (k_largest > input_width) {
    k_largest = input_width;
  }

  phi::DDim out_dims_tmp = out->dims();
  out_dims_tmp[axis] = k_largest;
  out->Resize(out_dims_tmp);
  indices->Resize(out_dims_tmp);

  int bs_size = in_dims[0];

  const auto& out_dims = out->dims();

  const T* input_data = input->data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  ops::InitVal<T, 256>
      <<<1, 256, 0, dev_ctx.stream()>>>(output_data, out->numel());
  ops::InitVal<int64_t, 256>
      <<<1, 256, 0, dev_ctx.stream()>>>(indices_data, indices->numel());
  if (axis == in_dims.size() - 1) {
    // if get the topK from the last axis
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));

    // NOTE: pass lds and dim same to input width.
    // NOTE: old matrix implementation of stride is different to eigen.
    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    int bs_offset = input_height / bs_size;
    paddle::platform::GpuLaunchConfig config =
        paddle::platform::GetGpuLaunchConfig1D(dev_ctx, input_width);
    switch (config.thread_per_block.x) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<T, 20, kBlockDim>
                      <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                          output_data,
                          k_largest,
                          indices_data,
                          input_data,
                          input_width,
                          input_width,
                          static_cast<int>(k_largest),
                          gridx,
                          input_height,
                          largest,
                          k_list.numel() > 1 ? k_list.data<int>() : nullptr,
                          bs_offset));
#else
      FIXED_BLOCK_DIM(switch (ops::getMaxLength(k_largest)) {
        FIXED_MAXLENGTH(ops::KeMatrixTopK<T, maxLength, kBlockDim>
                        <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                            output_data,
                            k_largest,
                            indices_data,
                            input_data,
                            input_width,
                            input_width,
                            static_cast<int>(k_largest),
                            gridx,
                            input_height,
                            largest,
                            k_list.numel() > 1 ? k_list.data<int>() : nullptr,
                            bs_offset));
        default:
          PADDLE_THROW(errors::Fatal(
              "the input k_largest has error when use getMaxLength "
              "function to get the maxLength."));
      });
#endif
      default:
        PADDLE_THROW(errors::Fatal(
            "the input data shape has error in the topk cuda kernel."));
    }
  } else {
    // if get topK not from the last axis, will tranpose the tensor and get
    // TopK

    // first step, prepare the trans args for the tranpose
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);

    phi::DDim trans_dims(in_dims);
    phi::DDim trans_out_dims(out->dims());
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
      trans_out_dims[i] = out_dims[trans[i]];
    }
    // second step, tranpose the input
    DenseTensor trans_input;
    trans_input.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_input);
    int ndims = trans.size();
    funcs::TransCompute<phi::GPUContext, T>(
        ndims, dev_ctx, *input, &trans_input, trans);
    // third step, calcluate the topk
    // allocate the tmp cuda memory for the tmp result
    DenseTensor trans_ind;
    DenseTensor trans_out;
    trans_ind.Resize(trans_out_dims);
    trans_out.Resize(trans_out_dims);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    dev_ctx.template Alloc<T>(&trans_out);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    int bs_offset = input_height / bs_size;
    paddle::platform::GpuLaunchConfig config =
        paddle::platform::GetGpuLaunchConfig1D(dev_ctx, input_width);
    switch (config.thread_per_block.x) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<T, 20, kBlockDim>
                      <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                          trans_out.data<T>(),
                          k_largest,
                          trans_ind.data<int64_t>(),
                          trans_input.data<T>(),
                          input_width,
                          input_width,
                          static_cast<int>(k_largest),
                          gridx,
                          input_height,
                          largest,
                          k_list.numel() > 1 ? k_list.data<int>() : nullptr,
                          bs_offset));
#else
      FIXED_BLOCK_DIM(switch (ops::getMaxLength(k_largest)) {
        FIXED_MAXLENGTH(ops::KeMatrixTopK<T, maxLength, kBlockDim>
                        <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                            trans_out.data<T>(),
                            k_largest,
                            trans_ind.data<int64_t>(),
                            trans_input.data<T>(),
                            input_width,
                            input_width,
                            static_cast<int>(k_largest),
                            gridx,
                            input_height,
                            largest,
                            k_list.numel() > 1 ? k_list.data<int>() : nullptr,
                            bs_offset));
        default:
          PADDLE_THROW(errors::Fatal(
              "the input k_largest has error when use getMaxLength "
              "function to get the maxLength."));
      });
#endif
      default:
        PADDLE_THROW(errors::Fatal(
            "the input data shape has error in the topk cuda kernel."));
    }

    // last step, tranpose back the indices and output
    funcs::TransCompute<phi::GPUContext, int64_t>(
        ndims, dev_ctx, trans_ind, indices, trans);
    funcs::TransCompute<phi::GPUContext, T>(
        ndims, dev_ctx, trans_out, out, trans);
  }
}
#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM
#undef FIXED_MAXLENGTH_BASE
#undef FIXED_MAXLENGTH

}  // namespace phi

PD_REGISTER_KERNEL(top_k_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopKTensorKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
