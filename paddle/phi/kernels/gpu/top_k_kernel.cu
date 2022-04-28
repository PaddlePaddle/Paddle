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
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace ops = paddle::operators;

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

template <typename T, typename Context>
void TopkKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                DenseTensor* out,
                DenseTensor* indices) {
  const auto* input = &x;
  // get the input dims
  const auto& in_dims = input->dims();
  // calcluate the real axis
  if (axis < 0) axis += in_dims.size();

  int k = k_scalar.to<int>();
  if (k_scalar.FromTensor()) {
    phi::DDim out_dims = out->dims();
    out_dims[axis] = k;
    out->Resize(out_dims);
    indices->Resize(out_dims);
  }

  const auto& out_dims = out->dims();

  const T* input_data = input->data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  if (axis == in_dims.size() - 1) {
    // if get the topK from the last axis
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];

    if (k > input_width) {
      k = input_width;
    }

    // The conclusion is drawn from the data through multiple sets of
    // statistics
    if (input_width >= 128 && k >= input_width * 0.75) {
      auto* ctx = reinterpret_cast<const paddle::platform::CUDADeviceContext*>(
          &dev_ctx);
      if (ops::SortTopk<T>(*ctx,
                           input,
                           input_width,
                           input_height,
                           k,
                           out,
                           indices,
                           largest)) {
        // Successed, return.
        return;
      } else {
        VLOG(4) << "TopKOP: Some errors happened when use cub sorting, use "
                   "default topk kernel.";
      }
    }

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 9000
    if (input_width >= 1024 && in_dims.size() == 1) {
      // 1. Gather TopK, but without sorting
      constexpr int max_num_threads = 1024;
      if (largest) {
        ops::RadixTopK<
            T,
            true><<<input_height, max_num_threads, 0, dev_ctx.stream()>>>(
            input_data,
            k,
            input_height,
            input_width,
            output_data,
            indices_data);
      } else {
        ops::RadixTopK<
            T,
            false><<<input_height, max_num_threads, 0, dev_ctx.stream()>>>(
            input_data,
            k,
            input_height,
            input_width,
            output_data,
            indices_data);
      }
      // 2. Sort if needed
      if (sorted) {
        DenseTensor sorted_output;
        DenseTensor sorted_indices;
        DenseTensor gather_indices;
        sorted_output.Resize(out->dims());
        sorted_indices.Resize(indices->dims());
        gather_indices.Resize(indices->dims());
        dev_ctx.template Alloc<T>(&sorted_output);
        dev_ctx.template Alloc<int64_t>(&sorted_indices);
        dev_ctx.template Alloc<int64_t>(&gather_indices);
        auto* ctx =
            reinterpret_cast<const paddle::platform::CUDADeviceContext*>(
                &dev_ctx);
        if (ops::SortTopk<T>(*ctx,
                             out,
                             k,
                             input_height,
                             k,
                             &sorted_output,
                             &sorted_indices,
                             largest)) {
          funcs::GPUGather<int64_t, int64_t>(
              dev_ctx, *indices, sorted_indices, &gather_indices);
          Copy(dev_ctx, gather_indices, indices->place(), false, indices);
          Copy(dev_ctx, sorted_output, out->place(), false, out);
          return;
        } else {
          VLOG(4) << "TopKOP: Some errors happened when use cub sorting, use "
                     "default topk kernel.";
        }
      } else {
        return;
      }
    }
#endif

    // NOTE: pass lds and dim same to input width.
    // NOTE: old matrix implementation of stride is different to eigen.
    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    switch (ops::GetDesiredBlockDim(input_width)) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<
                      T,
                      20,
                      kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
          output_data,
          k,
          indices_data,
          input_data,
          input_width,
          input_width,
          static_cast<int>(k),
          gridx,
          input_height,
          largest));
#else
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<
                      T,
                      5,
                      kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
          output_data,
          k,
          indices_data,
          input_data,
          input_width,
          input_width,
          static_cast<int>(k),
          gridx,
          input_height,
          largest));
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
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    if (k > input_width) k = input_width;

    // The conclusion is drawn from the data through multiple sets of
    // statistics
    if (input_width >= 128 && k >= input_width * 0.75) {
      auto* ctx = reinterpret_cast<const paddle::platform::CUDADeviceContext*>(
          &dev_ctx);
      if (ops::SortTopk<T>(*ctx,
                           &trans_input,
                           input_width,
                           input_height,
                           k,
                           &trans_out,
                           &trans_ind,
                           largest)) {
        // last step, tranpose back the indices and output
        funcs::TransCompute<phi::GPUContext, int64_t>(
            ndims, dev_ctx, trans_ind, indices, trans);
        funcs::TransCompute<phi::GPUContext, T>(
            ndims, dev_ctx, trans_out, out, trans);
        return;
      } else {
        VLOG(4) << "TopKOP: Some errors happened when use cub sorting, use "
                   "default topk kernel.";
      }
    }

    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    switch (ops::GetDesiredBlockDim(input_width)) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<
                      T,
                      20,
                      kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
          trans_out.data<T>(),
          k,
          trans_ind.data<int64_t>(),
          trans_input.data<T>(),
          input_width,
          input_width,
          static_cast<int>(k),
          gridx,
          input_height,
          largest));
#else
      FIXED_BLOCK_DIM(ops::KeMatrixTopK<
                      T,
                      5,
                      kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
          trans_out.data<T>(),
          k,
          trans_ind.data<int64_t>(),
          trans_input.data<T>(),
          input_width,
          input_width,
          static_cast<int>(k),
          gridx,
          input_height,
          largest));
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

}  // namespace phi

PD_REGISTER_KERNEL(top_k,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopkKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
