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

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/top_k_cuda_kernel.h"
#endif
namespace phi {

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
void TopkKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                DenseTensor* out,
                DenseTensor* indices) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<int64_t>(indices);
    return;
  }

  const auto* input = &x;
  // get the input dims
  const auto& in_dims = input->dims();

  // 0d input tensor
  if (in_dims.size() == 0) {
    Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    dev_ctx.template Alloc<int64_t>(indices);
    funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }
  // calculate the real axis
  if (axis < 0) axis += in_dims.size();

  int k = k_scalar.to<int>();
  // out shape [-1]
  if (k_scalar.FromTensor()) {
    DDim out_dims = out->dims();
    out_dims[axis] = k;
    out->Resize(out_dims);
    indices->Resize(out_dims);
  }
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), NAN, out);
    Full<int64_t, Context>(dev_ctx, indices->dims(), 0, indices);
    return;
  }
  PADDLE_ENFORCE_GE(
      x.numel(),
      k,
      errors::InvalidArgument(
          "x has only %d element, can not find %d top values.", x.numel(), k));

  const auto& out_dims = out->dims();

  const T* input_data = input->data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  if (axis == in_dims.size() - 1) {
    // if get the topK from the last axis
    const int64_t& input_height =
        common::product(slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];

    if (k > input_width) {
      k = input_width;
    }

    // The conclusion is drawn from the data through multiple sets of
    // statistics
    if (input_width >= 128 && k >= input_width * 0.25) {
      auto* ctx = reinterpret_cast<const GPUContext*>(&dev_ctx);
      if (funcs::SortTopk<T>(*ctx,
                             input,
                             input_width,
                             input_height,
                             k,
                             out,
                             indices,
                             largest)) {
        // Succeed, return.
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
        funcs::RadixTopK<T, true>
            <<<input_height, max_num_threads, 0, dev_ctx.stream()>>>(
                input_data,
                k,
                input_height,
                input_width,
                output_data,
                indices_data);
      } else {
        funcs::RadixTopK<T, false>
            <<<input_height, max_num_threads, 0, dev_ctx.stream()>>>(
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
        auto* ctx = reinterpret_cast<const GPUContext*>(&dev_ctx);
        if (funcs::SortTopk<T>(*ctx,
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
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, input_width);
    switch (config.thread_per_block.x) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(
          funcs::KeMatrixTopK<T, 20, kBlockDim>
          <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(output_data,
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
      FIXED_BLOCK_DIM(switch (funcs::getMaxLength(k)) {
        FIXED_MAXLENGTH(
            funcs::KeMatrixTopK<T, maxLength, kBlockDim>
            <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(output_data,
                                                        k,
                                                        indices_data,
                                                        input_data,
                                                        input_width,
                                                        input_width,
                                                        static_cast<int>(k),
                                                        gridx,
                                                        input_height,
                                                        largest));
        default:
          PADDLE_THROW(
              errors::Fatal("the input k has error when use getMaxLength "
                            "function to get the maxLength."));
      });
#endif
      default:
        PADDLE_THROW(errors::Fatal(
            "the input data shape has error in the topk cuda kernel."));
    }
  } else {
    // if get topK not from the last axis, will transpose the tensor and get
    // TopK

    // first step, prepare the trans args for the transpose
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.emplace_back(i);
    }
    trans.emplace_back(axis);

    DDim trans_dims(in_dims);
    DDim trans_out_dims(out->dims());
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
      trans_out_dims[i] = out_dims[trans[i]];
    }
    // second step, transpose the input
    DenseTensor trans_input;
    trans_input.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_input);
    int ndims = trans.size();
    funcs::TransCompute<GPUContext, T>(
        ndims, dev_ctx, *input, &trans_input, trans);
    // third step, calculate the topk
    // allocate the tmp cuda memory for the tmp result
    DenseTensor trans_ind;
    DenseTensor trans_out;
    trans_ind.Resize(trans_out_dims);
    trans_out.Resize(trans_out_dims);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    dev_ctx.template Alloc<T>(&trans_out);

    const int64_t input_height =
        common::product(slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    if (k > input_width) k = input_width;

    // The conclusion is drawn from the data through multiple sets of
    // statistics
    if (input_width >= 128 && k >= input_width * 0.75) {
      auto* ctx = reinterpret_cast<const GPUContext*>(&dev_ctx);
      if (funcs::SortTopk<T>(*ctx,
                             &trans_input,
                             input_width,
                             input_height,
                             k,
                             &trans_out,
                             &trans_ind,
                             largest)) {
        // last step, transpose back the indices and output
        funcs::TransCompute<GPUContext, int64_t>(
            ndims, dev_ctx, trans_ind, indices, trans);
        funcs::TransCompute<GPUContext, T>(
            ndims, dev_ctx, trans_out, out, trans);
        return;
      } else {
        VLOG(4) << "TopKOP: Some errors happened when use cub sorting, use "
                   "default topk kernel.";
      }
    }

    const int kMaxHeight = 2048;
    int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, input_width);
    switch (config.thread_per_block.x) {
#ifdef PADDLE_WITH_HIP
      FIXED_BLOCK_DIM(
          funcs::KeMatrixTopK<T, 20, kBlockDim>
          <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(trans_out.data<T>(),
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
      FIXED_BLOCK_DIM(switch (funcs::getMaxLength(k)) {
        FIXED_MAXLENGTH(funcs::KeMatrixTopK<T, maxLength, kBlockDim>
                        <<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
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
        default:
          PADDLE_THROW(
              errors::Fatal("the input k has error when use getMaxLength "
                            "function to get the maxLength."));
      });
#endif
      default:
        PADDLE_THROW(errors::Fatal(
            "the input data shape has error in the topk cuda kernel."));
    }

    // last step, transpose back the indices and output
    funcs::TransCompute<GPUContext, int64_t>(
        ndims, dev_ctx, trans_ind, indices, trans);
    funcs::TransCompute<GPUContext, T>(ndims, dev_ctx, trans_out, out, trans);
  }
}
#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM

template <typename T, typename Context>
void TopkV1Kernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& k_scalar,
                  DenseTensor* out,
                  DenseTensor* indices) {
  TopkKernel<T, Context>(dev_ctx, x, k_scalar, -1, true, true, out, indices);
}

#ifdef PADDLE_WITH_CUDA
template <typename T, typename Context>
void TopkKernelCuda(const Context& dev_ctx,
                    const DenseTensor& x,
                    const Scalar& k_scalar,
                    int axis,
                    bool largest,
                    bool sorted,
                    DenseTensor* out,
                    DenseTensor* indices) {
  const auto& in_dims = x.dims();

  // Handle empty output (e.g. when k comes from tensor, dims may contain -1)
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<int64_t>(indices);
    return;
  }

  // 0d input tensor
  if (in_dims.size() == 0) {
    Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    dev_ctx.template Alloc<int64_t>(indices);
    funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }

  if (axis < 0) axis += in_dims.size();
  int k = k_scalar.to<int>();

  // For k=1, call TopkKernel
  if (k == 1) {
    TopkKernel<T, Context>(
        dev_ctx, x, k_scalar, axis, largest, sorted, out, indices);
  }

  // Handle k from tensor: output dims may contain -1, resize before Alloc
  if (k_scalar.FromTensor()) {
    DDim out_dims = out->dims();
    out_dims[axis] = k;
    out->Resize(out_dims);
    indices->Resize(out_dims);
  }

  // Handle empty input
  if (x.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::vectorize(out->dims()), static_cast<T>(NAN), out);
    phi::Full<int64_t, Context>(dev_ctx,
                                phi::vectorize(indices->dims()),
                                static_cast<int64_t>(0),
                                indices);
    return;
  }

  // Now safe to allocate output memory
  T* output_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  phi::DenseTensor input_contiguous;

  if (x.meta().is_contiguous()) {
    input_contiguous = x;
  } else {
    input_contiguous.Resize(x.dims());
    dev_ctx.template Alloc<T>(&input_contiguous);
    phi::Copy<Context>(
        dev_ctx, x, dev_ctx.GetPlace(), false, &input_contiguous);
  }

  int64_t sliceSize = in_dims.size() == 0 ? 1 : in_dims[axis];
  int dim = axis;

  auto stream = dev_ctx.stream();
  int device_id = dev_ctx.GetPlace().GetDeviceId();
  auto place = dev_ctx.GetPlace();

  // Macro: inner kernel launch helpers (same as before)
#define TOPK_RUN_K(INDEX_T, DIM, LAUNCH_FUNCTION_NAME)               \
  LAUNCH_FUNCTION_NAME<T, INDEX_T, DIM>(                             \
      inputInfo,                                                     \
      static_cast<INDEX_T>(sliceSize),                               \
      static_cast<INDEX_T>(k),                                       \
      largest,                                                       \
      static_cast<INDEX_T>(numInputSlices),                          \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),     \
      topKInfo,                                                      \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),       \
      indicesInfo,                                                   \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]), \
      stream)

#define TOPK_RUN_K_MB(INDEX_T, DIM)                                  \
  topk_impl::mbtopk::launch<T, INDEX_T, DIM>(                        \
      inputInfo,                                                     \
      static_cast<INDEX_T>(sliceSize),                               \
      static_cast<INDEX_T>(k),                                       \
      largest,                                                       \
      static_cast<uint32_t>(numInputSlices),                         \
      static_cast<INDEX_T>(inputInfo.strides[collapseInputDim]),     \
      topKInfo,                                                      \
      static_cast<INDEX_T>(topKInfo.strides[collapseTopKDim]),       \
      indicesInfo,                                                   \
      static_cast<INDEX_T>(indicesInfo.strides[collapseIndicesDim]), \
      stream,                                                        \
      device_id,                                                     \
      place)

#define TOPK_RUN_MB(INDEX_T, DIM)                                    \
  if (topk_impl::should_use_multiblock(numInputSlices, sliceSize)) { \
    TOPK_RUN_K_MB(INDEX_T, DIM);                                     \
  } else {                                                           \
    TOPK_RUN_K(INDEX_T, DIM, topk_impl::sbtopk::launch);             \
  }

#define TOPK_RUN_DIM(INDEX_T) \
  if (allDims == 1) {         \
    TOPK_RUN_MB(INDEX_T, 1);  \
  } else if (allDims == 2) {  \
    TOPK_RUN_MB(INDEX_T, 2);  \
  } else if (allDims == 3) {  \
    TOPK_RUN_MB(INDEX_T, 3);  \
  } else {                    \
    TOPK_RUN_MB(INDEX_T, -1); \
  }

  // RUN_T: Build TensorInfo, collapse dims, and launch — all parameterized
  // by INDEX_T.
#define TOPK_RUN_T(INDEX_T)                                                  \
  do {                                                                       \
    auto inputInfo =                                                         \
        topk_impl::getTensorInfo<const T, INDEX_T>(input_contiguous);        \
    auto topKInfo = topk_impl::getTensorInfo<T, INDEX_T>(*out);              \
    auto indicesInfo = topk_impl::getTensorInfo<int64_t, INDEX_T>(*indices); \
                                                                             \
    /* Handle 0-d tensor: expand to 1-d */                                   \
    if (!in_dims.size()) {                                                   \
      inputInfo.dims = 1;                                                    \
      inputInfo.sizes[0] = 1;                                                \
      inputInfo.strides[0] = 1;                                              \
      topKInfo.dims = 1;                                                     \
      topKInfo.sizes[0] = 1;                                                 \
      topKInfo.strides[0] = 1;                                               \
      indicesInfo.dims = 1;                                                  \
      indicesInfo.sizes[0] = 1;                                              \
      indicesInfo.strides[0] = 1;                                            \
    }                                                                        \
                                                                             \
    /* Set sizes[dim] = 1 to calculate slice offsets */                      \
    inputInfo.sizes[dim] = 1;                                                \
    topKInfo.sizes[dim] = 1;                                                 \
    indicesInfo.sizes[dim] = 1;                                              \
                                                                             \
    /* Stash stride of dim because it can be accidentally collapsed */       \
    auto strideTopK = topKInfo.strides[dim];                                 \
    auto strideIndices = indicesInfo.strides[dim];                           \
                                                                             \
    /* Collapse dims */                                                      \
    int collapseInputDim = inputInfo.collapseDims(dim);                      \
    int collapseTopKDim = topKInfo.collapseDims(dim);                        \
    int collapseIndicesDim = indicesInfo.collapseDims(dim);                  \
                                                                             \
    /* Restore stride in case it was collapsed */                            \
    topKInfo.strides[collapseTopKDim] = strideTopK;                          \
    indicesInfo.strides[collapseIndicesDim] = strideIndices;                 \
                                                                             \
    int64_t numInputSlices = 1;                                              \
    for (int i = 0; i < inputInfo.dims; ++i) {                               \
      numInputSlices *= inputInfo.sizes[i];                                  \
    }                                                                        \
                                                                             \
    int allDims = inputInfo.dims;                                            \
    if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {           \
      allDims = -1;                                                          \
    }                                                                        \
                                                                             \
    TOPK_RUN_DIM(INDEX_T);                                                   \
  } while (0)

  // Dispatch: use 32-bit indexing when all tensors qualify, else 64-bit
  if (input_contiguous.numel() > 0) {
    if (topk_impl::canUse32BitIndexMath(input_contiguous) &&
        topk_impl::canUse32BitIndexMath(*out) &&
        topk_impl::canUse32BitIndexMath(*indices)) {
      TOPK_RUN_T(uint32_t);
    } else {
      TOPK_RUN_T(uint64_t);
    }
  }

#undef TOPK_RUN_K
#undef TOPK_RUN_K_MB
#undef TOPK_RUN_MB
#undef TOPK_RUN_DIM
#undef TOPK_RUN_T

  // Sort the results if needed
  if (sorted && k > 1 && out->numel() > 0) {
    // Three-tier sort dispatch:
    //   k <= 32:   Bitonic Sort
    //   k <= 128:  WarpMergeSort (CUB)
    //   k <= 4096: BlockRadixSort (CUB)
    //   k > 4096:  Fall back to ArgsortKernel + TakeAlongAxisKernel
    if (k <= 4096) {
      topk_impl::sortKeyValueInplace<T, Context>(
          dev_ctx, out, indices, axis, largest);
    } else {
      phi::DenseTensor sorted_indices;
      phi::DenseTensor sorted_values;
      sorted_indices.Resize(indices->dims());
      sorted_values.Resize(out->dims());
      dev_ctx.template Alloc<int64_t>(&sorted_indices);
      dev_ctx.template Alloc<T>(&sorted_values);

      phi::ArgsortKernel<T, Context>(dev_ctx,
                                     *out,
                                     axis,
                                     largest,
                                     /*stable=*/true,
                                     &sorted_values,
                                     &sorted_indices);

      phi::DenseTensor new_indices;
      new_indices.Resize(indices->dims());
      dev_ctx.template Alloc<int64_t>(&new_indices);
      phi::TakeAlongAxisKernel<int64_t, Context>(
          dev_ctx, *indices, sorted_indices, axis, &new_indices);

      phi::Copy<Context>(
          dev_ctx, sorted_values, dev_ctx.GetPlace(), false, out);
      phi::Copy<Context>(
          dev_ctx, new_indices, dev_ctx.GetPlace(), false, indices);
    }
  }
}
#endif
}  // namespace phi

#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(topk,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopkKernelCuda,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
#else
PD_REGISTER_KERNEL(topk,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopkKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
#endif

PD_REGISTER_KERNEL(topk_v1,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopkV1Kernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
