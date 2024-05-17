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

#include <vector>
#include "paddle/common/array.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#endif

namespace phi {

namespace funcs {

template <typename T, typename Context>
phi::DenseTensor GetReshapeAndExpandTensor(const Context& dev_ctx,
                                           const phi::DenseTensor& tensor,
                                           const phi::DDim& res_dim,
                                           const phi::DDim& bd_dim,
                                           int index) {
  std::vector<int64_t> before_dims = common::vectorize(tensor.dims());
  std::vector<int64_t> mid_dims(res_dim.size(), 1);

  if (index == 0) {
    for (size_t i = 0; i < before_dims.size(); ++i) {
      mid_dims[bd_dim.size() - i - 1] = before_dims[before_dims.size() - i - 1];
    }
  } else {
    mid_dims[index] = before_dims[0];
  }

  phi::DenseTensor mid_tensor(tensor.dtype());
  mid_tensor.Resize(common::make_ddim(mid_dims));
  ReshapeInferKernel<Context>(dev_ctx, tensor, IntArray(mid_dims), &mid_tensor);

  phi::DenseTensor res_tensor(tensor.dtype());
  res_tensor.Resize(res_dim);
  ExpandKernel<T, Context>(
      dev_ctx, mid_tensor, IntArray(common::vectorize(res_dim)), &res_tensor);
  return res_tensor;
}

template <typename T, typename Context>
std::vector<const phi::DenseTensor*> DealWithBoolIndices(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& indices_v,
    std::vector<phi::DenseTensor>* tmp_indices_v) {
  std::vector<const phi::DenseTensor*> res;

  bool contains_bool_tensor = false;
  for (size_t i = 0; i < indices_v.size(); ++i) {
    if (indices_v[i]->dtype() == phi::DataType::BOOL) {
      contains_bool_tensor = true;
      break;
    }
  }

  if (contains_bool_tensor) {
    for (size_t i = 0; i < indices_v.size(); ++i) {
      if (indices_v[i]->dtype() == phi::DataType::BOOL) {
        int rank = indices_v[i]->dims().size();
        PADDLE_ENFORCE_GE(rank,
                          1UL,
                          phi::errors::InvalidArgument(
                              "the only bool tensor in indices should "
                              "have number of dimension at least 1"));
        phi::DenseTensor nonzero_indices(phi::DataType::INT64);
        nonzero_indices.Resize(common::make_ddim({-1, rank}));
        NonZeroKernel<bool, Context>(dev_ctx, *indices_v[i], &nonzero_indices);

        if (nonzero_indices.numel() == 0) {
          std::vector<const phi::DenseTensor*> empty_indices;
          return empty_indices;
        }

        std::vector<phi::DenseTensor*> integer_indices(rank, nullptr);
        const int tmp_ix = tmp_indices_v->size();
        for (int i = 0; i < rank; ++i) {
          tmp_indices_v->emplace_back(
              DenseTensor(phi::DataType::INT64)
                  .Resize(common::make_ddim({nonzero_indices.dims()[0]})));
        }
        for (int i = 0; i < rank; ++i) {
          integer_indices[i] = &((*tmp_indices_v)[i + tmp_ix]);
        }
        SplitWithNumKernel<int64_t, Context>(
            dev_ctx, nonzero_indices, rank, 1, integer_indices);
#ifdef PADDLE_WITH_XPU
        auto place = dev_ctx.GetPlace();
        if (place.GetType() == phi::AllocationType::XPU) {
          auto& pool = phi::DeviceContextPool::Instance();
          auto* xpu_ctx = static_cast<phi::XPUContext*>(pool.Get(place));
          if (xpu_ctx->x_context()->xpu_stream) {
            dev_ctx.Wait();
          }
        }
#endif

      } else if ((indices_v[i]->dtype() == phi::DataType::INT64) ||
                 (indices_v[i]->dtype() == phi::DataType::INT32)) {
        tmp_indices_v->emplace_back(*indices_v[i]);
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "data type of tensor in indices must be int32, int64 or bool"));
      }
    }

    res.reserve(tmp_indices_v->size());
    for (size_t i = 0; i < tmp_indices_v->size(); ++i) {
      res.emplace_back(&((*tmp_indices_v)[i]));
    }
  } else {
    res = indices_v;
  }
  return res;
}

static phi::DDim BroadCastTensorsDims(
    const std::vector<const phi::DenseTensor*>& tensors) {
  int target_rank = 0;
  for (const auto& tensor : tensors) {
    target_rank = std::max(target_rank, tensor->dims().size());
  }

  PADDLE_ENFORCE_GT(target_rank,
                    0,
                    errors::InvalidArgument("BroadCastTensorsDims requires at "
                                            "least one input tensor to have "
                                            "rank greater than zero"));

  std::vector<int64_t> target_dims(target_rank, 0);
  for (int index = 0; index < target_rank; index++) {
    int target_dim_size = 1;
    for (const auto& tensor : tensors) {
      auto input_ddim = tensor->dims();
      int axis = static_cast<int>(input_ddim.size()) - index - 1;
      int dim_size = 1;
      if (axis >= 0) {
        dim_size = input_ddim[axis];
      }

      if (target_dim_size != 1 && dim_size != 1 &&
          target_dim_size != dim_size) {
        PADDLE_THROW(errors::InvalidArgument(
            "BroadCastTensorsDims inputs does not satisfy bcast semantics, "
            "please check axis = %d in reverse order",
            index));
      }

      target_dim_size = dim_size == 1 ? target_dim_size : dim_size;
    }
    target_dims[target_rank - index - 1] = target_dim_size;
  }
  return common::make_ddim(target_dims);
}

template <typename T, typename Context>
T** GetDevicePointerArray(const Context& ctx,
                          const std::vector<const DenseTensor*>& indices_v) {
  std::vector<const T*> h_indices_v(indices_v.size());
  for (size_t i = 0; i < indices_v.size(); ++i) {
    h_indices_v[i] = indices_v[i]->data<T>();
  }
  auto d_indices_data = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      h_indices_v.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  phi::memory_utils::Copy(ctx.GetPlace(),
                          d_indices_data->ptr(),
                          phi::CPUPlace(),
                          reinterpret_cast<void*>(h_indices_v.data()),
                          h_indices_v.size() * sizeof(T*),
                          ctx.stream());
  return reinterpret_cast<T**>(d_indices_data->ptr());
}

template <typename T, typename Context>
void DealWithIndices(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<const phi::DenseTensor*>& int_indices_v,
                     std::vector<const phi::DenseTensor*>* res_indices_v,
                     std::vector<DenseTensor>* tmp_res_indices_v,
                     const std::vector<DenseTensor>& range_tensor_v,
                     const phi::DDim& bd_dim,
                     std::vector<int64_t>* res_dim_v) {
  size_t total_dims = x.dims().size();
  if (int_indices_v.size() < total_dims) {
    std::vector<int64_t> tmp_x_dims = common::vectorize(x.dims());
    int len_bd_dim = bd_dim.size();
    res_dim_v->insert(res_dim_v->end(),
                      tmp_x_dims.begin() + int_indices_v.size(),
                      tmp_x_dims.end());
    phi::DDim res_dim = common::make_ddim(*res_dim_v);
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      phi::DenseTensor index_tensor;
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        index_tensor = phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64);
      } else {
        index_tensor = *int_indices_v[i];
      }
      tmp_res_indices_v->emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx, index_tensor, res_dim, bd_dim, 0));
    }
    for (size_t i = 0; i < range_tensor_v.size(); ++i) {
      tmp_res_indices_v->emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx, range_tensor_v[i], res_dim, bd_dim, i + len_bd_dim));
    }
    for (size_t i = 0; i < res_indices_v->size(); ++i) {
      (*res_indices_v)[i] = &(*tmp_res_indices_v)[i];
    }

  } else {
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      phi::DenseTensor index_tensor;
      phi::DenseTensor expand_index;
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        index_tensor = phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64);
      } else {
        index_tensor = *int_indices_v[i];
      }
      if (bd_dim != int_indices_v[i]->dims()) {
        expand_index = DenseTensor(phi::DataType::INT64).Resize(bd_dim);
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            index_tensor,
            IntArray(common::vectorize<int64_t>(bd_dim)),
            &expand_index);
      } else {
        expand_index = index_tensor;
      }
      tmp_res_indices_v->emplace_back(expand_index);
    }
    for (size_t i = 0; i < res_indices_v->size(); ++i) {
      (*res_indices_v)[i] = &(*tmp_res_indices_v)[i];
    }
  }
}

static void CalCompressedDimsWith1AndWithout1(
    std::vector<int64_t>* after_dims,
    std::vector<int64_t>* before_dims,
    std::vector<int64_t>* compress_dims,
    std::vector<int64_t>* dims_without_1) {
  int i = static_cast<int>(after_dims->size()) - 1;
  int j = static_cast<int>(before_dims->size()) - 1;
  if (i < j) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "shape of value can't not be broadcast to shape of x[indices]"));
  }

  while ((i >= 0) && (j >= 0)) {
    if ((*after_dims)[i] == (*before_dims)[j]) {
      dims_without_1->push_back((*before_dims)[j]);
      i--;
      j--;
      continue;
    } else if ((*before_dims)[j] == 1) {
      compress_dims->push_back(i);
      i--;
      j--;
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "shape of value can't not be broadcast to shape of x[indices]"));
    }
  }
  while (i >= 0) {
    compress_dims->push_back(i);
    i--;
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__global__ void range_cuda_kernel(int64_t N, T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= N) {
    return;
  }
  out[idx] = idx;
}

template <typename T, typename Context>
phi::DenseTensor GetRangeCudaTensor(const Context& dev_ctx,
                                    int64_t N,
                                    phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(common::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
  range_cuda_kernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          N, out);
  return res;
}
#endif

template <typename T>
void range_kernel(int64_t N, T* out) {
  for (int64_t idx = 0; idx < N; ++idx) {
    out[idx] = idx;
  }
}

template <typename T, typename Context>
phi::DenseTensor GetRangeTensor(const Context& dev_ctx,
                                int64_t N,
                                phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(common::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  range_kernel<T>(N, out);
  return res;
}

}  // namespace funcs
}  // namespace phi
