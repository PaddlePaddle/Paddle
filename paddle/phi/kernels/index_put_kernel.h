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

#pragma once

#include <vector>
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi {

template <typename T, typename Context>
static phi::DenseTensor GetReshapeAndExpandTensor(
    const Context& dev_ctx,
    const phi::DenseTensor& tensor,
    const phi::DDim& res_dim,
    const phi::DDim& bd_dim,
    int index) {
  std::vector<int64_t> before_dims = phi::vectorize(tensor.dims());
  std::vector<int64_t> mid_dims(res_dim.size(), 1);

  if (index == 0) {
    for (size_t i = 0; i < before_dims.size(); ++i) {
      mid_dims[bd_dim.size() - i - 1] = before_dims[before_dims.size() - i - 1];
    }
  } else {
    mid_dims[index] = before_dims[0];
  }
  std::cout << "this is mid_dim" << std::endl;
  for (auto dim : mid_dims) {
    std::cout << dim << std::endl;
  }

  phi::DenseTensor mid_tensor(tensor.dtype());
  mid_tensor.Resize(phi::make_ddim(mid_dims));
  ReshapeInferKernel<Context>(dev_ctx, tensor, IntArray(mid_dims), &mid_tensor);

  phi::DenseTensor res_tensor(tensor.dtype());
  res_tensor.Resize(res_dim);
  ExpandKernel<T, Context>(
      dev_ctx, mid_tensor, IntArray(phi::vectorize(res_dim)), &res_tensor);
  return res_tensor;
}

template <typename T, typename Context>
static std::vector<const phi::DenseTensor*> DealWithBoolIndices(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& indices_v,
    std::vector<phi::DenseTensor>* tmp_indices_v) {
  std::vector<const phi::DenseTensor*> res(indices_v.begin(), indices_v.end());
  bool contains_bool_tensor = false;
  for (size_t i = 0; i < indices_v.size(); ++i) {
    if (indices_v[i]->dtype() == phi::DataType::BOOL) {
      contains_bool_tensor = true;
    } else if ((indices_v[i]->dtype() == phi::DataType::INT64) ||
               (indices_v[i]->dtype() == phi::DataType::INT32)) {
      if (contains_bool_tensor) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "indices contains bool tensor and int32/int64 tensor at the same "
            "time"));
      }
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "data type of tensor in indices must be int32, int64 or bool"));
    }
  }

  if (contains_bool_tensor) {
    if (indices_v.size() != 1) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "the size of indices must be 1 when it containts bool tensor"));
    }
    int rank = indices_v[0]->dims().size();
    PADDLE_ENFORCE_GE(
        rank,
        1UL,
        phi::errors::InvalidArgument("the only bool tensor in indices should "
                                     "have number of dimension at least 1"));
    phi::DenseTensor nonzero_indices(phi::DataType::INT64);
    nonzero_indices.Resize(phi::make_ddim({-1, rank}));
    NonZeroKernel<bool, Context>(dev_ctx, *indices_v[0], &nonzero_indices);

    std::vector<phi::DenseTensor*> integer_indices(rank, nullptr);
    for (int i = 0; i < rank; ++i) {
      // tmp_indices_v.emplace_back(DenseTensor(phi::DataType::INT64).Resize(phi::make_ddim({nonzero_indices.dims()[0],1})));
      tmp_indices_v->emplace_back(
          DenseTensor(phi::DataType::INT64)
              .Resize(phi::make_ddim({nonzero_indices.dims()[0]})));
    }
    for (int i = 0; i < rank; ++i) {
      integer_indices[i] = &((*tmp_indices_v)[i]);
    }
    SplitWithNumKernel<int64_t, Context>(
        dev_ctx, nonzero_indices, rank, 1, integer_indices);

    std::vector<const phi::DenseTensor*> res_tmp(integer_indices.size(),
                                                 nullptr);
    for (int i = 0; i < rank; ++i) {
      res_tmp[i] = &((*tmp_indices_v)[i]);
    }
    res.swap(res_tmp);
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
  return phi::make_ddim(target_dims);
}

template <typename T, typename Context>
T** GetDevicePointerArray(const Context& ctx,
                          const std::vector<const DenseTensor*>& indices_v) {
  std::vector<const T*> h_indices_v(indices_v.size());
  for (int i = 0; i < indices_v.size(); ++i) {
    h_indices_v[i] = indices_v[i]->data<T>();
  }
  auto d_indices_data = paddle::memory::Alloc(
      ctx.GetPlace(),
      h_indices_v.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  paddle::memory::Copy(ctx.GetPlace(),
                       d_indices_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(h_indices_v.data()),
                       h_indices_v.size() * sizeof(T*),
                       ctx.stream());
  return reinterpret_cast<T**>(d_indices_data->ptr());
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    const DenseTensor& value,
                    bool accumulate,
                    DenseTensor* out);

}  // namespace phi
