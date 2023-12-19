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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/select_impl.cu.h"
#include "paddle/phi/kernels/nonzero_kernel.h"

namespace phi {
template <typename MaskT, typename IndexT, typename OutT>
struct IndexFunctor {
  IndexT strides[phi::DDim::kMaxRank];
  int rank;

  explicit IndexFunctor(const phi::DDim &in_dims) {
    rank = in_dims.size();
    // Get strides according to in_dims
    strides[0] = 1;
    for (IndexT i = 1; i < rank; i++) {
      strides[i] = strides[i - 1] * in_dims[rank - i];
    }
  }

  HOSTDEVICE inline void operator()(OutT *out,
                                    const MaskT *mask,
                                    const IndexT *index,
                                    const int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {
        IndexT data_index = index[idx];
        // get index
        for (int rank_id = rank - 1; rank_id >= 0; --rank_id) {
          out[store_fix] = static_cast<OutT>(data_index / strides[rank_id]);
          data_index = data_index % strides[rank_id];
          store_fix++;
        }
      }
    }
  }
};

template <typename T, typename Context>
void NonZeroKernel(const Context &dev_ctx,
                   const DenseTensor &condition,
                   DenseTensor *out) {
  DenseTensor in_data;
  auto dims = condition.dims();
  using Functor = IndexFunctor<T, int64_t, int64_t>;
  Functor index_functor = Functor(dims);
  phi::funcs::SelectKernel<T, T, int64_t, 0, Functor>(
      dev_ctx, condition, in_data, out, index_functor);
}
}  // namespace phi

PD_REGISTER_KERNEL(nonzero,
                   GPU,
                   ALL_LAYOUT,
                   phi::NonZeroKernel,
                   int64_t,
                   int,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
