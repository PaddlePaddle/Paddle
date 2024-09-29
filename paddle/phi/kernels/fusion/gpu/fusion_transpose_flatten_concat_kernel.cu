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

#include <algorithm>
#include <type_traits>

#include "paddle/common/errors.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;

template <typename T, typename Context>
void TransposeFlattenConcatFusionKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& x,
    const std::vector<int>& trans_axis,
    const int flatten_axis,
    const int concat_axis,
    DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA)
  dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));
  auto odims = out->dims();

  int rank = x[0]->dims().size();
  // use at least 4D in cudnnTransformTensor
  int max_dim = rank < 4 ? 4 : rank;
  std::vector<int> stride_x(max_dim, 0);
  std::vector<int> stride_y(max_dim, 0);
  std::vector<int> dims_y(max_dim, 0);

  cudnnTensorDescriptor_t in_desc;
  cudnnTensorDescriptor_t out_desc;
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&in_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnCreateTensorDescriptor(&out_desc));
  cudnnDataType_t cudnn_dtype = CudnnDataType<T>::type;

  auto handle = dev_ctx.cudnn_handle();

  T* odata = out->data<T>();
  for (auto& item : x) {
    auto perm_shape = phi::funcs::GetPermuteShape(trans_axis, item->dims());
    int osize = 1;
    auto idims = item->dims();
    for (int i = 0; i < rank; i++) {
      stride_x[i] = 1;
      for (int j = trans_axis[i] + 1; j < rank; j++) {
        stride_x[i] *= idims[j];
      }
      dims_y[i] = perm_shape[i];
      osize *= perm_shape[i];
    }
    stride_y[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      if (((i + 1) == flatten_axis) && (concat_axis == 1)) {
        stride_y[i] = odims[1];
      } else {
        stride_y[i] = stride_y[i + 1] * perm_shape[i + 1];
      }
    }

    // Since concat is after flatten, the output is 2D tensor.
    // If concat_axis is 0, each input's permutated tensor is continuous.
    // If concat_axis is 1, the stride of 0-th dim of each input's
    // permutated tensor is odims()[1].

    for (int i = rank; i < max_dim; i++) {
      stride_x[i] = 1;
      stride_y[i] = 1;
      dims_y[i] = 1;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        in_desc, cudnn_dtype, max_dim, dims_y.data(), stride_x.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        out_desc, cudnn_dtype, max_dim, dims_y.data(), stride_y.data()));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnTransformTensor(
        handle,
        CudnnDataType<T>::kOne(),
        in_desc,
        static_cast<const void*>(item->data<T>()),
        CudnnDataType<T>::kZero(),
        out_desc,
        static_cast<void*>(odata)));
    if (concat_axis == 0) {
      odata += osize;
    } else {
      auto flat_shape = phi::funcs::GetFlattenShape(flatten_axis, perm_shape);
      odata += flat_shape[1];
    }
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(in_desc));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnDestroyTensorDescriptor(out_desc));
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The fusion_transpose_flatten_concat operator is not supported on HIP."));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fusion_transpose_flatten_concat,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::TransposeFlattenConcatFusionKernel,
                   float,
                   double) {}
