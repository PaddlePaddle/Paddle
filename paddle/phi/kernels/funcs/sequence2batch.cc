/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace phi {
namespace funcs {

template <typename T>
class CopyMatrixRowsFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& src,
                  paddle::framework::Vector<size_t> index_lod,
                  phi::DenseTensor* dst,
                  bool is_src_index) {
    size_t* index = index_lod.data();
    auto src_dims = src.dims();
    auto dst_dims = dst->dims();
    PADDLE_ENFORCE_EQ(src_dims.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The source tensor must be a matrix with rank 2, but "
                          "got the source tensor rank is %lu. "
                          "Please check the rank of the source tensor",
                          src_dims.size()));
    PADDLE_ENFORCE_EQ(dst_dims.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The destination tensor must be a matrix with rank, "
                          "but got the destination tensor rank is %lu. "
                          "Please check the rank of the destination tensor",
                          dst_dims.size()));
    PADDLE_ENFORCE_EQ(
        src_dims[1],
        dst_dims[1],
        phi::errors::InvalidArgument(
            "The width of the source tensor and the destination tensor must be "
            "same. But got %lu != %lu.Please check the rank of the source "
            "tensor",
            src_dims.size(),
            dst_dims.size()));
    auto height = dst_dims[0];
    auto width = dst_dims[1];
    auto* src_data = src.data<T>();
    auto* dst_data = dst->data<T>();
    const int sz = width * sizeof(T);
    if (is_src_index) {
      for (int i = 0; i < height; ++i) {
        memcpy(dst_data + i * width, src_data + index[i] * width, sz);
      }
    } else {
      for (int i = 0; i < height; ++i) {
        memcpy(dst_data + index[i] * width, src_data + i * width, sz);
      }
    }
  }
};

template class CopyMatrixRowsFunctor<phi::CPUContext, float>;
template class CopyMatrixRowsFunctor<phi::CPUContext, double>;

template class LoDTensor2BatchFunctor<phi::CPUContext, float>;
template class LoDTensor2BatchFunctor<phi::CPUContext, double>;
template class Batch2LoDTensorFunctor<phi::CPUContext, float>;
template class Batch2LoDTensorFunctor<phi::CPUContext, double>;

}  // namespace funcs
}  // namespace phi
