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

#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class CopyMatrixRowsFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& src,
                  framework::Vector<size_t> index_lod, framework::Tensor* dst,
                  bool is_src_index) {
    size_t* index = index_lod.data();
    auto src_dims = src.dims();
    auto dst_dims = dst->dims();
    PADDLE_ENFORCE_EQ(src_dims.size(), 2UL,
                      "The src must be matrix with rank 2.");
    PADDLE_ENFORCE_EQ(dst_dims.size(), 2UL,
                      "The dst must be matrix with rank 2.");
    PADDLE_ENFORCE_EQ(src_dims[1], dst_dims[1],
                      "The width of src and dst must be same.");
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

template class CopyMatrixRowsFunctor<platform::CPUDeviceContext, float>;
template class CopyMatrixRowsFunctor<platform::CPUDeviceContext, double>;

template class LoDTensor2BatchFunctor<platform::CPUDeviceContext, float>;
template class LoDTensor2BatchFunctor<platform::CPUDeviceContext, double>;
template class Batch2LoDTensorFunctor<platform::CPUDeviceContext, float>;
template class Batch2LoDTensorFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
