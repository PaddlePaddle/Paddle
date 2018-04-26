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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void CopyMatrixRowsKernel(const T* src, T* dst, const size_t* index,
                                     int64_t height, int64_t width,
                                     bool is_src_index) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int id = blockIdx.x + idy * GridDimX;
  while (id < height) {
    int src_idx = is_src_index ? index[id] : id;
    int dst_idx = is_src_index ? id : index[id];
    const T* src_data = src + src_idx * width;
    T* dst_data = dst + dst_idx * width;
    for (int i = idx; i < width; i += BlockDimX) {
      dst_data[i] = src_data[i];
    }
    id += BlockDimY * GridDimX;
  }
}

template <typename T>
class CopyMatrixRowsFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& src,
                  framework::Vector<size_t> index_lod, framework::Tensor* dst,
                  bool is_src_index) {
    auto src_dims = src.dims();
    auto dst_dims = dst->dims();
    PADDLE_ENFORCE_EQ(src_dims.size(), 2,
                      "The src must be matrix with rank 2.");
    PADDLE_ENFORCE_EQ(dst_dims.size(), 2,
                      "The dst must be matrix with rank 2.");
    PADDLE_ENFORCE_EQ(src_dims[1], dst_dims[1],
                      "The width of src and dst must be same.");
    auto height = dst_dims[0];
    auto width = dst_dims[1];
    auto* src_data = src.data<T>();
    auto* dst_data = dst->data<T>();

    dim3 threads(128, 8);
    dim3 grid(8, 1);
    auto stream = context.stream();
    CopyMatrixRowsKernel<T, 128, 8, 8><<<grid, threads, 0, stream>>>(
        src_data, dst_data, index_lod.CUDAData(context.GetPlace()), height,
        width, is_src_index);
  }
};

template class CopyMatrixRowsFunctor<platform::CUDADeviceContext, float>;
template class CopyMatrixRowsFunctor<platform::CUDADeviceContext, double>;

template class LoDTensor2BatchFunctor<platform::CUDADeviceContext, float>;
template class LoDTensor2BatchFunctor<platform::CUDADeviceContext, double>;
template class Batch2LoDTensorFunctor<platform::CUDADeviceContext, float>;
template class Batch2LoDTensorFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
