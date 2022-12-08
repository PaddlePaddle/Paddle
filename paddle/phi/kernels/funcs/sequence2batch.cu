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

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void CopyMatrixRowsKernel(const T* src,
                                     T* dst,
                                     const size_t* index,
                                     int64_t height,
                                     int64_t width,
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
class CopyMatrixRowsFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& src,
                  paddle::framework::Vector<size_t> index_lod,
                  phi::DenseTensor* dst,
                  bool is_src_index) {
    auto src_dims = src.dims();
    auto dst_dims = dst->dims();
    PADDLE_ENFORCE_EQ(src_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The source tensor must be a matrix with rank 2, but "
                          "got the source tensor rank is %lu. "
                          "Please check the rank of the source tensor",
                          src_dims.size()));
    PADDLE_ENFORCE_EQ(dst_dims.size(),
                      2,
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

    dim3 threads(128, 8);
    dim3 grid(8, 1);
    auto stream = context.stream();
    paddle::framework::MixVector<size_t> mix_index_lod(&index_lod);
    CopyMatrixRowsKernel<T, 128, 8, 8><<<grid, threads, 0, stream>>>(
        src_data,
        dst_data,
        mix_index_lod.CUDAData(context.GetPlace()),
        height,
        width,
        is_src_index);
  }
};

template class CopyMatrixRowsFunctor<phi::GPUContext, float>;
template class CopyMatrixRowsFunctor<phi::GPUContext, double>;

template class LoDTensor2BatchFunctor<phi::GPUContext, float>;
template class LoDTensor2BatchFunctor<phi::GPUContext, double>;
template class Batch2LoDTensorFunctor<phi::GPUContext, float>;
template class Batch2LoDTensorFunctor<phi::GPUContext, double>;

}  // namespace funcs
}  // namespace phi
