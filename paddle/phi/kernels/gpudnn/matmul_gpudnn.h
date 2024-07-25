// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MUSA
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/expect.h"

namespace phi {
using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;
using ScopedBatchMatmulDescriptor =
    phi::backends::gpu::ScopedBatchMatmulDescriptor;
using ScopedMatMulDescriptor = phi::backends::gpu::ScopedMatMulDescriptor;
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

static void InternalMemFree_matmul_bmm(void* ptr) {

}


static inline DenseTensor LogicalTransposeLast2Dim(const DenseTensor& x) {
  int ndim = x.dims().size();
  if (ndim < 2) {
    return x;
  }
  auto x_tmp = std::make_shared<phi::DenseTensor>(x);
  paddle::Tensor x_tensor(x_tmp);

  std::vector<int> perms(ndim);
  for (int i = 0; i < ndim; i++) {
    perms[i] = i;
  }
  std::swap(perms[ndim - 2], perms[ndim - 1]);
  paddle::Tensor x_tensor_trans =
      paddle::experimental::transpose(x_tensor, perms);

  return *static_cast<phi::DenseTensor*>(x_tensor_trans.impl().get());
}

static inline bool IsContiguous(const paddle::Tensor& x) {
  if (unlikely(!(x.is_dense_tensor()))) {
    return false;
  }
  DDim dims = x.dims();
  DDim strides = x.strides();
  for (int i = 0; i < dims.size() - 1; i++) {
    if (strides[i] != (strides[i + 1] * dims[i + 1])) {
      return false;
    }
  }
  // shared_ptr
  auto dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
  return dense_tensor->meta().is_contiguous();
}

static inline bool IsTransposedFromContiguousTensor(const DenseTensor& x) {
  int ndim = x.dims().size();
  if (ndim >= 2) {
    auto x_tmp = std::make_shared<phi::DenseTensor>(x);
    paddle::Tensor x_tensor(x_tmp);

    std::vector<int> perms(ndim);
    for (int i = 0; i < ndim; i++) {
      perms[i] = i;
    }
    std::swap(perms[ndim - 2], perms[ndim - 1]);
    paddle::Tensor x_tensor_trans =
        paddle::experimental::transpose(x_tensor, perms);

    return IsContiguous(x_tensor_trans);
  }
  return false;
}

static inline void ContiguousTensorRef(const DenseTensor& x,
                                       DenseTensor* x_contig,
                                       bool* trans) {
  // The values of Tensor that passed into muDNN should be laied out in the
  // storage starting from the rightmost dimension onward(Contiguous). seems
  // paddle already done something like pre-contiguous before calling into the
  // op kernel(see matmul in paddle/phi/api/lib/api.cc), but for safety, I still
  // add CheckAndTrans2NewContiguousTensor here.
  // if (IsTransposedFromContiguousTensor(x)) {
  if constexpr (false) {
    // TODO(mingyuan.wang): remove this, whether the matrix requires a
    // transformation is indicated by the flag of trans, and the matrix is
    // contigious in most cases, namely, we use `trans` to indicate whether the
    // matrix needs to be transposed instead of transposing the matrix directly
    // by tensor.transpose.
    *x_contig = LogicalTransposeLast2Dim(x);
    *trans = !(*trans);
  } else {
    *x_contig = paddle::experimental::CheckAndTrans2NewContiguousTensor(x);
  }
}

template <typename T, typename Context>
void MatMulGPUDNNKernelImpl(const Context& dev_ctx,
                            const DenseTensor& x,
                            bool trans_x,
                            const DenseTensor& y,
                            bool trans_y,
                            DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  DenseTensor x_contig;
  DenseTensor y_contig;
  ContiguousTensorRef(x, &x_contig, &trans_x);
  ContiguousTensorRef(y, &y_contig, &trans_y);

  auto handle = dev_ctx.cudnn_handle();
  ScopedMatMulDescriptor mm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      *out, GPUDNNDataLayout::kNCHW, common::vectorize<int>(out->dims()));
  auto& left_desc =
      left_scoped_desc.descriptor<T>(x_contig,
                                     GPUDNNDataLayout::kNCHW,
                                     common::vectorize<int>(x_contig.dims()));
  auto& right_desc =
      right_scoped_desc.descriptor<T>(y_contig,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(y_contig.dims()));



  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };

  mm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

// also provide the sizes of tensor as a parameter, thus we dont have to
// call tensor.Resize in the cases like dimension folding.
template <typename T, typename Context>
void MatMulGPUDNNKernelImpl(const Context& dev_ctx,
                            const T* x_data,
                            bool trans_x,
                            const std::vector<int>& x_dims,
                            const T* y_data,
                            bool trans_y,
                            const std::vector<int>& y_dims,
                            T* out_data,
                            const std::vector<int>& out_dims) {
  VLOG(3) << "it's user's responsibility to ensure the continuity of input "
             "tensors.";

  auto handle = dev_ctx.cudnn_handle();
  ScopedMatMulDescriptor mm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      out_data, GPUDNNDataLayout::kNCHW, out_dims);
  auto& left_desc =
      left_scoped_desc.descriptor<T>(x_data, GPUDNNDataLayout::kNCHW, x_dims);
  auto& right_desc =
      right_scoped_desc.descriptor<T>(y_data, GPUDNNDataLayout::kNCHW, y_dims);

  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };

  mm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

template <typename T, typename Context>
void MatMulGPUDNNKernelImpl(const Context& dev_ctx,
                            const DenseTensor& x,
                            bool trans_x,
                            const std::vector<int>& x_dims,
                            const DenseTensor& y,
                            bool trans_y,
                            const std::vector<int>& y_dims,
                            DenseTensor* out,
                            const std::vector<int>& out_dims) {
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  DenseTensor x_contig;
  DenseTensor y_contig;
  ContiguousTensorRef(x, &x_contig, &trans_x);
  ContiguousTensorRef(y, &y_contig, &trans_y);

  auto handle = dev_ctx.cudnn_handle();
  ScopedMatMulDescriptor mm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      out->data<T>(), GPUDNNDataLayout::kNCHW, out_dims);
  auto& left_desc = left_scoped_desc.descriptor<T>(
      x_contig.data<T>(), GPUDNNDataLayout::kNCHW, x_dims);
  auto& right_desc = right_scoped_desc.descriptor<T>(
      y_contig.data<T>(), GPUDNNDataLayout::kNCHW, y_dims);

  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };
        
  mm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

template <typename T, typename Context>
void BmmGPUDNNKernelImpl(const Context& dev_ctx,
                         const DenseTensor& x,
                         bool trans_x,
                         const DenseTensor& y,
                         bool trans_y,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }

  DenseTensor x_contig;
  DenseTensor y_contig;
  ContiguousTensorRef(x, &x_contig, &trans_x);
  ContiguousTensorRef(y, &y_contig, &trans_y);
  auto handle = dev_ctx.cudnn_handle();
  ScopedBatchMatmulDescriptor bmm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      *out, GPUDNNDataLayout::kNCHW, common::vectorize<int>(out->dims()));
  auto& left_desc =
      left_scoped_desc.descriptor<T>(x_contig,
                                     GPUDNNDataLayout::kNCHW,
                                     common::vectorize<int>(x_contig.dims()));
  auto& right_desc =
      right_scoped_desc.descriptor<T>(y_contig,
                                      GPUDNNDataLayout::kNCHW,
                                      common::vectorize<int>(y_contig.dims()));
  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };                          

  bmm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

template <typename T, typename Context>
void BmmGPUDNNKernelImpl(const Context& dev_ctx,
                         const T* x_data,
                         bool trans_x,
                         const std::vector<int>& x_dims,
                         const T* y_data,
                         bool trans_y,
                         const std::vector<int>& y_dims,
                         T* out_data,
                         const std::vector<int>& out_dims) {
  VLOG(3) << "it's user's responsibility to ensure the continuity of input "
             "tensors.";

  auto handle = dev_ctx.cudnn_handle();
  ScopedBatchMatmulDescriptor bmm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      out_data, GPUDNNDataLayout::kNCHW, out_dims);
  auto& left_desc =
      left_scoped_desc.descriptor<T>(x_data, GPUDNNDataLayout::kNCHW, x_dims);
  auto& right_desc =
      right_scoped_desc.descriptor<T>(y_data, GPUDNNDataLayout::kNCHW, y_dims);
  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };                                
  bmm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

template <typename T, typename Context>
void BmmGPUDNNKernelImpl(const Context& dev_ctx,
                         const DenseTensor& x,
                         bool trans_x,
                         const std::vector<int>& x_dims,
                         const DenseTensor& y,
                         bool trans_y,
                         const std::vector<int>& y_dims,
                         DenseTensor* out,
                         const std::vector<int>& out_dims) {
  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }
  DenseTensor x_contig;
  DenseTensor y_contig;
  ContiguousTensorRef(x, &x_contig, &trans_x);
  ContiguousTensorRef(y, &y_contig, &trans_y);

  auto handle = dev_ctx.cudnn_handle();
  ScopedBatchMatmulDescriptor bmm_desc;
  ScopedTensorDescriptor out_scoped_desc;
  ScopedTensorDescriptor left_scoped_desc;
  ScopedTensorDescriptor right_scoped_desc;

  auto& out_desc = out_scoped_desc.descriptor<T>(
      out->data<T>(), GPUDNNDataLayout::kNCHW, out_dims);
  auto& left_desc = left_scoped_desc.descriptor<T>(
      x_contig.data<T>(), GPUDNNDataLayout::kNCHW, x_dims);
  auto& right_desc = right_scoped_desc.descriptor<T>(
      y_contig.data<T>(), GPUDNNDataLayout::kNCHW, y_dims);
  Allocator::AllocationPtr memory_for_mudnn; //this is a unique ptr so the memory it holds will be free when it is out of its scope

  auto InternalMemAlloc_matmul = [&memory_for_mudnn, &dev_ctx](size_t s) {
    memory_for_mudnn = std::move(phi::memory_utils::Alloc(dev_ctx.GetPlace(),s,phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream()))));
    return dynload::MemoryHandler(memory_for_mudnn->ptr(), InternalMemFree_matmul_bmm);
  };            
  bmm_desc.descriptor(trans_x, trans_y)
      .Run(*handle,
           out_desc,
           left_desc,
           right_desc,
           InternalMemAlloc_matmul);
}

}  // namespace phi
#endif
