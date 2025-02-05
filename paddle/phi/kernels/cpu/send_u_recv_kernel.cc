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

#include "paddle/phi/kernels/send_u_recv_kernel.h"

#include <algorithm>
#include <set>
#include <vector>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_recv_funcs.h"

namespace phi {

template <typename T, typename IndexT, typename Functor>
void GraphSendRecvCpuLoop(const int& input_size,
                          const int& index_size,
                          const IndexT* s_index,
                          const IndexT* d_index,
                          const DenseTensor& src,
                          DenseTensor* dst,
                          const std::string& reduce_op,
                          int* dst_count = nullptr) {
  Functor functor;
  if (reduce_op == "SUM") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      ElementwiseInnerOperation<T, IndexT, Functor>(
          src, dst, src_idx, dst_idx, false, functor);
    }
  } else if (reduce_op == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      ElementwiseInnerOperation<T, IndexT, Functor>(
          src, dst, src_idx, dst_idx, false, functor);
    }
    for (int i = 0; i < index_size; ++i) {
      IndexT dst_idx = d_index[i];
      *(dst_count + dst_idx) += 1;
    }
    for (int i = 0; i < input_size; ++i) {
      if (*(dst_count + i) == 0) continue;
      auto dst_slice = dst->Slice(i, i + 1);
      auto eigen_dst = phi::EigenVector<T>::Flatten(dst_slice);
      eigen_dst = eigen_dst / static_cast<T>(*(dst_count + i));
    }
  } else if (reduce_op == "MIN" || reduce_op == "MAX") {
    std::set<IndexT> existed_dst;
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      bool in_set = existed_dst.find(dst_idx) != existed_dst.end();
      if (!in_set) {
        ElementwiseInnerOperation<T, IndexT, Functor>(
            src, dst, src_idx, dst_idx, true, functor);
        existed_dst.emplace(dst_idx);
      } else {
        ElementwiseInnerOperation<T, IndexT, Functor>(
            src, dst, src_idx, dst_idx, false, functor);
      }
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendRecvOpKernelLaunchHelper(const Context& ctx,
                                       const DenseTensor& x,
                                       const DenseTensor& src_index,
                                       const DenseTensor& dst_index,
                                       const std::string& reduce_op,
                                       int64_t out_size,
                                       DenseTensor* out,
                                       DenseTensor* dst_count = nullptr) {
  const int& index_size = src_index.dims()[0];  // NOLINT

  const auto& src_dims = x.dims();
  int64_t memset_size = 1;
  if (out_size <= 0) {
    out->Resize(src_dims);
    for (int i = 0; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
  } else {
    // Set out dim following out_size.
    std::vector<int64_t> dims_ = common::vectorize(src_dims);
    if (!dims_.empty()) {
      dims_[0] = out_size;
    }
    out->Resize(common::make_ddim(dims_));
    memset_size = out_size;
    for (int i = 1; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
  }

  ctx.template Alloc<T>(out);
  T* p_output = out->data<T>();
  const size_t& memset_bytes = memset_size * sizeof(T);
  memset(p_output, 0, memset_bytes);

  if (index_size == 0) return;
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (reduce_op == "SUM") {
    GraphSendRecvCpuLoop<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, x, out, reduce_op);
  } else if (reduce_op == "MIN") {
    GraphSendRecvCpuLoop<T, IndexT, GraphSendRecvMinFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, x, out, reduce_op);
  } else if (reduce_op == "MAX") {
    GraphSendRecvCpuLoop<T, IndexT, GraphSendRecvMaxFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, x, out, reduce_op);
  } else if (reduce_op == "MEAN") {
    int64_t input_size = out_size <= 0 ? src_dims[0] : out_size;
    dst_count->Resize({input_size});
    ctx.template Alloc<int>(dst_count);
    int* p_dst_count = dst_count->data<int>();
    memset(p_dst_count, 0, input_size * sizeof(int));
    GraphSendRecvCpuLoop<T, IndexT, GraphSendRecvSumFunctor<T>>(input_size,
                                                                index_size,
                                                                s_index,
                                                                d_index,
                                                                x,
                                                                out,
                                                                reduce_op,
                                                                p_dst_count);
  }
}

template <typename T, typename Context>
void SendURecvKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& src_index,
                     const DenseTensor& dst_index,
                     const std::string& reduce_op,
                     const IntArray& out_size,
                     DenseTensor* out,
                     DenseTensor* dst_count) {
  auto index_type = src_index.dtype();
  auto& out_size_data = out_size.GetData();
  if (index_type == phi::DataType::INT32) {
    GraphSendRecvOpKernelLaunchHelper<Context, T, int32_t>(ctx,
                                                           x,
                                                           src_index,
                                                           dst_index,
                                                           reduce_op,
                                                           out_size_data[0],
                                                           out,
                                                           dst_count);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendRecvOpKernelLaunchHelper<Context, T, int64_t>(ctx,
                                                           x,
                                                           src_index,
                                                           dst_index,
                                                           reduce_op,
                                                           out_size_data[0],
                                                           out,
                                                           dst_count);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(send_u_recv,
                   CPU,
                   ALL_LAYOUT,
                   phi::SendURecvKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
