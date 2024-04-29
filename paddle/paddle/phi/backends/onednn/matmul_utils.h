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

#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace phi {
namespace funcs {

DDim RowMatrixDimsFromVector(const DDim& x_dim);
DDim ColumnMatrixDimsFromVector(const DDim& y_dim);

std::vector<int64_t> TransposeAxis(const std::vector<int64_t>& x,
                                   const std::vector<int>& axis);

std::vector<int64_t> GetInputStrides(const std::string input_name,
                                     const DDim& input_dims,
                                     const bool transpose_input,
                                     std::vector<int> shape,
                                     std::vector<int> axis);

template <typename XT, typename YT, typename OT>
class MatmulOneDNNHandler : public OneDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatmulOneDNNHandler(const OneDNNContext& dev_ctx,
                      const std::vector<int64_t>& x_org_dims,
                      const std::vector<int64_t>& y_org_dims,
                      bool trans_x,
                      bool trans_y)
      : OneDNNHandlerNoCachingT<XT, dnnl::matmul>(dev_ctx.GetEngine(),
                                                  dev_ctx.GetPlace()) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = x_dims.size() - 3;
    const int H_idx = x_dims.size() - 2;
    const int W_idx = x_dims.size() - 1;

    if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
    if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

    const memory::dim M = x_dims[H_idx];
    const memory::dim K = x_dims[W_idx];
    const memory::dim N = y_dims[W_idx];

    std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

    x_strides.reserve(x_dims.size());
    y_strides.reserve(x_dims.size());
    out_strides.reserve(x_dims.size());

    if (trans_x) {
      x_strides.insert(x_strides.end(), {M * K, 1, M});
    } else {
      x_strides.insert(x_strides.end(), {M * K, K, 1});
    }

    if (trans_y) {
      y_strides.insert(y_strides.end(), {N * K, 1, K});
    } else {
      y_strides.insert(y_strides.end(), {N * K, N, 1});
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = x_dims.size() - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      y_strides[i] = y_dims[i + 1] * y_strides[i + 1];

      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    auto x_md = memory::desc(x_dims, OneDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, OneDNNGetDataType<YT>(), y_strides);
    auto out_md = memory::desc(out_ddims, OneDNNGetDataType<OT>(), out_strides);

    this->AcquireForwardPrimitiveDescriptor(x_md, y_md, out_md);
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const DenseTensor* input) {
    const YT* input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<YT>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(const OneDNNContext& dev_ctx,
                                                 DenseTensor* output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence DenseTensor size is bigger that
    // dst memory primitive size. So would we request less memory that is there
    // and it triggers an assertion.  So as there is no 'any' format here we can
    // leave default size of DenseTensor as computed in ComputeInferShape
    OT* ptr = dev_ctx.template Alloc<OT>(output);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

template <typename T>
inline void ExecuteMul(const OneDNNContext& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const std::vector<int64_t>& x_dims,
                       const std::vector<int64_t>& y_dims,
                       bool trans_x,
                       bool trans_y,
                       DenseTensor* out) {
  MatmulOneDNNHandler<T, T, T> handler(
      dev_ctx, x_dims, y_dims, trans_x, trans_y);

  const auto src_memory_p = handler.AcquireSrcMemory(&x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y);
  const auto dst_memory_p = handler.AcquireDstMemory(dev_ctx, out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto& astream = OneDNNContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  // This kernel is flattening dims so then we need to unflattened version
  // that should be set in out reshape require plain layout, but
  // MatmulV2MKLDNNHanlder enforces one so it should work
  auto reshape_dims = out->dims().size() != 0 ? common::vectorize(out->dims())
                                              : std::vector<int64_t>{1};
  out->set_mem_desc(dst_memory_p->get_desc().reshape(reshape_dims));
}

template <typename T, typename T_out>
inline void ExecuteMatmul(const OneDNNContext& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          const std::vector<int64_t>& x_dims,
                          const std::vector<int64_t>& y_dims,
                          bool trans_x,
                          bool trans_y,
                          DenseTensor* out) {
  MatmulOneDNNHandler<T, T, T_out> handler(
      dev_ctx, x_dims, y_dims, trans_x, trans_y);

  const auto src_memory_p = handler.AcquireSrcMemory(&x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y);
  const auto dst_memory_p = handler.AcquireDstMemory(dev_ctx, out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto& astream = OneDNNContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  auto reshape_dims = out->dims().size() != 0 ? common::vectorize(out->dims())
                                              : std::vector<int64_t>{1};
  out->set_mem_desc(dst_memory_p->get_desc().reshape(reshape_dims));
}

}  // namespace funcs
}  // namespace phi
