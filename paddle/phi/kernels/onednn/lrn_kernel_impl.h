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
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
class LRNOneDNNHandler
    : public phi::funcs::
          OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward> {
 public:
  LRNOneDNNHandler(int n,
                   T k_in,
                   T alpha_in,
                   T beta_in,
                   bool is_test,
                   const dnnl::engine onednn_engine,
                   phi::Place cpu_place,
                   const phi::DenseTensor* input)

      : phi::funcs::
            OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward>(
                onednn_engine, cpu_place) {
    // MKL-DNN implements LRN in a caffe way:
    // http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
    // Where sum of squares is divided by size of normalization window
    // this is not the case for PaddlePaddle LRN.
    // Hence we need to compensate for this difference by
    // multipliing alpha by size of window(n)
    const float alpha = static_cast<float>(alpha_in) * static_cast<float>(n);
    const float beta = static_cast<float>(beta_in);
    const float k = static_cast<float>(k_in);

    this->AcquireForwardPrimitiveDescriptor(
        is_test ? dnnl::prop_kind::forward_inference
                : dnnl::prop_kind::forward_training,
        dnnl::algorithm::lrn_across_channels,
        input->mem_desc(),
        input->mem_desc(),
        n,
        alpha,
        beta,
        k);
  }

  LRNOneDNNHandler(int n,
                   T k_in,
                   T alpha_in,
                   T beta_in,
                   bool is_test,
                   const dnnl::engine onednn_engine,
                   phi::Place cpu_place,
                   const phi::DenseTensor* in_x,
                   const phi::DenseTensor* out_grad,
                   phi::DenseTensor* in_x_grad)
      : phi::funcs::
            OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward>(
                onednn_engine, cpu_place) {
    PADDLE_ENFORCE_EQ(
        is_test,
        false,
        common::errors::PreconditionNotMet(
            "is_test attribute should be set to False in training phase."));

    const float alpha = static_cast<float>(alpha_in) * static_cast<float>(n);
    const float beta = static_cast<float>(beta_in);
    const float k = static_cast<float>(k_in);

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training,
        dnnl::algorithm::lrn_across_channels,
        in_x->mem_desc(),
        in_x->mem_desc(),
        n,
        alpha,
        beta,
        k);

    this->AcquireBackwardPrimitiveDescriptor(
        dnnl::algorithm::lrn_across_channels,
        out_grad->mem_desc(),
        out_grad->mem_desc(),
        in_x->mem_desc(),
        n,
        alpha,
        beta,
        k);
  }

  std::shared_ptr<dnnl::memory> AcquireWorkspaceMemory(
      phi::DenseTensor* workspace, const Context& dev_ctx) {
    T* ptr = dev_ctx.template HostAlloc<T>(
        workspace, this->fwd_pd_->workspace_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->workspace_desc(),
                                            ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireBackwardWorkspaceMemory(
      const phi::DenseTensor* workspace) {
    const T* workspace_data = workspace->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->workspace_desc(),
        phi::funcs::to_void_cast<T>(workspace_data));
  }
};

template <typename T, typename Context>
void LRNMKLDNNOpKernel(const Context& dev_ctx,
                       const DenseTensor& x_in,
                       int n,
                       T k,
                       T alpha,
                       T beta,
                       const std::string& data_format,
                       DenseTensor* out,
                       DenseTensor* mid_out) {
  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;
  const bool is_float_type = std::is_same<T, float>::value;
  PADDLE_ENFORCE_EQ(
      is_float_type,
      true,
      common::errors::PreconditionNotMet("DNNL LRN must use float data."));
  bool eq_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU;
  PADDLE_ENFORCE_EQ(eq_place,
                    true,
                    common::errors::PreconditionNotMet(
                        "Operator DNNL LRN must use CPUPlace"));
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto x = &x_in;
  auto mid = mid_out;

  LRNOneDNNHandler<T, Context> handler(
      n, k, alpha, beta, is_test, onednn_engine, dev_ctx.GetPlace(), x);

  auto src_memory = handler.AcquireSrcMemory(x);
  auto dst_memory = handler.AcquireDstMemory(out);

  auto lrn_p = handler.AcquireForwardPrimitive();

  auto workspace_memory = handler.AcquireWorkspaceMemory(mid, dev_ctx);
  mid->set_layout(phi::DataLayout::ONEDNN);

  auto& astream = OneDNNContext::tls().get_stream();
  if (!workspace_memory->get_desc().is_zero()) {
    mid->set_mem_desc(workspace_memory->get_desc());
    lrn_p->execute(astream,
                   {{DNNL_ARG_SRC, *src_memory},
                    {DNNL_ARG_DST, *dst_memory},
                    {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    lrn_p->execute(astream,
                   {{DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *dst_memory}});
  }
  astream.wait();

  out->set_mem_desc(dst_memory->get_desc());
}

template <typename T, typename Context>
void LRNMKLDNNGradOpKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& out,
                           const DenseTensor& mid_out,
                           const DenseTensor& out_grad_in,
                           int n,
                           T k,
                           T alpha,
                           T beta,
                           const std::string& data_format,
                           DenseTensor* x_grad) {
  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;
  const bool is_float_type = std::is_same<T, float>::value;
  PADDLE_ENFORCE_EQ(is_float_type,
                    true,
                    common::errors::PreconditionNotMet(
                        "DNNL LRN GradOpKernel must use float data."));
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
                    true,
                    common::errors::PreconditionNotMet(
                        "Operator DNNL LRNGrad must use CPUPlace"));

  auto in_x = &x;
  auto mid = &mid_out;
  auto out_grad = &out_grad_in;
  auto in_x_grad = x_grad;

  const auto& onednn_engine = dev_ctx.GetEngine();

  LRNOneDNNHandler<T, Context> handler(n,
                                       k,
                                       alpha,
                                       beta,
                                       is_test,
                                       onednn_engine,
                                       dev_ctx.GetPlace(),
                                       in_x,
                                       out_grad,
                                       in_x_grad);

  auto src_memory = handler.AcquireSrcMemory(in_x);
  auto workspace = handler.AcquireBackwardWorkspaceMemory(mid);
  auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

  auto lrn_bwd = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  lrn_bwd->execute(astream,
                   {{DNNL_ARG_SRC, *src_memory},
                    {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                    {DNNL_ARG_DIFF_SRC, *diff_src_memory},
                    {DNNL_ARG_WORKSPACE, *workspace}});
  astream.wait();

  in_x_grad->set_mem_desc(diff_src_memory->get_desc());
}
}  // namespace phi
