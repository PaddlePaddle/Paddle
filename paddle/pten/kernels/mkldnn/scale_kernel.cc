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

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/pten/backends/mkldnn/mkldnn_context.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T, typename Context>
class ScaleMKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<T,
                                                       dnnl::eltwise_forward,
                                                       dnnl::eltwise_backward> {
 public:
  ScaleMKLDNNHandler(dnnl::algorithm algorithm,
                     const Context& dev_ctx,
                     const DenseTensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale)
      : paddle::platform::MKLDNNHandlerNoCachingT<T,
                                                  dnnl::eltwise_forward,
                                                  dnnl::eltwise_backward>(
            dev_ctx.GetEngine(), dev_ctx.GetPlace()) {
    float alpha = scale.to<float>();
    float beta = bias;
    // if bias_after_scale == true
    //   out = scale*X + bias
    // else
    //   out = scale*(X + bias) = scale*X + scale*bias
    if (!bias_after_scale) {
      beta *= alpha;
    }

    PADDLE_ENFORCE(
        x.dims().size() >= 1 || x.dims().size() <= 6,
        pten::errors::Unimplemented("Input dimension size can be 1, 2, 3, 4, "
                                    "5, or 6, but now the dimension size is",
                                    x.dims().size()));

    auto src_tz = pten::framework::vectorize<int64_t>(x.dims());
    auto src_fmt =
        src_tz.size() == 2 ? paddle::MKLDNNMemoryFormat::nc : x.format();
    auto md = dnnl::memory::desc(
        src_tz, paddle::platform::MKLDNNGetDataType<T>(), src_fmt);

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training, algorithm, md, alpha, beta);
  }
};

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  bool is_inplaced = x.IsSharedBufferWith(*out);

  ScaleMKLDNNHandler<T, Context> handler(dnnl::algorithm::eltwise_linear,
                                         dev_ctx,
                                         x,
                                         scale,
                                         bias,
                                         bias_after_scale);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = pten::MKLDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_layout(paddle::framework::DataLayout::kMKLDNN);
  out->set_format(paddle::platform::GetMKLDNNFormat(*dst_memory_p));
}

}  // namespace pten

PT_REGISTER_KERNEL(
    scale, MKLDNN, MKLDNN, pten::ScaleKernel, float, pten::dtype::bfloat16) {}
