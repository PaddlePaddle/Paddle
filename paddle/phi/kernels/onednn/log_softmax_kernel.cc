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

#include "paddle/phi/kernels/log_softmax_kernel.h"

<<<<<<< HEAD
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
=======
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
<<<<<<< HEAD
class LogSoftmaxMKLDNNHandler
    : public paddle::platform::
          MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward> {
 public:
  LogSoftmaxMKLDNNHandler(const dnnl::engine mkldnn_engine,
                          Place cpu_place,
                          const DenseTensor& x,
                          const int axis)
      : paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward>(
            mkldnn_engine, cpu_place) {
=======
class LogSoftmaxOneDNNHandler
    : public funcs::OneDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward> {
 public:
  LogSoftmaxOneDNNHandler(const dnnl::engine onednn_engine,
                          Place cpu_place,
                          const DenseTensor& x,
                          const int axis)
      : funcs::OneDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward>(
            onednn_engine, cpu_place) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_inference, x.mem_desc(), axis);
  }
};

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
<<<<<<< HEAD
  const auto& mkldnn_engine = dev_ctx.GetEngine();
  axis = axis >= 0 ? axis : x.dims().size() + axis;

  LogSoftmaxMKLDNNHandler<T> handler(
      mkldnn_engine, dev_ctx.GetPlace(), x, axis);
=======
  const auto& onednn_engine = dev_ctx.GetEngine();
  axis = axis >= 0 ? axis : x.dims().size() + axis;

  LogSoftmaxOneDNNHandler<T> handler(
      onednn_engine, dev_ctx.GetPlace(), x, axis);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto logsoftmax_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  logsoftmax_p->execute(
      astream, {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax,
                   OneDNN,
<<<<<<< HEAD
                   ALL_LAYOUT,
=======
                   ONEDNN,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                   phi::LogSoftmaxKernel,
                   float,
                   phi::dtype::bfloat16) {}
