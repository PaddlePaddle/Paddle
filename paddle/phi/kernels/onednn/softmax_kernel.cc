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

#include "paddle/phi/kernels/softmax_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <template <typename> class H, typename T, typename Context>
void SoftmaxExecute(bool is_inplace,
                    H<T>* handler,
                    const DenseTensor& x,
                    DenseTensor* out,
                    const Context& dev_ctx) {
  auto src_memory_p = handler->AcquireSrcMemory(&x);

  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplace) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler->AcquireDstMemory(out);
  }

  auto softmax_p = handler->AcquireForwardPrimitive();
  auto& astream = OneDNNContext::tls().get_stream();
  softmax_p->execute(
      astream, {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;
  if (!is_test) {
    T* out_data = dev_ctx.template Alloc<T>(out);
    std::for_each(out_data, &out_data[out->numel()], [](T& val) {
      val = std::max(val, static_cast<T>(exp(-64)));
    });
  }
  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int axis,
                   DenseTensor* out) {
  bool is_inplace = x.IsSharedBufferWith(*out);

  if (phi::funcs::is_int8<T>()) {
    PADDLE_ENFORCE(!is_inplace,
                   phi::errors::Unimplemented(
                       "Inplace not implemeneted for int8 onednn softmax."));

    out->set_mem_desc(x.mem_desc());
    funcs::SoftmaxV2OneDNNHandler<T> handler(
        dev_ctx.GetEngine(), dev_ctx.GetPlace(), axis, &x, out);
    SoftmaxExecute(is_inplace, &handler, x, out, dev_ctx);
  } else {
    funcs::SoftmaxOneDNNHandler<T> handler(
        dev_ctx.GetEngine(), dev_ctx.GetPlace(), axis, &x, out);
    SoftmaxExecute(is_inplace, &handler, x, out, dev_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(softmax,
                   OneDNN,
                   ONEDNN,
                   phi::SoftmaxKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
