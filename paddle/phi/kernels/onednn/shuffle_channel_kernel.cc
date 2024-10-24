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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
class ShuffleChannelMKLDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::shuffle_forward> {
 public:
  ShuffleChannelMKLDNNHandler(const phi::DenseTensor* x,
                              const int group,
                              const dnnl::engine engine,
                              phi::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::shuffle_forward>(
            engine, cpu_place) {
    static constexpr int channel_axis = 1;
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            x->mem_desc(),
                                            x->mem_desc(),
                                            channel_axis,
                                            group);
  }
};

template <typename T, typename Context>
void ShuffleChannelMKLDNNKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                int group,
                                DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  // oneDNN handles group using C/g instead of g
  const int tmp_group = x.dims()[1] / group;

  ShuffleChannelMKLDNNHandler<T> handler(
      &x, tmp_group, onednn_engine, dev_ctx.GetPlace());

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto shuffle_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  shuffle_p->execute(
      astream, {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(shuffle_channel,
                   OneDNN,
                   ONEDNN,
                   phi::ShuffleChannelMKLDNNKernel,
                   float,
                   phi::dtype::bfloat16) {}
