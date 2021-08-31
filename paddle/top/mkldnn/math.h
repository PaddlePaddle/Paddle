/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_MKLDNN

#include "paddle/top/core/mkldnn_dense_tensor.h"
#include "paddle/top/mkldnn/base.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pt {

using MKLDNNDContext = paddle::platform::MKLDNNDeviceContext;

template <typename T>
void Scale(const MKLDNNDContext& dev_ctx,
           const MKLDNNDenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           MKLDNNDenseTensor* out) {
  const auto mkldnn_engine = dev_ctx.GetEngine();

  ScaleMKLDNNHandler<T> handler(mkldnn_engine,
                                x,
                                /*alpha=*/scale,
                                /*beta=*/bias,
                                bias_after_scale);

  bool is_inplaced = x.allocation() && x.allocation() == out->allocation();

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p =
      is_inplaced ? src_memory_p : handler.AcquireDstMemory(out);
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = MKLDNNDContext::tls().get_stream();
  activation_p->execute(
      astream,
      {{MKLDNN_ARG_FROM, *src_memory_p}, {MKLDNN_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->mutable_meta()->layout = DataLayout::kMKLDNN;
  // TODO(chenweihang): format is also meta info, how to deal with here?
  out->set_format(paddle::platform::GetMKLDNNFormat(*dst_memory_p));
}

}  // namespace pt

#endif
