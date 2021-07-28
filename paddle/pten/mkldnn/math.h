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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/mkldnn/base.h"

namespace pt {

using MKLDNNDeviceContext = paddle::platform::MKLDNNDeviceContext;

template <typename T>
void Scale(const MKLDNNDeviceContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  bool is_inplaced = x.allocation() && x.allocation() == out->allocation();

  // TODO(chenweihang): add `name` into TensorMeta?
  ScaleMKLDNNHandler<T> handler(dev_ctx,
                                x,
                                /*unique_name=*/"X",
                                is_inplaced,
                                /*alpha=*/scale,
                                /*beta=*/bias,
                                bias_after_scale);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = MKLDNNDeviceContext::tls().get_stream();
  activation_p->execute(
      astream,
      {{MKLDNN_ARG_FROM, *src_memory_p}, {MKLDNN_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->mutable_meta()->layout = DataLayout::kMKLDNN;
  // TODO(chenweihang): we should use dynamic_cast get MKLDNNTensorMeta,
  // Is there any better way here?
  dynamic_cast<MKLDNNTensorMeta*>(out->mutable_meta())->format =
      paddle::platform::GetMKLDNNFormat(*dst_memory_p);
}

}  // namespace pt

#endif
