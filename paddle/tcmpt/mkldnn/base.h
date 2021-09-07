/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/tcmpt/core/mkldnn_dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace pt {

using MKLDNNDContext = paddle::platform::MKLDNNDeviceContext;

// TODO(chenweihang): the handlers in `mkldnn_reuse.h` are coupled to
// `ExecutionContext`, refactoring that may be a big project!

template <typename T>
class ScaleMKLDNNHandler : public paddle::platform::MKLDNNHandlerNoCachingT<
                               T,
                               mkldnn::eltwise_forward,
                               mkldnn::eltwise_backward> {
 public:
  ScaleMKLDNNHandler(const mkldnn::engine& engine,
                     const pt::MKLDNNDenseTensor& in_x,
                     float alpha,
                     float beta,
                     bool bias_after_scale)
      : paddle::platform::MKLDNNHandlerNoCachingT<T,
                                                  mkldnn::eltwise_forward,
                                                  mkldnn::eltwise_backward>(
            engine, in_x.place()) {
    if (!bias_after_scale) {
      beta *= alpha;
    }

    PADDLE_ENFORCE(in_x.dims().size() >= 1 || in_x.dims().size() <= 6,
                   paddle::platform::errors::Unimplemented(
                       "Input dimension size can be 1, 2, 3, 4, "
                       "5, or 6, but now the dimension size is",
                       in_x.dims().size()));

    auto src_tz = paddle::framework::vectorize<int64_t>(in_x.dims());
    auto src_fmt =
        src_tz.size() == 2 ? paddle::MKLDNNMemoryFormat::nc : in_x.format();
    auto md = mkldnn::memory::desc(
        src_tz, paddle::platform::MKLDNNGetDataType<T>(), src_fmt);

    this->AcquireForwardPrimitiveDescriptor(mkldnn::prop_kind::forward_training,
                                            mkldnn::algorithm::eltwise_linear,
                                            md,
                                            alpha,
                                            beta);
  }
};

}  // namespace pt

#endif
