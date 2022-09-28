/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mkldnn/softplus_mkldnn_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;
using framework::DataLayout;

using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    Functor functor;
    functor(ctx);
  }
};

template <typename T>
struct SoftplusMKLDNNFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    custom_softplus_eltwise_forward<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_FWD_ACTIVATION_MKLDNN_KERNEL(act_type, functor) \
  REGISTER_OP_KERNEL(                                            \
      act_type,                                                  \
      MKLDNN,                                                    \
      ::paddle::platform::CPUPlace,                              \
      ops::MKLDNNActivationKernel<ops::functor<float>>,          \
      ops::MKLDNNActivationKernel<ops::functor<paddle::platform::bfloat16>>);

REGISTER_FWD_ACTIVATION_MKLDNN_KERNEL(softplus, SoftplusMKLDNNFunctor);
