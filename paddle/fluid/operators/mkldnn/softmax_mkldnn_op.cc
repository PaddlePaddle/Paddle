/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace paddle {
namespace operators {

using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using dnnl::memory;  // Note: paddle has also "memory" namespace
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::softmax_backward;
using dnnl::softmax_forward;
using dnnl::stream;
using platform::to_void_cast;

template <typename T>
class SoftmaxMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T,
                                               dnnl::softmax_forward,
                                               dnnl::softmax_backward> {
 public:
  SoftmaxMKLDNNHandler(const dnnl::engine mkldnn_engine,
                       platform::Place cpu_place,
                       const phi::DenseTensor* input,
                       phi::DenseTensor* output,
                       const int axis)
      : platform::MKLDNNHandlerNoCachingT<T,
                                          dnnl::softmax_forward,
                                          dnnl::softmax_backward>(mkldnn_engine,
                                                                  cpu_place) {
    PADDLE_ENFORCE_EQ(
        input->dims(),
        output->dims(),
        platform::errors::InvalidArgument(
            "The shape of input and output tensor must be identical."));

    this->AcquireForwardPrimitiveDescriptor(
        prop_kind::forward_scoring, input->mem_desc(), axis);
  }
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const phi::DenseTensor* input = ctx.Input<phi::DenseTensor>("X");
    phi::DenseTensor* output = ctx.Output<phi::DenseTensor>("Out");
    bool is_inplaced = input->IsSharedBufferWith(*output);

    const int axis =
        phi::funcs::CanonicalAxis(ctx.Attr<int>("axis"), input->dims().size());

    SoftmaxMKLDNNHandler<T> handler(
        mkldnn_engine, ctx.GetPlace(), input, output, axis);

    auto softmax_src_memory_p = handler.AcquireSrcMemory(input);
    // For Inplace src and and dst are the same memory object
    std::shared_ptr<dnnl::memory> softmax_dst_memory_p = nullptr;
    if (is_inplaced) {
      softmax_dst_memory_p = softmax_src_memory_p;
      output->mutable_data<T>(ctx.GetPlace());
    } else {
      softmax_dst_memory_p = handler.AcquireDstMemory(output);
    }
    auto softmax_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    softmax_p->execute(astream,
                       {{DNNL_ARG_SRC, *softmax_src_memory_p},
                        {DNNL_ARG_DST, *softmax_dst_memory_p}});
    astream.wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T* output_data = output->mutable_data<T>(ctx.GetPlace());
      std::for_each(output_data, &output_data[output->numel()], [](T& val) {
        val = std::max(val, static_cast<T>(exp(-64)));
      });
    }

    output->set_mem_desc(softmax_dst_memory_p->get_desc());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>,
                   ops::SoftmaxMKLDNNKernel<paddle::platform::bfloat16>);
