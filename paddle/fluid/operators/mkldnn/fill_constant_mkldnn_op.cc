/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class FillConstantMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  FillConstantMKLDNNHandler(Tensor* out,
                            dnnl::engine engine,
                            platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src0_md =
        dnnl::memory::desc({out->numel(), sizeof(T)},
                           platform::MKLDNNGetDataType<uint8_t>(),
                           dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attrs;
    attrs.set_scales(DNNL_ARG_SRC_0, /* mask = */ 0, {0.0f});

    this->AcquireForwardPrimitiveDescriptor(
        attrs, dnnl::algorithm::binary_add, src0_md, src1_md, src0_md);
  }

  static const dnnl::memory::desc src1_md;
};

template <typename T>
const dnnl::memory::desc FillConstantMKLDNNHandler<T>::src1_md(
    {1, sizeof(T)},
    platform::MKLDNNGetDataType<uint8_t>(),
    dnnl::memory::format_tag::ab);

template <typename T>
class FillConstantMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();

    auto* out = ctx.Output<Tensor>("Out");
    T fill_value = CalculateFillValue(ctx);

    auto shape = GetShape(ctx);
    out->Resize(shape);

    FillConstantMKLDNNHandler<T> handler(out, dnnl_engine, ctx.GetPlace());

    dnnl::memory constant_value_memory =
        dnnl::memory(FillConstantMKLDNNHandler<T>::src1_md,
                     dnnl_engine,
                     reinterpret_cast<uint8_t*>(&fill_value));

    auto src0_memory_p = handler.AcquireDstMemory(out);
    auto fill_constant_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    fill_constant_p->execute(astream,
                             {{DNNL_ARG_SRC_0, *src0_memory_p},
                              {DNNL_ARG_SRC_1, constant_value_memory},
                              {DNNL_ARG_DST, *src0_memory_p}});
    astream.wait();

    // src0_memory_p's md was just to allow the usage of a binary
    // primitive as a memset, and now we need to create a real one
    out->set_mem_desc({phi::vectorize(shape),
                       platform::MKLDNNGetDataType<T>(),
                       platform::GetPlainMKLDNNFormat(shape.size())});
  }

  T CalculateFillValue(const framework::ExecutionContext& ctx) const {
    const auto str_value = ctx.Attr<std::string>("str_value");
    const auto float_value = ctx.Attr<float>("value");

    T value;

    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<float>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<float>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        double tmp_value;
        convert_stream >> tmp_value;
        value = static_cast<T>(tmp_value);
      }
    }

    if (ctx.HasInput("ValueTensor")) {
      const auto* value_tensor = ctx.Input<Tensor>("ValueTensor");
      PADDLE_ENFORCE_EQ(
          value_tensor->numel(),
          1,
          platform::errors::InvalidArgument(
              "When use Tensor as value to set Tensor value in fill_constant, "
              "value input(ValueTensor) size must be 1, but got %d",
              value_tensor->numel()));
      value = value_tensor->data<T>()[0];
    }

    return value;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fill_constant,
                   MKLDNN,
                   paddle::platform::CPUPlace,
                   ops::FillConstantMKLDNNKernel<float>);
