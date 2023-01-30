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

<<<<<<< HEAD
#include "paddle/fluid/operators/quantize_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
=======
#include "dnnl.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/quantize_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using dnnl::reorder;
<<<<<<< HEAD
using dnnl::stream;
using phi::DataLayout;
=======
using platform::to_void_cast;
using Tensor = framework::Tensor;
using dnnl::stream;
using framework::DataLayout;
using platform::GetMKLDNNFormat;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

template <typename T>
class QuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
<<<<<<< HEAD
    auto* x = ctx.Input<phi::DenseTensor>("Input");
    auto* out = ctx.Output<phi::DenseTensor>("Output");
=======
    auto* x = ctx.Input<Tensor>("Input");
    auto* out = ctx.Output<Tensor>("Output");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    const auto quantization_scale = ctx.Attr<float>("Scale");
    const auto quantization_shift = ctx.Attr<float>("Shift");
    const bool with_scale = quantization_scale != 1.0f;
    const bool with_shift = quantization_shift != 0.0f;

    PADDLE_ENFORCE_NE(quantization_scale,
                      0.0f,
                      platform::errors::InvalidArgument(
                          "Quantization scale must be different than 0.0f"));
    PADDLE_ENFORCE(quantization_shift <= 255 && quantization_shift >= 0,
                   platform::errors::InvalidArgument(
                       "Quantization shift must be lower or equal to ",
                       "255 and greater or equal to 0, but got %f",
                       quantization_shift));

<<<<<<< HEAD
    auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();
=======
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    auto x_tz = phi::vectorize<int64_t>(x->dims());

    const bool is_negative_input = ctx.Attr<bool>("is_negative_input");
    const bool bfloat16 = ctx.Attr<bool>("bfloat16");

    dnnl::primitive_attr attrs;
    static constexpr int32_t mask = 0;

    if (with_scale) {
      attrs.set_output_scales(mask, {quantization_scale});
    }

    if (with_shift) {
      attrs.set_zero_points(
          DNNL_ARG_DST, mask, {static_cast<int32_t>(quantization_shift)});
    }

<<<<<<< HEAD
    auto x_type = phi::funcs::ToOneDNNDataType(x->dtype());
    DataType out_dtype;

    if (bfloat16) {
      out_dtype = DataType::BFLOAT16;
    } else if (is_negative_input && !with_shift) {
      out_dtype = DataType::INT8;
    } else {
      out_dtype = DataType::UINT8;
    }

    auto out_type = phi::funcs::ToOneDNNDataType(out_dtype);

    phi::funcs::ReorderOneDNNHandler reorder_handler(
        x_tz, x->dtype(), x_type, out_dtype, out_type, dev_ctx.GetEngine());

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), phi::funcs::to_void_cast(x->data<T>()));
=======
    framework::proto::VarType::Type x_paddle_dtype =
        framework::TransToProtoVarType(x->dtype());
    framework::proto::VarType::Type out_paddle_dtype;

    if (bfloat16) {
      out_paddle_dtype = framework::proto::VarType::BF16;
    } else if (is_negative_input && !with_shift) {
      out_paddle_dtype = framework::proto::VarType::INT8;
    } else {
      out_paddle_dtype = framework::proto::VarType::UINT8;
    }

    platform::ReorderMKLDNNHandler reorder_handler(
        x_tz,
        x_paddle_dtype,
        framework::ToMKLDNNDataType(x_paddle_dtype),
        out_paddle_dtype,
        framework::ToMKLDNNDataType(out_paddle_dtype),
        dev_ctx.GetEngine());

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), platform::to_void_cast(x->data<T>()));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, x->mem_desc(), dev_ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(
        reorder_dst_memory_p, reorder_src_memory_p, attrs);

<<<<<<< HEAD
    auto& astream = phi::OneDNNContext::tls().get_stream();
=======
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    out->set_mem_desc(reorder_dst_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(quantize,
                   MKLDNN,
<<<<<<< HEAD
                   ::phi::CPUPlace,
=======
                   ::paddle::platform::CPUPlace,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                   ops::QuantOpKernel<float>);
