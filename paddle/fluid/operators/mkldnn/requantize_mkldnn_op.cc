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

#include <iterator>  // NOLINT
#include "dnnl.hpp"  // NOLINT
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/requantize_op.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::reorder;

namespace {

inline uint8_t clip_to_uint8(float x) {
  return std::max(0L, std::min(255L, std::lround(x)));
}

}  // namespace

template <typename T>
class ReQuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto scale_in = ctx.Attr<float>("Scale_in");
    auto shift_in = ctx.Attr<float>("Shift_in");
    auto scale_out = ctx.Attr<float>("Scale_out");
    auto shift_out = ctx.Attr<float>("Shift_out");
    bool with_shift = shift_in != 0.0f || shift_out != 0.0f;
    auto* output = ctx.Output<phi::DenseTensor>("Output");

    PADDLE_ENFORCE_NE(
        scale_in,
        0.0f,
        platform::errors::InvalidArgument("Scale of input cannot be 0.0"));
    PADDLE_ENFORCE_NE(
        scale_out,
        0.0f,
        platform::errors::InvalidArgument("Scale of output cannot be 0.0"));
    if (shift_in != 0.0f) {
      PADDLE_ENFORCE_EQ(
          input->dtype(),
          DataType::UINT8,
          platform::errors::Unimplemented("Requantize does not support nonzero "
                                          "shift for signed input."));
    }

    auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();

    auto src_tz = phi::vectorize(input->dims());

    auto src_paddle_dt = input->dtype();
    auto dst_paddle_dt = with_shift ? DataType::UINT8 : src_paddle_dt;

    auto xstrides = input->mem_desc().data.format_desc.blocking.strides;
    std::vector<dnnl_dim_t> vstrides(xstrides,
                                     xstrides + input->mem_desc().data.ndims);

    dnnl::primitive_attr attrs;
    int mask = 0;
    float reorder_scale = scale_out / scale_in;
    attrs.set_output_scales(mask, {reorder_scale});
    if (with_shift) {
      uint8_t reorder_shift =
          clip_to_uint8(shift_out - reorder_scale * shift_in);
      attrs.set_zero_points(
          DNNL_ARG_DST, mask, {static_cast<int32_t>(reorder_shift)});
    }

    phi::funcs::ReorderOneDNNHandler reorder_handler(
        src_tz,
        src_paddle_dt,
        phi::funcs::ToOneDNNDataType(src_paddle_dt),
        dst_paddle_dt,
        phi::funcs::ToOneDNNDataType(dst_paddle_dt),
        dev_ctx.GetEngine());

    auto src_memory_p = reorder_handler.AcquireSrcMemory(
        input->mem_desc(), phi::funcs::to_void_cast(input->data<T>()));
    auto dst_memory_p = reorder_handler.AcquireDstMemory(
        output, src_tz, vstrides, dev_ctx.GetPlace());

    auto reorder_p =
        reorder_handler.AcquireReorder(dst_memory_p, src_memory_p, attrs);

    auto& astream = phi::OneDNNContext::tls().get_stream();
    reorder_p->execute(astream, *src_memory_p, *dst_memory_p);
    astream.wait();

    output->set_mem_desc(dst_memory_p->get_desc());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(requantize,
                   MKLDNN,
                   ::phi::CPUPlace,
                   ops::ReQuantOpKernel<int8_t>,
                   ops::ReQuantOpKernel<uint8_t>,
                   ops::ReQuantOpKernel<paddle::platform::bfloat16>);
