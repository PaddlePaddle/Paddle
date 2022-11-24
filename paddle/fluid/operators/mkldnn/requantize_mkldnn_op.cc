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
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/requantize_op.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::reorder;
using Tensor = phi::DenseTensor;

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
<<<<<<< HEAD
    auto* output = ctx.Output<Tensor>("Output");
=======
    auto* output = ctx.Output<phi::DenseTensor>("Output");
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

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

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    auto src_tz = phi::vectorize(input->dims());

    auto src_paddle_dt = input->dtype();
    auto dst_paddle_dt = with_shift ? DataType::UINT8 : src_paddle_dt;

<<<<<<< HEAD
    std::string key = platform::CreateKey(
        dev_ctx, src_tz, scale_in, scale_out, ctx.OutputName("Output"));
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
    const std::string key_prim = key + "@r";
    const std::string key_src_mem = key + "@s";
    const std::string key_dst_mem = key + "@d";

    std::shared_ptr<dnnl::memory> src_memory;
    std::shared_ptr<dnnl::memory> dst_memory;
    std::shared_ptr<reorder> reorder_p;
    reorder_p = std::static_pointer_cast<reorder>(dev_ctx.GetBlob(key_prim));

    const T* input_data = input->data<T>();

    if (reorder_p == nullptr) {
      auto dst_tz = phi::vectorize(output->dims());
      auto src_dt = framework::ToMKLDNNDataType(
          framework::TransToProtoVarType(input->dtype()));
      auto dst_dt = with_shift ? framework::MKLDNNDataType::u8 : src_dt;

      auto src_md = platform::MKLDNNMemDesc({src_tz}, src_dt, input->format());
      src_memory = std::make_shared<dnnl::memory>(
          src_md, engine, to_void_cast<T>(input_data));
      auto dst_md = platform::MKLDNNMemDesc({dst_tz}, dst_dt, input->format());

      dnnl::primitive_attr attri;
      int mask = 0;
      attri.set_output_scales(mask, {reorder_scale});
      if (with_shift) {
        dnnl::post_ops post_operations;
        post_operations.append_sum();
        attri.set_post_ops(post_operations);
        uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
        uint8_t reorder_shift =
            clip_to_uint8(shift_out - reorder_scale * shift_in);
        std::memset(output_data, reorder_shift, output->numel());
        dst_memory = std::make_shared<dnnl::memory>(
            dst_md, engine, to_void_cast<uint8_t>(output_data));
      } else {
        T* output_data = output->mutable_data<T>(ctx.GetPlace());
        dst_memory = std::make_shared<dnnl::memory>(
            dst_md, engine, to_void_cast<T>(output_data));
      }

      auto reorder_pd =
          reorder::primitive_desc(*src_memory, *dst_memory, attri);
      reorder_p = std::make_shared<reorder>(reorder_pd);

      dev_ctx.SetBlob(key_prim, reorder_p);
      dev_ctx.SetBlob(key_src_mem, src_memory);
      dev_ctx.SetBlob(key_dst_mem, dst_memory);
    } else {
      src_memory =
          std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(key_src_mem));
      src_memory->set_data_handle(to_void_cast<T>(input_data));

      dst_memory =
          std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(key_dst_mem));
      if (with_shift) {
        uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
        uint8_t reorder_shift =
            clip_to_uint8(shift_out - reorder_scale * shift_in);
        std::memset(output_data, reorder_shift, output->numel());
        dst_memory->set_data_handle(output_data);

      } else {
        T* output_data = output->mutable_data<T>(ctx.GetPlace());
        dst_memory->set_data_handle(output_data);
      }
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
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
                   ::paddle::platform::CPUPlace,
                   ops::ReQuantOpKernel<int8_t>,
                   ops::ReQuantOpKernel<uint8_t>,
                   ops::ReQuantOpKernel<paddle::platform::bfloat16>);
