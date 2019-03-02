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

#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/quantize_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

std::string CreateKey(const paddle::framework::ExecutionContext& ctx,
                      const std::vector<int>& src_tz, const float scale_data,
                      const bool is_negative) {
  std::string key;
  key.reserve(platform::MKLDNNHandler::MaxKeyLength);
  platform::MKLDNNHandler::AppendKeyDims(&key, src_tz);
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(scale_data));
  platform::MKLDNNHandler::AppendKey(&key, std::to_string(is_negative));
  platform::MKLDNNHandler::AppendKey(&key, ctx.op().Output("Output"));
  return key;
}

template <typename T>
class QuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto scale_data = ctx.Attr<float>("Scale");
    auto* output = ctx.Output<Tensor>("Output");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    std::vector<primitive> pipeline;
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    const T* input_data = input->data<T>();

    bool is_negative = ctx.Attr<bool>("is_negative_input");
    std::string key = CreateKey(ctx, src_tz, scale_data, is_negative);
    const std::string key_prim = key + "@reorder_p";
    const std::string key_src_mem = key + "@src_mem";
    const std::string key_dst_mem = key + "@dst_mem";

    std::shared_ptr<mkldnn::memory> src_memory;
    std::shared_ptr<mkldnn::memory> dst_memory;
    std::shared_ptr<reorder> reorder_p;
    reorder_p = std::static_pointer_cast<reorder>(dev_ctx.GetBlob(key_prim));

    if (reorder_p == nullptr) {
      mkldnn::primitive_attr attri;
      int mask = 0;
      attri.set_output_scales(mask, {scale_data});

      auto src_md = platform::MKLDNNMemDesc({src_tz}, memory::data_type::f32,
                                            input->format());
      auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);
      src_memory =
          std::make_shared<mkldnn::memory>(src_pd, to_void_cast<T>(input_data));
      std::shared_ptr<primitive::at> src_memory_p =
          std::shared_ptr<primitive::at>(new primitive::at(*src_memory));

      std::shared_ptr<mkldnn::memory::primitive_desc> dst_pd;
      if (is_negative) {
        platform::ConvMKLDNNHandler::SetDstMemory<int8_t>(
            ctx, output, dst_tz, engine, dst_pd, dst_memory);
      } else {
        platform::ConvMKLDNNHandler::SetDstMemory<uint8_t>(
            ctx, output, dst_tz, engine, dst_pd, dst_memory);
      }
      auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
          new reorder::primitive_desc(src_pd, *dst_pd, attri));
      reorder_p = std::shared_ptr<reorder>(
          new reorder(*reorder_pd, *src_memory_p, *dst_memory));

      dev_ctx.SetBlob(key_prim, reorder_p);
      dev_ctx.SetBlob(key_src_mem, src_memory);
      dev_ctx.SetBlob(key_dst_mem, dst_memory);
    } else {
      src_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_src_mem));
      src_memory->set_data_handle(to_void_cast<T>(input_data));

      dst_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_dst_mem));
      auto place = ctx.GetPlace();
      if (is_negative) {
        dst_memory->set_data_handle(output->mutable_data<int8_t>(place));
      } else {
        dst_memory->set_data_handle(output->mutable_data<uint8_t>(place));
      }
    }

    pipeline.push_back(*reorder_p);
    stream(stream::kind::eager).submit(pipeline).wait();
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

// TODO(Xiaoli) Support FP32->S8 quantization.

REGISTER_OP_KERNEL(quantize, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::QuantOpKernel<float>);
