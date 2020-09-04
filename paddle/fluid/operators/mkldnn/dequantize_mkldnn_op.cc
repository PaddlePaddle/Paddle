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
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/dequantize_op.h"
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

template <typename T>
class DeQuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto scale_data = ctx.Attr<float>("Scale");
    auto* output = ctx.Output<Tensor>("Output");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    const T* input_data = input->data<T>();
    float* output_data = output->mutable_data<float>(ctx.GetPlace());
    std::vector<float> reorder_scale = {1.0f / scale_data};

    auto src_tz = paddle::framework::vectorize<int64_t>(input->dims());
    auto dst_tz = paddle::framework::vectorize<int64_t>(output->dims());
    mkldnn::memory::data_type src_dt =
        paddle::framework::ToMKLDNNDataType(input->type());
    MKLDNNMemoryFormat src_fmt = input->format();
    std::string key = platform::CreateKey(platform::ThreadIDasStr(), src_dt,
                                          src_tz, ctx.OutputName("Output"));
    const std::string key_prim = key + "@r";
    const std::string key_src_mem = key + "@s";
    const std::string key_dst_mem = key + "@d";

    std::shared_ptr<mkldnn::memory> src_memory;
    std::shared_ptr<mkldnn::memory> dst_memory;
    std::shared_ptr<reorder> reorder_p;
    reorder_p = std::static_pointer_cast<reorder>(dev_ctx.GetBlob(key_prim));

    if (reorder_p == nullptr) {
      mkldnn::primitive_attr attri;
      int mask = 0;
      attri.set_output_scales(mask, reorder_scale);

      auto src_md = platform::MKLDNNMemDesc({src_tz}, src_dt, src_fmt);
      src_memory = std::make_shared<mkldnn::memory>(
          src_md, engine, to_void_cast<T>(input_data));

      auto dst_md =
          platform::MKLDNNMemDesc({dst_tz}, memory::data_type::f32,
                                  platform::MKLDNNFormatForSize(
                                      dst_tz.size(), MKLDNNMemoryFormat::nchw));

      dst_memory = std::make_shared<mkldnn::memory>(
          dst_md, engine, to_void_cast<float>(output_data));

      auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
          new reorder::primitive_desc(*src_memory, *dst_memory, attri));
      reorder_p = std::shared_ptr<reorder>(new reorder(*reorder_pd));
      dev_ctx.SetBlob(key_prim, reorder_p);
      dev_ctx.SetBlob(key_src_mem, src_memory);
      dev_ctx.SetBlob(key_dst_mem, dst_memory);
    } else {
      src_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_src_mem));
      src_memory->set_data_handle(to_void_cast<T>(input_data));

      dst_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_dst_mem));
      dst_memory->set_data_handle(output->mutable_data<float>(ctx.GetPlace()));
    }

    mkldnn::stream astream(engine);
    reorder_p->execute(astream, *src_memory, *dst_memory);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(dequantize, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::DeQuantOpKernel<uint8_t>, ops::DeQuantOpKernel<int8_t>);
