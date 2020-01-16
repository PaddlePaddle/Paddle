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
#include "paddle/fluid/operators/requantize_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;

template <typename T>
class ReQuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto scale_in = ctx.Attr<float>("Scale_in");
    auto scale_out = ctx.Attr<float>("Scale_out");
    auto* output = ctx.Output<Tensor>("Output");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    auto src_tz = paddle::framework::vectorize(input->dims());

    std::string key = platform::CreateKey(src_tz, scale_in, scale_out,
                                          ctx.OutputName("Output"));
    const std::string key_prim = key + "@reorder_p";
    const std::string key_src_mem = key + "@src_mem";
    const std::string key_dst_mem = key + "@dst_mem";

    std::shared_ptr<dnnl::memory> src_memory;
    std::shared_ptr<dnnl::memory> dst_memory;
    std::shared_ptr<reorder> reorder_p;
    reorder_p = std::static_pointer_cast<reorder>(dev_ctx.GetBlob(key_prim));

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    if (reorder_p == nullptr) {
      dnnl::primitive_attr attri;
      int mask = 0;
      float scale_shift = scale_out / scale_in;
      attri.set_output_scales(mask, {scale_shift});

      auto dst_tz = paddle::framework::vectorize(output->dims());
      dnnl::memory::data_type src_dt =
          paddle::framework::ToMKLDNNDataType(input->type());
      dnnl::memory::data_type dst_dt = src_dt;

      auto src_md =
          platform::MKLDNNMemDesc({src_tz}, src_dt, MKLDNNMemoryFormat::nhwc);
      src_memory = std::make_shared<dnnl::memory>(src_md, engine,
                                                  to_void_cast<T>(input_data));

      auto dst_md =
          platform::MKLDNNMemDesc({dst_tz}, dst_dt, MKLDNNMemoryFormat::nhwc);
      dst_memory = std::make_shared<dnnl::memory>(dst_md, engine,
                                                  to_void_cast<T>(output_data));

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
      dst_memory->set_data_handle(output_data);
    }

    dnnl::stream astream(engine);
    reorder_p->execute(astream, *src_memory, *dst_memory);
    astream.wait();

    output->set_layout(framework::DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(requantize, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ReQuantOpKernel<int8_t>, ops::ReQuantOpKernel<uint8_t>);
