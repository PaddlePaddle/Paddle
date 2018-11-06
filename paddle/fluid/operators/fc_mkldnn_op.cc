/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T>
class FCMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<Tensor>("Out");

    std::vector<int> fc_src_tz =
        paddle::framework::vectorize2int(input->dims());
    std::vector<int> fc_weights_tz =
        paddle::framework::vectorize2int(w->dims());
    std::vector<int> fc_dst_tz =
        paddle::framework::vectorize2int(output->dims());

    // MKLDNN requires weights layout to be column major.
    // The values have already been transposed, but the shape needs to be fixed.
    // It cannot be done during an earlier stage since InferShape verifies
    // dimensions assuming the weights weren't transposed.
    std::swap(fc_weights_tz[0], fc_weights_tz[1]);

    auto fc_usr_src_md = platform::MKLDNNMemDesc(
        fc_src_tz, platform::MKLDNNGetDataType<T>(), input->format());
    auto fc_usr_src_memory_pd =
        memory::primitive_desc(fc_usr_src_md, mkldnn_engine);
    auto fc_usr_src_memory =
        memory(fc_usr_src_memory_pd, to_void_cast<T>(input->data<T>()));

    auto fc_src_memory = fc_usr_src_memory;
    auto fc_src_md = fc_usr_src_md;

    // flatten the input dimensions to 2d if necessary
    if (input->dims().size() == 4) {
      fc_src_md =
          platform::MKLDNNMemDesc(fc_src_tz, platform::MKLDNNGetDataType<T>(),
                                  mkldnn::memory::format::nchw);
      auto fc_src_memory_pd =
          memory::primitive_desc(fc_usr_src_md, mkldnn_engine);
      auto fc_src_memory = memory(fc_usr_src_memory_pd);
      auto reorder = mkldnn::reorder(fc_usr_src_memory, fc_src_memory);
      std::vector<mkldnn::primitive> pipeline{reorder};
      stream(stream::kind::eager).submit(pipeline).wait();
      fc_src_tz = {fc_src_tz[0], fc_src_tz[1] * fc_src_tz[2] * fc_src_tz[3]};

      fc_src_md =
          platform::MKLDNNMemDesc(fc_src_tz, platform::MKLDNNGetDataType<T>(),
                                  mkldnn::memory::format::nc);
      fc_src_memory_pd = memory::primitive_desc(fc_usr_src_md, mkldnn_engine);
      fc_src_memory =
          memory(fc_usr_src_memory_pd, fc_src_memory.get_data_handle());
    }

    auto fc_weights_md =
        platform::MKLDNNMemDesc(fc_weights_tz, platform::MKLDNNGetDataType<T>(),
                                mkldnn::memory::format::oi);
    auto fc_weights_memory_pd =
        memory::primitive_desc(fc_weights_md, mkldnn_engine);
    auto fc_weights_memory =
        memory(fc_weights_memory_pd, to_void_cast<T>(w->data<T>()));

    auto fc_dst_md = platform::MKLDNNMemDesc(fc_dst_tz, mkldnn::memory::f32,
                                             mkldnn::memory::format::any);

    std::shared_ptr<memory> fc_bias_memory_p;
    std::shared_ptr<inner_product_forward::desc> fc_desc_p;
    if (bias) {
      std::vector<int> fc_bias_tz =
          paddle::framework::vectorize2int(bias->dims());
      auto fc_bias_md = platform::MKLDNNMemDesc(
          fc_bias_tz, platform::MKLDNNGetDataType<T>(), bias->format());
      auto fc_bias_memory_pd =
          memory::primitive_desc(fc_bias_md, mkldnn_engine);
      fc_bias_memory_p.reset(
          new memory(fc_bias_memory_pd, to_void_cast<T>(bias->data<T>())));

      fc_desc_p.reset(new inner_product_forward::desc(
          prop_kind::forward, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md));
    } else {
      fc_desc_p.reset(new inner_product_forward::desc(
          prop_kind::forward, fc_src_md, fc_weights_md, fc_dst_md));
    }
    auto fc_prim_desc =
        inner_product_forward::primitive_desc(*fc_desc_p, mkldnn_engine);

    auto fc_dst_memory_pd = fc_prim_desc.dst_primitive_desc();
    auto fc_dst_memory_sz = fc_dst_memory_pd.get_size();
    T* output_data = output->mutable_data<T>(ctx.GetPlace(), fc_dst_memory_sz);

    auto fc_dst_memory = memory(fc_dst_memory_pd, to_void_cast<T>(output_data));

    auto fc = bias ? inner_product_forward(fc_prim_desc, fc_src_memory,
                                           fc_weights_memory, *fc_bias_memory_p,
                                           fc_dst_memory)
                   : inner_product_forward(fc_prim_desc, fc_src_memory,
                                           fc_weights_memory, fc_dst_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{fc};
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(fc_dst_memory));
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(fc, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNOpKernel<float>);
