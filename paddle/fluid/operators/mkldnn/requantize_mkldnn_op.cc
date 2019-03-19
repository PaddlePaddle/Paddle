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

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

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

    std::vector<primitive> pipeline;
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    mkldnn::memory::data_type src_dt =
        paddle::framework::ToMKLDNNDataType(input->type());
    mkldnn::memory::data_type dst_dt = src_dt;  // TODO(Xiaoli) support
                                                // requantize from different
                                                // data type (e.g., s8 to u8)
    mkldnn::memory::format dst_fmt = memory::format::nhwc;

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    float scale_shift = scale_out / scale_in;

    mkldnn::primitive_attr attri;
    int mask = 0;
    attri.set_output_scales(mask, {scale_shift});

    auto src_pd = input->get_mkldnn_prim_desc();
    auto src_memory =
        std::make_shared<mkldnn::memory>(src_pd, to_void_cast<T>(input_data));
    std::shared_ptr<primitive::at> src_memory_p =
        std::shared_ptr<primitive::at>(new primitive::at(*src_memory));

    auto dst_md = platform::MKLDNNMemDesc({dst_tz}, dst_dt, dst_fmt);
    auto dst_pd = mkldnn::memory::primitive_desc(dst_md, engine);
    auto dst_memory = mkldnn::memory(dst_pd, to_void_cast<T>(output_data));

    auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(src_pd, dst_pd, attri));

    auto reorder_p = std::shared_ptr<reorder>(
        new reorder(*reorder_pd, *src_memory_p, dst_memory));
    pipeline.push_back(*reorder_p);
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_mkldnn_prim_desc(dst_pd);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(requantize, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ReQuantOpKernel<int8_t>, ops::ReQuantOpKernel<uint8_t>);
