/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/operators/interpolate_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using dnnl::memory;
using dnnl::primitive;
using dnnl::reorder;
using dnnl::stream;
using dnnl::resampling_forward;
using platform::GetMKLDNNFormat;
using platform::to_void_cast;

template <typename T = float>
class InterpolateMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::resampling_forward> {
 public:
  InterpolateMKLDNNHandler(const dnnl::algorithm algo,
                           const dnnl::engine engine, platform::Place cpu_place,
                           const Tensor* x, Tensor* out)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::resampling_forward>(
            engine, cpu_place) {
    const auto dst_tz = phi::vectorize(out->dims());
    const auto dst_md = memory::desc(dst_tz, platform::MKLDNNGetDataType<T>(),
                                     MKLDNNMemoryFormat::any);
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_inference,
                                            algo, x->mem_desc(), dst_md);
  }
};

template <typename T = float>
class InterpolateMKLDNNKernel : public framework::OpKernel<T> {
  std::vector<int> ComputeOutputShape(
      const framework::ExecutionContext& ctx) const {
    const auto* x = ctx.Input<Tensor>("X");
    const auto& in_dims = x->dims();

    const framework::DDim in_dhw_dims =
        phi::slice_ddim(in_dims, 2, in_dims.size());

    std::vector<int> out_dims;
    out_dims.reserve(5);
    if (in_dhw_dims.size() == 1) {
      out_dims.push_back(ctx.Attr<int>("out_w"));
    } else if (in_dhw_dims.size() == 2) {
      out_dims.push_back(ctx.Attr<int>("out_h"));
      out_dims.push_back(ctx.Attr<int>("out_w"));
    } else if (in_dhw_dims.size() == 3) {
      out_dims.push_back(ctx.Attr<int>("out_d"));
      out_dims.push_back(ctx.Attr<int>("out_h"));
      out_dims.push_back(ctx.Attr<int>("out_w"));
    }

    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    auto out_size = ctx.Input<Tensor>("OutSize");
    if (list_new_size_tensor.size() > 0) {
      auto new_size = get_new_shape(list_new_size_tensor);
      if (new_size.size() == out_dims.size()) {
        out_dims = new_size;
      }
    } else if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      if (out_size_data.size() == out_dims.size()) {
        out_dims = out_size_data;
      }
    } else {
      std::vector<float> scale;
      scale.reserve(3);
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        scale.resize(3, scale_data[0]);
        std::copy(scale_data.begin(), scale_data.end(), scale.begin());
      } else {
        std::string op_type = ctx.Type();

        if (op_type.find("v2") == std::string::npos) {  // v1
          scale.push_back(ctx.Attr<float>("scale"));
          scale.push_back(scale[0]);
          scale.push_back(scale[0]);
        } else {  // v2
          std::vector<float> scale_attr = ctx.Attr<std::vector<float>>("scale");
          if (scale_attr.size() > 0) {
            scale.resize(3, scale_attr[0]);
            std::copy(scale_attr.begin(), scale_attr.end(), scale.begin());
          }
        }
      }
      if (scale[0] > 0.0f && scale[1] > 0.0f && scale[2] > 0.0f) {
        int j = 0;
        std::vector<int64_t> in_dhw_vec = phi::vectorize(in_dhw_dims);
        std::transform(
            in_dhw_vec.begin(), in_dhw_vec.end(), out_dims.begin(),
            [&](int64_t i) -> int { return static_cast<int>(i * scale[j++]); });
      }
    }

    PADDLE_ENFORCE_GT(std::all_of(out_dims.begin(), out_dims.end(),
                                  [](int i) { return i > 0; }),
                      0, platform::errors::InvalidArgument(
                             "out_d, out_h, out_w of Op(interpolate) "
                             "should be greater than 0."));

    const std::vector<int64_t> nc_dims = {in_dims[0], in_dims[1]};
    out_dims.insert(out_dims.begin(), nc_dims.begin(), nc_dims.end());
    return out_dims;
  }

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    const auto interp_method = ctx.Attr<std::string>("interp_method");
    const dnnl::algorithm algo = (interp_method == "nearest")
                                     ? dnnl::algorithm::resampling_nearest
                                     : dnnl::algorithm::resampling_linear;

    const auto out_dims_vec = ComputeOutputShape(ctx);
    framework::DDim dim_out = phi::make_ddim(out_dims_vec);
    out->Resize(dim_out);

    InterpolateMKLDNNHandler<T> handler(algo, mkldnn_engine, ctx.GetPlace(), x,
                                        out);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);

    auto resampling_prim = handler.AcquireForwardPrimitive();
    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    resampling_prim->execute(astream, args);
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(nearest_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>,
                   ops::InterpolateMKLDNNKernel<int8_t>,
                   ops::InterpolateMKLDNNKernel<uint8_t>);
REGISTER_OP_KERNEL(bilinear_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);

REGISTER_OP_KERNEL(nearest_interp_v2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>,
                   ops::InterpolateMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::InterpolateMKLDNNKernel<int8_t>,
                   ops::InterpolateMKLDNNKernel<uint8_t>);
REGISTER_OP_KERNEL(bilinear_interp_v2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);
