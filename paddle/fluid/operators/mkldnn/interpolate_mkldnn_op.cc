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
    : public platform::MKLDNNHandlerT<T, dnnl::resampling_forward> {
 public:
  InterpolateMKLDNNHandler(const dnnl::algorithm algo,
                           const paddle::platform::MKLDNNDeviceContext& dev_ctx,
                           const dnnl::engine engine, platform::Place cpu_place,
                           const Tensor* x, Tensor* z,
                           const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, dnnl::resampling_forward>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(
                framework::vectorize(x->dims()),
                uniq_name + (algo == dnnl::algorithm::resampling_nearest
                                 ? "N"
                                 : "L"))) {
    if (!this->isCached()) {
      const auto src_x_tz = framework::vectorize(x->dims());
      const auto dst_tz = framework::vectorize(z->dims());
      const auto src0_md = dnnl::memory::desc(
          src_x_tz, platform::MKLDNNGetDataType<T>(), x->format());
      const auto dst_md = memory::desc(dst_tz, platform::MKLDNNGetDataType<T>(),
                                       MKLDNNMemoryFormat::any);
      auto resampling_d = dnnl::resampling_forward::desc(
          dnnl::prop_kind::forward_inference, algo, src0_md, dst_md);  // scale

      this->fwd_pd_.reset(new dnnl::resampling_forward::primitive_desc(
          resampling_d, this->engine_));

      auto key_pd = this->key_ + "@fwd_pd";
      this->dev_ctx_.SetBlob(key_pd, this->fwd_pd_);
    }
  }

  std::shared_ptr<resampling_forward::primitive_desc>
  AcquireForwardPrimitiveDescriptor() {
    const std::string key_pd = this->key_ + "@fwd_pd";
    this->fwd_pd_ =
        std::static_pointer_cast<dnnl::resampling_forward::primitive_desc>(
            this->dev_ctx_.GetBlob(key_pd));
    return this->fwd_pd_;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    auto user_src_md = platform::MKLDNNMemDesc(
        framework::vectorize(input->dims()), platform::MKLDNNGetDataType<T>(),
        input->format());

    return this->AcquireMemoryWithReorder(
        user_src_md, this->fwd_pd_->src_desc(), to_void_cast<T>(input_data),
        "@src_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<dnnl::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr = output->mutable_data<T_out>(
        this->place_, this->fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr,
                                            "@dst_mem_p");
  }
};

// Nearest Neighbor
// Linear (or Bilinear for 2D spatial tensor, Trilinear for 3D spatial tensor).

template <typename T = float>
class InterpolateMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    std::vector<float> scale_prior;
    auto* z = ctx.Output<Tensor>("Out");

    auto interp_method = ctx.Attr<std::string>("interp_method");
    std::cout << "interp_method:" << interp_method << std::endl;
    dnnl::algorithm algo = (interp_method == "nearest")
                               ? dnnl::algorithm::resampling_nearest
                               : dnnl::algorithm::resampling_linear;
    // high priority
    if (ctx.HasInput("OutSize")) {
      auto* out_size = ctx.Input<Tensor>("OutSize");
      auto* out_size_data = out_size->data<int>();
      framework::DDim dim_out;
      const DataLayout data_layout =
          framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));
      auto dim_x = x->dims();
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {dim_x[0], dim_x[1], out_size_data[0], out_size_data[1]};
      } else {
        dim_out = {dim_x[0], out_size_data[0], out_size_data[1], dim_x[3]};
      }
      z->Resize(dim_out);
    }

    InterpolateMKLDNNHandler<> handler(algo, dev_ctx, mkldnn_engine,
                                       ctx.GetPlace(), x, z,
                                       ctx.OutputName("Out"));

    auto resampling_pd = handler.AcquireForwardPrimitiveDescriptor();
    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(x);
    auto dst_memory_p = handler.AcquireDstMemory(z);

    auto resampling_prim = handler.AcquireForwardPrimitive();
    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};
    mkldnn::stream astream(mkldnn_engine);
    resampling_prim->execute(astream, args);
    astream.wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(nearest_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);
REGISTER_OP_KERNEL(linear_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);
REGISTER_OP_KERNEL(bilinear_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);
REGISTER_OP_KERNEL(trilinear_interp, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::InterpolateMKLDNNKernel<float>);
