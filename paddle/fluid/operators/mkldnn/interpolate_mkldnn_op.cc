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

// template <typename T>
// static void Interpolate2D(const framework::ExecutionContext& ctx,
//                           const Tensor* input, Tensor* output) {
//   const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
//   const DataLayout data_layout =
//   framework::StringToDataLayout(data_layout_str);
//   int n, c, in_d, in_h, in_w;
//   ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

//   auto interp_method = ctx.Attr<std::string>("interp_method");

//   int out_h = ctx.Attr<int>("out_h");
//   int out_w = ctx.Attr<int>("out_w");

//   auto list_new_size_tensor =
//   ctx.MultiInput<framework::Tensor>("SizeTensor");
//   if (list_new_size_tensor.size() > 0) {
//     // have size tensor
//     auto new_size = get_new_shape(list_new_size_tensor);
//     out_h = new_size[0];
//     out_w = new_size[1];
//   } else {
//     float scale;
//     auto scale_tensor = ctx.Input<Tensor>("Scale");
//     if (scale_tensor != nullptr) {
//       auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
//       scale = scale_data[0];
//     } else {
//       scale = ctx.Attr<float>("scale");
//     }
//     if (scale > 0) {
//       out_h = static_cast<int>(in_h * scale);
//       out_w = static_cast<int>(in_w * scale);
//     }
//     auto out_size = ctx.Input<Tensor>("OutSize");
//     if (out_size != nullptr) {
//       auto out_size_data = get_new_data_from_tensor<int>(out_size);
//       out_h = out_size_data[0];
//       out_w = out_size_data[1];
//     }
//   }
//   PADDLE_ENFORCE_GT(out_h, 0, platform::errors::InvalidArgument(
//                                   "out_h in Attr(out_shape) of
//                                   Op(interpolate) "
//                                   "should be greater than 0."));
//   PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
//                                   "out_w in Attr(out_shape) of
//                                   Op(interpolate) "
//                                   "should be greater than 0."));
//   framework::DDim dim_out;
//   if (data_layout == DataLayout::kNCHW) {
//     dim_out = {n, c, out_h, out_w};
//   } else {
//     dim_out = {n, out_h, out_w, c};
//   }
//   output->mutable_data<T>(dim_out, ctx.GetPlace());
// }

// template <typename T>
// static void Interpolate1D(const framework::ExecutionContext& ctx,
//                           const Tensor* input, Tensor* output) {
//   const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
//   const DataLayout data_layout =
//   framework::StringToDataLayout(data_layout_str);
//   int n, c, in_d, in_h, in_w;
//   ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);
//   int out_w = ctx.Attr<int>("out_w");
//   auto list_new_size_tensor =
//   ctx.MultiInput<framework::Tensor>("SizeTensor");
//   if (list_new_size_tensor.size() > 0) {
//     // have size tensor
//     auto new_size = get_new_shape(list_new_size_tensor);
//     out_w = new_size[0];
//   } else {
//     float scale;
//     auto scale_tensor = ctx.Input<Tensor>("Scale");
//     if (scale_tensor != nullptr) {
//       auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
//       scale = scale_data[0];
//     } else {
//       scale = ctx.Attr<float>("scale");
//     }
//     if (scale > 0) {
//       out_w = static_cast<int>(in_w * scale);
//     }
//     auto out_size = ctx.Input<Tensor>("OutSize");
//     if (out_size != nullptr) {
//       auto out_size_data = get_new_data_from_tensor<int>(out_size);
//       out_w = out_size_data[0];
//     }
//   }
//   PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
//                                   "out_w in Attr(out_shape) of
//                                   Op(interpolate) "
//                                   "should be greater than 0."));
//   framework::DDim dim_out;
//   if (data_layout == DataLayout::kNCHW) {
//     dim_out = {n, c, out_w};
//   } else {
//     dim_out = {n, out_w, c};
//   }
//   output->mutable_data<T>(dim_out, ctx.GetPlace());
// }

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
  std::vector<int> ComputeOutputShape(
      const framework::ExecutionContext& ctx) const {
    const auto* x = ctx.Input<Tensor>("X");
    auto in_dims = x->dims();
    // const DataLayout data_layout = framework::StringToDataLayout(
    //   ctx.Attr<std::string>("data_layout"));

    const bool is_channel_last =
        false;  // used in mkldnn kernel, we must use NCHW

    framework::DDim in_dhw_dims;
    if (is_channel_last) {  // NDHWC, NHWC, NWC
      in_dhw_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {  // NCDHW, NCHW, NCW
      in_dhw_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    }
    std::vector<int> out_dims;
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
    PADDLE_ENFORCE_GT(
        std::all_of(out_dims.begin(), out_dims.end(),
                    [](int i) { return i > 0; }),
        0, platform::errors::InvalidArgument(
               "out_d, out_h, out_w in Attr(out_shape) of Op(interpolate) "
               "should be greater than 0."));

    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      auto new_size = get_new_shape(list_new_size_tensor);
      if (new_size.size() == out_dims.size()) {
        out_dims = new_size;
      }
    } else {
      float scale;
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        scale = scale_data[0];
      } else {
        scale = ctx.Attr<float>("scale");
      }
      if (scale > 0) {
        std::transform(out_dims.begin(), out_dims.end(), out_dims.begin(),
                       [&](int i) -> int { return i * scale; });
      }
      auto out_size = ctx.Input<Tensor>("OutSize");
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        if (out_size_data.size() == out_dims.size()) {
          out_dims = out_size_data;
        }
      }
    }
    out_dims.insert(out_dims.begin(), in_dims[0]);
    if (is_channel_last) {
      out_dims.push_back(in_dims[in_dims.size() - 1]);
    } else {
      out_dims.insert(out_dims.begin() + 1, in_dims[in_dims.size() - 1]);
    }
    // for (auto i =0; i < out_dims.size();i++){
    //   std::cout<<i<<",";
    // }
    // std::cout<<std::endl;
    return out_dims;
  }

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

    // auto dim_x = x->dims();
    // if (dim_x.size() == 3) {
    //   Interpolate1D<T>(ctx, x, z);
    // } else if (dim_x.size() == 4) {
    //   Interpolate2D<T>(ctx, x, z);
    // }
    auto out_dims_vec = ComputeOutputShape(ctx);
    auto dim_out = framework::make_ddim(out_dims_vec);
    z->mutable_data<T>(dim_out, ctx.GetPlace());

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
