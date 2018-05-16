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

#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when saving info into device context
    const std::string key = ctx.op().Output("Out");
    const std::string key_pool_pd = key + "@pool_pd";
    const std::string key_pool_workspace_memory =
        key + "@pool_workspace_memory";

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    // Only 2D pooling is supported now
    PADDLE_ENFORCE(ksize.size() == 2, "ksize must be 2D, i.e. 2D pooling");
    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "avg",
                   "pooling_type must be 'max' or 'avg'");
    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input dim must be with 4, i.e. NCHW");

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    // TODO(pzelazko-intel): support more formats
    auto src_md = platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                          mkldnn::memory::format::nchw);
    auto dst_md = platform::MKLDNNMemDesc(dst_tz, mkldnn::memory::f32,
                                          mkldnn::memory::format::nchw);

    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd =
        CreatePrimitiveDesc(src_md, dst_md, strides, paddings, ksize,
                            pooling_type, mkldnn_engine);

    // save pool_pd into global device context to be referred in backward path
    dev_ctx.SetBlob(key_pool_pd, pool_pd);

    std::shared_ptr<mkldnn::memory> workspace_memory =
        CreateWorkspaceMemory(pool_pd, pooling_type, mkldnn_engine);

    // save pool_workspace_memory to be referred in backward path
    dev_ctx.SetBlob(key_pool_workspace_memory, workspace_memory);

    auto src_memory =
        mkldnn::memory({src_md, mkldnn_engine},
                       static_cast<void*>(const_cast<T*>(input_data)));
    auto dst_memory =
        mkldnn::memory({dst_md, mkldnn_engine},
                       static_cast<void*>(const_cast<T*>(output_data)));

    auto pool_prim = mkldnn::pooling_forward(*pool_pd, src_memory, dst_memory,
                                             *workspace_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{pool_prim};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }

 private:
  std::unique_ptr<mkldnn::pooling_forward::primitive_desc> CreatePrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& dst,
      const std::vector<int>& stride, const std::vector<int>& padding,
      const std::vector<int>& kernel, const std::string& pooling_type,
      const mkldnn::engine& engine) const {
    auto pool_desc = mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward,
        pooling_type == "max" ? mkldnn::algorithm::pooling_max
                              : mkldnn::algorithm::pooling_avg,
        src, dst, stride, kernel, padding, padding, mkldnn::padding_kind::zero);

    auto p_pool_pd =
        new mkldnn::pooling_forward::primitive_desc(pool_desc, engine);
    return std::unique_ptr<mkldnn::pooling_forward::primitive_desc>(p_pool_pd);
  }

  std::unique_ptr<mkldnn::memory> CreateWorkspaceMemory(
      std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd,
      const std::string& pooling_type, const mkldnn::engine& engine) const {
    mkldnn::memory::primitive_desc workspace_md =
        pooling_type == "max"
            ? pool_pd->workspace_primitive_desc()
            : mkldnn::memory::primitive_desc(
                  {{}, mkldnn::memory::f32, mkldnn::memory::format::nchw},
                  engine);

    auto p_workspace_memory = new mkldnn::memory(workspace_md);
    return std::unique_ptr<mkldnn::memory>(p_workspace_memory);
  }
};

template <typename T>
class PoolMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    const Tensor* in_x = ctx.Input<Tensor>("X");
    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when referring info from device context
    const std::string key = ctx.op().Input("Out");
    const std::string key_pool_pd = key + "@pool_pd";
    const std::string key_pool_workspace_memory =
        key + "@pool_workspace_memory";

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const mkldnn::engine& mkldnn_engine = dev_ctx.GetEngine();

    const T* out_grad_data = out_grad->data<T>();
    T* in_x_grad_data = in_x_grad->mutable_data<T>(ctx.GetPlace());

    std::vector<int> diff_src_tz =
        paddle::framework::vectorize2int(in_x_grad->dims());
    std::vector<int> diff_dst_tz =
        paddle::framework::vectorize2int(out_grad->dims());

    auto diff_src_md = platform::MKLDNNMemDesc(diff_src_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nchw);
    auto diff_dst_md = platform::MKLDNNMemDesc(diff_dst_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nchw);

    // Retrieve pool_pd/pool_workspace_memory from device context
    auto pool_pd =
        std::static_pointer_cast<mkldnn::pooling_forward::primitive_desc>(
            dev_ctx.GetBlob(key_pool_pd));
    PADDLE_ENFORCE(pool_pd != nullptr,
                   "Fail to find pool_pd in device context");

    auto workspace_memory = std::static_pointer_cast<mkldnn::memory>(
        dev_ctx.GetBlob(key_pool_workspace_memory));
    PADDLE_ENFORCE(workspace_memory != nullptr,
                   "Fail to find workspace_memory in device context");

    auto pool_bwd_desc = mkldnn::pooling_backward::desc(
        pooling_type == "max" ? mkldnn::algorithm::pooling_max
                              : mkldnn::algorithm::pooling_avg,
        diff_src_md, diff_dst_md, strides, ksize, paddings, paddings,
        mkldnn::padding_kind::zero);
    auto pool_bwd_pd = mkldnn::pooling_backward::primitive_desc(
        pool_bwd_desc, mkldnn_engine, *pool_pd);

    auto diff_src_memory =
        mkldnn::memory({diff_src_md, mkldnn_engine},
                       static_cast<void*>(const_cast<T*>(in_x_grad_data)));
    auto diff_dst_memory =
        mkldnn::memory({diff_dst_md, mkldnn_engine},
                       static_cast<void*>(const_cast<T*>(out_grad_data)));

    auto bwd_prim = mkldnn::pooling_backward(
        pool_bwd_pd, diff_dst_memory, *workspace_memory, diff_src_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{bwd_prim};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(pool2d, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::PoolMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::PoolMKLDNNGradOpKernel<float>);
