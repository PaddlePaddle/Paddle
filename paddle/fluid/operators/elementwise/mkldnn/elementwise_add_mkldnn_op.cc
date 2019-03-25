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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::reorder;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::sum;

template <typename T>
class EltwiseAddMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* z_data = z->mutable_data<T>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");

    auto x_dims = x->dims();
    auto y_dims_untrimed = y->dims();
    auto z_dims = z->dims();

    // Execute default elementwise_add operator when
    // broadcast operations need to performed.
    if (x_dims != y_dims_untrimed) {
      auto sum_func = [](T a, T b) -> T { return a + b; };

      TransformFunctor<decltype(sum_func), T,
                       paddle::platform::CPUDeviceContext, T>
          functor(
              x, y, z,
              ctx.template device_context<paddle::platform::CPUDeviceContext>(),
              sum_func);

      axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
      PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                     "Axis should be in range [0, x_dims)");

      auto y_dims = trim_trailing_singular_dims(y_dims_untrimed);
      axis = (y_dims.size() == 0) ? x_dims.size() : axis;

      int pre, n, post;
      get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

      if (post == 1) {
        functor.RunRowWise(n, pre);
      } else {
        functor.RunMidWise(n, pre, post);
      }
      z->set_mkldnn_prim_desc(x->get_mkldnn_prim_desc());
    } else {
      PADDLE_ENFORCE(x->layout() == DataLayout::kMKLDNN &&
                         x->format() != memory::format::format_undef,
                     "Wrong layout/format set for X tensor");
      PADDLE_ENFORCE(y->layout() == DataLayout::kMKLDNN &&
                         y->format() != memory::format::format_undef,
                     "Wrong layout/format set for Y tensor");

      std::vector<int> src_x_tz = framework::vectorize2int(x_dims);
      std::vector<int> src_y_tz = framework::vectorize2int(y_dims_untrimed);
      std::vector<int> dst_tz = framework::vectorize2int(z_dims);

      std::vector<memory::primitive_desc> srcs_pd;
      std::vector<memory> srcs;
      std::vector<float> scales = {1.0f, 1.0f};

      auto src_x_pd = memory::primitive_desc(
          {{src_x_tz}, memory::data_type::f32, x->format()}, mkldnn_engine);
      auto src_y_pd = memory::primitive_desc(
          {{src_y_tz}, memory::data_type::f32, y->format()}, mkldnn_engine);
      auto src_x_memory =
          memory(src_x_pd, paddle::platform::to_void_cast(x_data));
      auto src_y_memory =
          memory(src_y_pd, paddle::platform::to_void_cast(y_data));

      srcs_pd.push_back(src_x_pd);
      srcs_pd.push_back(src_y_pd);
      srcs.push_back(src_x_memory);
      srcs.push_back(src_y_memory);

      auto dst_md =
          memory::desc({dst_tz}, memory::data_type::f32, memory::format::any);

      // create primitive descriptor for sum
      auto sum_pd = sum::primitive_desc(dst_md, scales, srcs_pd);

      // create mkldnn memory for dst
      auto dst_mem_pd = sum_pd.dst_primitive_desc();
      memory dst_memory = memory(dst_mem_pd, z_data);

      std::vector<primitive::at> inputs;
      inputs.push_back(srcs[0]);
      inputs.push_back(srcs[1]);

      // create sum primitive
      auto sum_prim = sum(sum_pd, inputs, dst_memory);

      std::vector<primitive> pipeline;
      pipeline.push_back(sum_prim);
      stream(stream::kind::eager).submit(pipeline).wait();

      z->set_mkldnn_prim_desc(dst_mem_pd);
    }
  }
};

template <typename T>
class EltwiseAddMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    // skip out, x, y,
    // dout length is larger or equal than dx, dy.
    auto* out = dout;
    auto *x = dout, *y = dout;

    if (dx != nullptr && dy != nullptr && dx->dims() == dy->dims()) {
      if (dx->dims() == dy->dims()) {
        auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, T>(ctx);
        if (dx) {
          blas.VCOPY(dout->numel(), dout->data<T>(),
                     dx->mutable_data<T>(ctx.GetPlace()));
          dx->set_mkldnn_prim_desc(dout->get_mkldnn_prim_desc());
        }

        if (dy) {
          blas.VCOPY(dout->numel(), dout->data<T>(),
                     dy->mutable_data<T>(ctx.GetPlace()));
          dy->set_mkldnn_prim_desc(dout->get_mkldnn_prim_desc());
        }
      }
    } else {
      // Execute default kernel when broadcast is needed
      ElemwiseExplicitGradCompute<paddle::platform::CPUDeviceContext, T,
                                  IdentityGrad<T>, IdentityGrad<T>>(
          ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
          IdentityGrad<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNKernel<float>)

REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNGradKernel<float>)
