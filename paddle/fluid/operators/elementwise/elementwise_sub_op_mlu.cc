/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ElementwiseSubMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUOpTensorKernel<T>(ctx, CNNL_OP_TENSOR_SUB);
  }
};

template <typename T>
class ElementwiseSubGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis);

    MLUCnnlTensorDesc dout_desc(*dout);

    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      if (dx->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        GetReduceAxesAndDstDims(axis, dout->dims(), dx->dims(), &reduce_axes,
                                &dst_dims_vec);

        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, ToCnnlDataType<T>(),
            CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dx_desc(dst_dims_vec.size(), dst_dims_vec.data(),
                                  ToCnnlDataType<T>());
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(dout), 0, nullptr,
                        nullptr, dx_desc.get(), GetBasePtr(dx));
      } else {
        framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dx);
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor* tmp_dout = const_cast<Tensor*>(dout);
      if (dy->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        GetReduceAxesAndDstDims(axis, dout->dims(), dy->dims(), &reduce_axes,
                                &dst_dims_vec);

        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, ToCnnlDataType<T>(),
            CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dy_desc(dst_dims_vec.size(), dst_dims_vec.data(),
                                  ToCnnlDataType<T>());
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(dout), 0, nullptr,
                        nullptr, dy_desc.get(), GetBasePtr(dy));
        tmp_dout = dy;
      }

      // call neg op, dy = -dout
      MLUCnnlTensorDesc tmp_dout_desc(*tmp_dout);
      MLUCnnlTensorDesc dy_desc(*dy);

      MLUUnary<NEG>(ctx, CNNL_COMPUTATION_HIGH_PRECISION, tmp_dout_desc.get(),
                    GetBasePtr(tmp_dout), dy_desc.get(), GetBasePtr(dy));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(elementwise_sub, ops::ElementwiseSubMLUKernel<int>,
                       ops::ElementwiseSubMLUKernel<float>,
                       ops::ElementwiseSubMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(elementwise_sub_grad,
                       ops::ElementwiseSubGradMLUKernel<int>,
                       ops::ElementwiseSubGradMLUKernel<float>,
                       ops::ElementwiseSubGradMLUKernel<plat::float16>);
