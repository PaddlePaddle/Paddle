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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace paddle {
namespace operators {

template <cnnlSoftmaxAlgorithm_t softmax_algo, typename T>
class SoftmaxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    const int rank = in->dims().size();
    const int axis = phi::funcs::CanonicalAxis(ctx.Attr<int>("axis"), rank);

    // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
    const int cnnl_softmax_dims = 3;
    const int d1 = phi::funcs::SizeToAxis(axis, in->dims());
    const int d2 = in->dims()[axis];
    const int d3 = phi::funcs::SizeOutAxis(axis, in->dims());

    // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
    // possible.
    cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    std::vector<int> regard_in_shape{d1, 1, d2};
    if (d3 != 1) {
      mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
      regard_in_shape = {d1, d2, d3};
    }

    static const cnnlSoftmaxAlgorithm_t algo = softmax_algo;
    MLUCnnlTensorDesc in_desc(
        cnnl_softmax_dims, regard_in_shape.data(), ToCnnlDataType<T>());
    MLUCnnl::SoftmaxForward(ctx,
                            algo,
                            mode,
                            NULL,
                            in_desc.get(),
                            GetBasePtr(in),
                            NULL,
                            in_desc.get(),
                            GetBasePtr(out));
  }
};

template <cnnlSoftmaxAlgorithm_t softmax_algo, typename T>
class SoftmaxGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<framework::LoDTensor>("Out");
    auto* dOut = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto* dX = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(ctx.GetPlace());

    const int rank = out->dims().size();
    const int axis = phi::funcs::CanonicalAxis(ctx.Attr<int>("axis"), rank);

    // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
    const int cnnl_softmax_dims = 3;
    const int d1 = phi::funcs::SizeToAxis(axis, out->dims());
    const int d2 = out->dims()[axis];
    const int d3 = phi::funcs::SizeOutAxis(axis, out->dims());

    // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
    // possible.
    cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    std::vector<int> regard_out_shape{d1, 1, d2};
    if (d3 != 1) {
      mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
      regard_out_shape = {d1, d2, d3};
    }

    static const cnnlSoftmaxAlgorithm_t algo = softmax_algo;
    MLUCnnlTensorDesc out_desc(
        cnnl_softmax_dims, regard_out_shape.data(), ToCnnlDataType<T>());
    MLUCnnl::SoftmaxBackward(ctx,
                             algo,
                             mode,
                             out_desc.get(),
                             GetBasePtr(out),
                             out_desc.get(),
                             GetBasePtr(dOut),
                             out_desc.get(),
                             GetBasePtr(dX));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(
    softmax,
    ops::SoftmaxMLUKernel<CNNL_SOFTMAX_ACCURATE, float>,
    ops::SoftmaxMLUKernel<CNNL_SOFTMAX_ACCURATE, plat::float16>);
REGISTER_OP_MLU_KERNEL(softmax_grad,
                       ops::SoftmaxGradMLUKernel<CNNL_SOFTMAX_ACCURATE, float>,
                       ops::SoftmaxGradMLUKernel<CNNL_SOFTMAX_ACCURATE,
                                                 paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(log_softmax,
                       ops::SoftmaxMLUKernel<CNNL_SOFTMAX_LOG, float>,
                       ops::SoftmaxMLUKernel<CNNL_SOFTMAX_LOG, plat::float16>);
REGISTER_OP_MLU_KERNEL(
    log_softmax_grad,
    ops::SoftmaxGradMLUKernel<CNNL_SOFTMAX_LOG, float>,
    ops::SoftmaxGradMLUKernel<CNNL_SOFTMAX_LOG, paddle::platform::float16>);
