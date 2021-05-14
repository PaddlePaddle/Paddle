/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include "paddle/fluid/operators/collective/c_split_op.h"
#include "paddle/fluid/operators/math/concat_and_split.h"

namespace paddle {
namespace operators {

template <typename T>
class CSplitOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_GE(rank, 0, platform::errors::PreconditionNotMet(
                                   "The value of rank (%d) for c_split must be "
                                   "greater than or equal to 0.",
                                   rank));
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank, nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank, nranks));

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    std::vector<const framework::Tensor*> shape_refer;
    std::vector<framework::Tensor*> results;
    size_t numel = x->numel();
    auto dims = x->dims();
    numel /= nranks;
    int axis = dims.size() - 1;
    dims[dims.size() - 1] /= nranks;
    for (int i = 0; i < nranks; i++) {
      framework::Tensor* out = new framework::Tensor();
      out->mutable_data<T>(dims, place);
      shape_refer.emplace_back(out);
      results.emplace_back(out);
    }

    math::SplitFunctor<platform::CUDADeviceContext, T> functor;
    functor(dev_ctx, *x, shape_refer, axis, &results);
    out->mutable_data<T>(dims, place);
    paddle::framework::TensorCopySync(*results[rank], out->place(), out);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_split, ops::CSplitOpCUDAKernel<float>,
                        ops::CSplitOpCUDAKernel<double>,
                        ops::CSplitOpCUDAKernel<int>,
                        ops::CSplitOpCUDAKernel<int64_t>,
                        ops::CSplitOpCUDAKernel<plat::float16>);
