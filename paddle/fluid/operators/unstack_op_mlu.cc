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

namespace paddle {
namespace operators {

template <typename T>
class UnStackMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.MultiOutput<phi::DenseTensor>("Y");
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += x->dims().size();
    int num = x->dims()[axis];

    std::vector<MLUCnnlTensorDesc> out_descs;
    std::vector<cnnlTensorDescriptor_t> out_raw_descs;
    std::vector<void *> out_ptrs;
    std::vector<int64_t> new_dims = phi::vectorize(x->dims());
    new_dims[axis] = 1;
    for (int i = 0; i < num; i++) {
      out[i]->mutable_data<T>(ctx.GetPlace());
      out_descs.emplace_back(MLUCnnlTensorDesc(
          new_dims.size(), new_dims.data(), ToCnnlDataType<T>()));
      out_raw_descs.push_back(out_descs.back().get());
      out_ptrs.push_back(GetBasePtr(out[i]));
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnl::Split(ctx,
                   num,
                   axis,
                   x_desc.get(),
                   GetBasePtr(x),
                   out_raw_descs.data(),
                   out_ptrs.data());
  }
};

template <typename T>
class UnStackGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.MultiInput<phi::DenseTensor>(framework::GradVarName("Y"));
    auto *y = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);
    int num = static_cast<int>(x.size());

    std::vector<MLUCnnlTensorDesc> x_descs;
    std::vector<cnnlTensorDescriptor_t> x_raw_descs;
    std::vector<const void *> x_ptrs;
    for (int i = 0; i < num; i++) {
      if (x[i]->dims().size() != 0) {
        std::vector<int64_t> in_dims = phi::vectorize(x[i]->dims());
        in_dims.insert(in_dims.begin() + axis, 1);
        x_descs.emplace_back(MLUCnnlTensorDesc(
            in_dims.size(), in_dims.data(), ToCnnlDataType<T>()));
      } else {
        int input_dims = 1;
        x_descs.emplace_back(
            MLUCnnlTensorDesc(1, &input_dims, ToCnnlDataType<T>()));
      }
      x_raw_descs.push_back(x_descs.back().get());
      x_ptrs.push_back(GetBasePtr(x[i]));
    }
    y->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnl::Concat(ctx,
                    num,
                    axis,
                    x_raw_descs.data(),
                    x_ptrs.data(),
                    y_desc.get(),
                    GetBasePtr(y));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(unstack,
                       ops::UnStackMLUKernel<float>,
                       ops::UnStackMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(unstack_grad,
                       ops::UnStackGradMLUKernel<float>,
                       ops::UnStackGradMLUKernel<plat::float16>);
