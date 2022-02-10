/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include <algorithm>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

struct FillOpVisitor {
  FillOpVisitor(framework::LoDTensor *tensor, const std::vector<float> &value)
      : tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    platform::CPUPlace cpu;
    auto *data = tensor_->mutable_data<T>(cpu);
    std::transform(value_.data(), value_.data() + tensor_->numel(), data,
                   [](float dat) { return static_cast<T>(dat); });
  }

  framework::LoDTensor *tensor_;
  const std::vector<float> &value_;
};

template <typename T>
class FillKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto &out = GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Out"),
                                "Output", "Out", "Fill");
    out.Resize(framework::make_ddim(ctx.Attr<std::vector<int>>("shape")));
    auto dtype =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto pten_dtype = framework::TransToPtenDataType(dtype);
    platform::CPUPlace cpu;
    auto force_cpu = ctx.Attr<bool>("force_cpu");
    out.mutable_data(force_cpu ? cpu : ctx.GetPlace(), pten_dtype);

    framework::LoDTensor tensor;

    if (force_cpu || platform::is_cpu_place(ctx.GetPlace())) {
      tensor.ShareDataWith(out);
    } else {
      // Always make tensor in CPU memory.
      tensor.Resize(out.dims());
      tensor.mutable_data(cpu, pten_dtype);
    }

    framework::VisitDataType(
        dtype, FillOpVisitor(&tensor, ctx.Attr<std::vector<float>>("value")));

    if (!force_cpu && platform::is_gpu_place(ctx.GetPlace())) {
      // Copy tensor to out
      framework::TensorCopy(
          tensor, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &out);
    }
  }
};
}  // namespace operators
}  // namespace paddle
