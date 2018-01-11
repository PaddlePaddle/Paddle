/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/data_layout_transform.h"

#include "paddle/framework/tensor.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace framework {

struct CastDataLayout {
  CastDataLayout(const platform::DeviceContext* ctx,
                 const std::vector<int>& axis, const framework::Tensor& in,
                 framework::Tensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void operator()() {
    auto place = ctx_->GetPlace();

    if (platform::is_cpu_place(place)) {
      operators::math::Transpose<platform::CPUDeviceContext, T, 4> trans4;
      auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
      trans4(*context, in_, out_, axis_);
    } else {
      PADDLE_THROW("Unsupport CPU <-> GPU!");
    }
  }
};

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type,
                     const std::vector<int>& axis, const Tensor& in,
                     const platform::Place& dst_place, Tensor* out) {
  PADDLE_ENFORCE(
      platform::places_are_same_class(in.place(), dst_place),
      "TransDataLayout only support DataLayout transform on same place!");

  PADDLE_ENFORCE(arity(in.dims()) == 4, "Input Arity only support 4!");

  auto src_dim = in.dims();
  std::vector<int64_t> dst_dim;

  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->Resize(make_ddim(dst_dim));
  out->mutable_data(in.place(), in.type());

  framework::VisitDataType(framework::ToDataType(in.type()),
                           CastDataLayout(ctx, axis, in, out));

  out->set_layout(expected_kernel_type.data_layout_);
}

}  // namespace framework
}  // namespace paddle
