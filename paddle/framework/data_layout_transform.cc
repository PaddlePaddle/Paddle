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

void TransDataLayout(const std::vector<int>& axis,
                     const platform::DeviceContext* ctx,
                     const KernelTypePair& kernel_pair, const Variable& in,
                     Variable* out) {
  PADDLE_ENFORCE(in.IsType<Tensor>(), "Only support Tensor transform!.");
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_pair.first.place_,
                                      kernel_pair.second.place_),
      "TransDataLayout only support DataLayout transform on same place!");
  PADDLE_ENFORCE(kernel_pair.first.data_type_ == kernel_pair.second.data_type_,
                 "TransDataLayout only support Datatype are same!");

  auto src = in.Get<Tensor>();
  auto* dst = out->GetMutable<Tensor>();
  PADDLE_ENFORCE(arity(src.dims()) == 4, "Input Arity Only Suppport 4!");

  auto src_dim = src.dims();
  std::vector<int64_t> dst_dim;

  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  dst->Resize(make_ddim(dst_dim));
  auto place = kernel_pair.second.place_;
  dst->mutable_data(place, src.type());

  auto src_type = kernel_pair.first.data_type_;
  framework::VisitDataType(src_type, CastDataLayout(ctx, axis, src, dst));

  dst->set_layout(kernel_pair.second.data_layout_);
}

}  // namespace framework
}  // namespace paddle
