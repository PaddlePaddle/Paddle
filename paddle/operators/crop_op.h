/* Copyright (c) 2016 CropdleCropdle Authors. All Rights Reserve.

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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {  // Internal

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::Tensor;

template <typename T>
class CropKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto x_stride = framework::stride(x->dims());
    auto out_stride = framework::stride(out->dims());
    auto offsets = context.Attr<std::vector<int>>("offsets");
    PADDLE_ENFORCE_EQ(
        x->dims().size(), static_cast<int64_t>(offsets.size()),
        "Offsets size should be equal to dimension size of input tensor.");
    int64_t offset = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
      offset += (x_stride[i] * offsets[i]);
    }
    StridedMemcpy<T>(context.device_context(), x_data + offset, x_stride,
                     out->dims(), out_stride, out_data);
  }
};

template <typename Place, typename T, size_t D>
void CropGradFunction(const framework::ExecutionContext& context) {
  auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
  if (d_x != nullptr) {
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    d_x->mutable_data<T>(context.GetPlace());
    auto offsets = context.Attr<std::vector<int>>("offsets");
    Eigen::array<std::pair<int, int>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = d_x->dims()[i] - d_out->dims()[i] - offsets[i];
    }
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    d_x_tensor.device(context.GetEigenDevice<Place>()) =
        d_out_tensor.pad(paddings, 0);
  }
}

template <typename Place, typename T>
class CropGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
    switch (rank) {
      case 1:
        CropGradFunction<Place, T, 1>(context);
        break;
      case 2:
        CropGradFunction<Place, T, 2>(context);
        break;
      case 3:
        CropGradFunction<Place, T, 3>(context);
        break;
      case 4:
        CropGradFunction<Place, T, 4>(context);
        break;
      case 5:
        CropGradFunction<Place, T, 5>(context);
        break;
      case 6:
        CropGradFunction<Place, T, 6>(context);
        break;
      default:
        PADDLE_THROW(
            "CropOp only support tensors with no more than 6 dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle
