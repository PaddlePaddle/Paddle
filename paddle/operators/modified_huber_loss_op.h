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

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
struct CheckLabelValue {
  HOSTDEVICE T operator()(const T& val) const {
    PADDLE_ASSERT(val == static_cast<T>(0) || val == static_cast<T>(1));
  }
};

template <typename T>
struct ModifiedHuberLossForward {
  HOSTDEVICE T operator()(const T& val) const {
    if (val < -1) {
      return -4 * val;
    } else if (val < 1) {
      return (1 - val) * (1 - val);
    } else {
      return static_cast<T>(0);
    }
  }
};

template <typename Place, typename T>
class ModifiedHuberLossKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto* in1 = context.Input<Tensor>("Y");
    auto* out0 = context.Output<Tensor>("intermediate_val");
    auto* out1 = context.Output<Tensor>("Out");

    out0->mutable_data<T>(context.GetPlace());
    out1->mutable_data<T>(context.GetPlace());
    auto place = context.GetEigenDevice<Place>();

    auto x = EigenVector<T>::Flatten(*in0);
    auto y = EigenVector<T>::Flatten(*in1);
    // make sure value's of Y in {0, 1}
    y.unaryExpr(CheckLabelValue<T>());
    auto inter_val = EigenVector<T>::Flatten(*out0);
    // scale y to {-1, +1} and compute x * y
    inter_val.device(place) = x * (2 * y - static_cast<T>(1));
    auto loss = EigenVector<T>::Flatten(*out1);
    loss.device(place) = inter_val.unaryExpr(ModifiedHuberLossForward<T>());
  }
};

// Use thrust lib to unify cpu and gpu
// CPU backward kernel
template <typename T>
class ModifiedHuberLossGradCPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto* in1 = context.Input<Tensor>("Y");
    auto* in2 = context.Input<Tensor>("intermediate_val");
    auto* in3 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    auto* out1 = context.Output<Tensor>(framework::GradVarName("X"));

    // loop inter_val (x<-1) (x<1) otherwise
    const T* p_inter_val = in2->data<T>();
    const T* p_out_grad = in3->data<T>();
    size_t counts = static_cast<size_t>(framework::product(in2->dims()));

    if (out0) {
      T* p_x_grad = out0->mutable_data<T>(context.GetPlace());
      const T* p_y = in1->data<T>();
      ModifiedHuberLossBackward(p_inter_val, p_y, p_out_grad, p_x_grad, counts);
    }

    if (out1) {
      T* p_y_grad = out1->mutable_data<T>(context.GetPlace());
      const T* p_x = in0->data<T>();
      ModifiedHuberLossBackward(p_inter_val, p_x, p_out_grad, p_y_grad, counts);
    }
  }

 protected:
  void ModifiedHuberLossBackward(const T* p_inter_data, const T* p_in_data,
                                 const T* p_in_grad, T* p_out_grad,
                                 size_t counts) const {
    for (size_t i = 0; i < counts; ++i) {
      if (p_inter_data[i] < -1) {
        p_out_grad[i] = -4 * p_in_data[i] * p_in_grad[i];
      } else if (p_inter_data[i] < 1) {
        p_out_grad[i] =
            -2 * (1 - p_inter_data[i]) * p_in_data[i] * p_in_grad[i];
      } else {
        p_out_grad[i] = 0;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
