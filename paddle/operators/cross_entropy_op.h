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

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
    PADDLE_ASSERT(std::is_floating_point<T>::value);
    const T kApproInf = 1e20;

    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

template <typename T>
class CrossEntropyOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* y = ctx.Output<Tensor>("Y");
    T* y_data = y->mutable_data<T>(ctx.GetPlace());

    const int batch_size = x->dims()[0];
    if (ctx.Attr<bool>("softLabel")) {
      auto prob = EigenMatrix<T>::From(*x);
      auto lbl_mat = EigenMatrix<T>::From(*labels);
      auto loss = EigenMatrix<T>::From(*y);

      loss.device(ctx.GetEigenDevice<platform::CPUPlace>()) =
          -((lbl_mat * prob.log().unaryExpr(TolerableValue<T>()))
                .sum(Eigen::DSizes<int, 1>(1))
                .reshape(Eigen::DSizes<int, 2>(batch_size, 1)));
    } else {
      const int class_num = x->dims()[1];
      const T* x_data = x->data<T>();

      const int* label_data = labels->data<int>();
      for (int i = 0; i < batch_size; ++i) {
        int index = i * class_num + label_data[i];
        y_data[i] = -TolerableValue<T>()(std::log(x_data[index]));
      }
    }
  }
};

template <typename T>
class CrossEntropyGradientOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const Tensor* label = ctx.Input<Tensor>("Label");
    Tensor* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int class_num = x->dims()[1];
    if (ctx.Attr<bool>("softLabel")) {
      auto x_mat = EigenMatrix<T>::From(*x);
      auto dy_mat = EigenMatrix<T>::From(*dy);
      auto lbl_mat = EigenMatrix<T>::From(*label);
      auto dx_mat = EigenMatrix<T>::From(*dx);

      dx_mat.device(ctx.GetEigenDevice<platform::CPUPlace>()) =
          -(lbl_mat * dy_mat.broadcast(Eigen::DSizes<int, 2>(1, class_num)) /
            x_mat);
    } else {
      int batch_size = x->dims()[0];
      const T* dy_data = dy->data<T>();
      const T* x_data = x->data<T>();
      const int* label_data = label->data<int>();

      // TODO(qingqing): make zero setting a common function.
      memset(dx_data, 0, sizeof(T) * batch_size * class_num);

      for (int i = 0; i < batch_size; ++i) {
        PADDLE_ASSERT(label_data[i] >= 0 || label_data[i] < class_num);
        int index = i * class_num + label_data[i];
        dx_data[index] = -dy_data[i] / x_data[index];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
