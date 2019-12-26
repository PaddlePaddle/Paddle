/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class FTRLOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    const auto* grad_var = ctx.InputVar("Grad");

    auto l1 = static_cast<T>(ctx.Attr<float>("l1")) + static_cast<T>(1e-10);
    auto l2 = static_cast<T>(ctx.Attr<float>("l2")) + static_cast<T>(1e-10);
    auto lr_power = static_cast<T>(ctx.Attr<float>("lr_power"));

    if (param_var->IsType<framework::LoDTensor>()) {
      auto* param_out = ctx.Output<Tensor>("ParamOut");
      auto* sq_accum_out = ctx.Output<Tensor>("SquaredAccumOut");
      auto* lin_accum_out = ctx.Output<Tensor>("LinearAccumOut");

      if (grad_var->IsType<framework::LoDTensor>()) {
        auto grad = ctx.Input<Tensor>("Grad");

        auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
        auto sq_accum =
            EigenVector<T>::Flatten(*ctx.Input<Tensor>("SquaredAccumulator"));
        auto lin_accum =
            EigenVector<T>::Flatten(*ctx.Input<Tensor>("LinearAccumulator"));
        auto g = EigenVector<T>::Flatten(*grad);
        auto lr = EigenVector<T>::Flatten(*ctx.Input<Tensor>("LearningRate"));

        auto p_out = EigenVector<T>::Flatten(*param_out);
        auto s_acc_out = EigenVector<T>::Flatten(*sq_accum_out);
        auto l_acc_out = EigenVector<T>::Flatten(*lin_accum_out);
        auto& place =
            *ctx.template device_context<DeviceContext>().eigen_device();

        Eigen::DSizes<int, 1> grad_dsize(grad->numel());

        auto new_accum = sq_accum + g * g;
        // Special case for lr_power = -0.5
        if (lr_power == static_cast<T>(-0.5)) {
          l_acc_out.device(place) =
              lin_accum + g -
              ((new_accum.sqrt() - sq_accum.sqrt()) /
              lr.broadcast(grad_dsize)) * p;
        } else {
          l_acc_out.device(place) =
              lin_accum + g -
              (new_accum.pow(-lr_power) - sq_accum.pow(-lr_power)) /
              lr.broadcast(grad_dsize) * p;
        }

        auto x = (l_acc_out.constant(l1) * l_acc_out.sign() - l_acc_out);
        if (lr_power == static_cast<T>(-0.5)) {
          auto y = (new_accum.sqrt() / lr.broadcast(grad_dsize)) +
                   l_acc_out.constant(static_cast<T>(2) * l2);
          auto pre_shrink = x / y;
          p_out.device(place) =
              (l_acc_out.abs() > l_acc_out.constant(l1))
                  .select(pre_shrink, p.constant(static_cast<T>(0)));
        } else {
          auto y = (new_accum.pow(-lr_power) / lr.broadcast(grad_dsize)) +
                   l_acc_out.constant(static_cast<T>(2) * l2);
          auto pre_shrink = x / y;
          p_out.device(place) =
              (l_acc_out.abs() > l_acc_out.constant(l1))
                  .select(pre_shrink, p.constant(static_cast<T>(0)));
        }

        s_acc_out.device(place) = sq_accum + g * g;
      } else if (grad_var->IsType<framework::SelectedRows>()) {
        auto param_data = param_out->mutable_data<T>(ctx.GetPlace());
        auto sq_accum_data = sq_accum_out->mutable_data<T>(ctx.GetPlace());
        auto lin_accum_data = lin_accum_out->mutable_data<T>(ctx.GetPlace());

        auto lr = *ctx.Input<framework::Tensor>("LearningRate")->data<T>();

        auto grad = ctx.Input<framework::SelectedRows>("Grad");
        auto grad_row_width = grad->value().dims()[1];

        math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
        auto grad_merge =
            merge_func(ctx.template device_context<DeviceContext>(), *grad);
        auto* grad_merge_data =
            grad_merge.mutable_value()->template data<T>();

        for (size_t i = 0; i < grad_merge.rows().size(); i++) {
          auto tensor_row_idx = grad_merge.rows()[i];

          for (int64_t j = 0; j < grad_row_width; j++) {
            auto grad_ele_idx = i * grad_row_width + j;
            auto tensor_ele_idx = tensor_row_idx * grad_row_width + j;

            auto grad_ele = grad_merge_data[grad_ele_idx];
            auto sq_accum_ele = sq_accum_data[tensor_ele_idx];

            auto new_accum = sq_accum_ele + grad_ele * grad_ele;

            if (lr_power == static_cast<T>(-0.5)) {
              lin_accum_data[tensor_ele_idx] +=
                  grad_ele -
                  (std::sqrt(new_accum) - std::sqrt(sq_accum_ele)) /
                  lr * param_data[tensor_ele_idx];
            } else {
              lin_accum_data[tensor_ele_idx] +=
                  grad_ele -
                  (std::pow(new_accum, -lr_power) -
                   std::pow(sq_accum_ele, -lr_power)) /
                  lr * param_data[tensor_ele_idx];
            }

            auto lin_accum_ele = lin_accum_data[tensor_ele_idx];

            if (std::fabs(lin_accum_ele) > l1) {
              auto x = -lin_accum_ele;
              if (lin_accum_ele >= static_cast<T>(0)) {
                x += l1;
              } else {
                x -= l1;
              }

              auto y = static_cast<T>(2) * l2;
              if (lr_power == static_cast<T>(-0.5)) {
                y += std::sqrt(new_accum) / lr;
              } else {
                y += std::pow(new_accum, -lr_power) / lr;
              }

              auto pre_shrink = x / y;
              param_data[tensor_ele_idx] = pre_shrink;
            } else {
              param_data[tensor_ele_idx] = static_cast<T>(0);
            }

            sq_accum_data[tensor_ele_idx] += grad_ele * grad_ele;
          }
        }
      } else {
        PADDLE_THROW("Unsupported Variable Type of Grad");
      }
    } else if (param_var->IsType<framework::SelectedRows>()) {
      auto lr = *ctx.Input<framework::Tensor>("LearningRate")->data<T>();

      auto* param_out =
          ctx.Output<framework::SelectedRows>("ParamOut");
      auto* sq_accum_out =
          ctx.Output<framework::SelectedRows>("SquaredAccumOut");
      auto* lin_accum_out =
          ctx.Output<framework::SelectedRows>("LinearAccumOut");

      auto* param_data = param_out->mutable_value()->data<T>();
      auto* sq_accum_data = sq_accum_out->mutable_value()->data<T>();
      auto* lin_accum_data = lin_accum_out->mutable_value()->data<T>();

      auto grad = ctx.Input<framework::SelectedRows>("Grad");
      auto grad_row_width = grad->value().dims()[1];

      math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
      auto grad_merge =
          merge_func(ctx.template device_context<DeviceContext>(), *grad);
      auto* grad_merge_data =
          grad_merge.mutable_value()->template data<T>();

      for (size_t i = 0; i < grad_merge.rows().size(); i++) {
        auto param_row_idx =
            param_out->AutoGrownIndex(grad_merge.rows()[i], false);
        auto sq_accum_row_idx =
            sq_accum_out->AutoGrownIndex(grad_merge.rows()[i], true);
        auto lin_accum_row_idx =
            lin_accum_out->AutoGrownIndex(grad_merge.rows()[i], true);

        for (int64_t j = 0; j < grad_row_width; j++) {
          auto grad_ele_idx = i * grad_row_width + j;
          auto param_ele_idx = param_row_idx * grad_row_width + j;
          auto sq_accum_ele_idx = sq_accum_row_idx * grad_row_width + j;
          auto lin_accum_ele_idx = lin_accum_row_idx * grad_row_width + j;

          auto grad_ele = grad_merge_data[grad_ele_idx];
          auto sq_accum_ele = sq_accum_data[sq_accum_ele_idx];

          auto new_accum = sq_accum_ele + grad_ele * grad_ele;

          if (lr_power == static_cast<T>(-0.5)) {
            lin_accum_data[lin_accum_ele_idx] +=
                grad_ele -
                (std::sqrt(new_accum) - std::sqrt(sq_accum_ele)) /
                lr * param_data[param_ele_idx];
          } else {
            lin_accum_data[lin_accum_ele_idx] +=
                grad_ele -
                (std::pow(new_accum, -lr_power) -
                 std::pow(sq_accum_ele, -lr_power)) /
                lr * param_data[param_ele_idx];
          }

          auto lin_accum_ele = lin_accum_data[lin_accum_ele_idx];

          if (std::fabs(lin_accum_ele) > l1) {
            auto x = -lin_accum_ele;
            if (lin_accum_ele >= static_cast<T>(0)) {
              x += l1;
            } else {
              x -= l1;
            }

            auto y = static_cast<T>(2) * l2;
            if (lr_power == static_cast<T>(-0.5)) {
              y += std::sqrt(new_accum) / lr;
            } else {
              y += std::pow(new_accum, -lr_power) / lr;
            }

            auto pre_shrink = x / y;
            param_data[param_ele_idx] = pre_shrink;
          } else {
            param_data[param_ele_idx] = static_cast<T>(0);
          }

          sq_accum_data[sq_accum_ele_idx] += grad_ele * grad_ele;
        }
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Parameter");
    }
  }
};

}  // namespace operators
}  // namespace paddle
