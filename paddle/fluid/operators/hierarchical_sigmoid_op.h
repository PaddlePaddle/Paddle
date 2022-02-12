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

#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/operators/math/matrix_bit_code.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
using platform::Transform;
using framework::LoDTensor;

static std::vector<int64_t> PathToRows(const LoDTensor& path) {
  std::set<int64_t> rows;
  const int64_t* paths = path.data<int64_t>();
  for (int64_t i = 0; i < path.numel(); ++i) {
    int64_t row = paths[i];
    if (row < 0) {
      continue;
    }
    rows.emplace(row);
  }
  return std::vector<int64_t>(rows.begin(), rows.end());
}
template <typename DeviceContext, typename T>
class HierarchicalSigmoidOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& in = GET_DATA_SAFELY(ctx.Input<LoDTensor>("X"), "Input", "X",
                               "HierarchicalSigmoid");
    auto& w = GET_DATA_SAFELY(ctx.Input<LoDTensor>("W"), "Input", "W",
                              "HierarchicalSigmoid");
    auto* path = ctx.Input<LoDTensor>("PathTable");
    auto* code = ctx.Input<LoDTensor>("PathCode");
    auto& label = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Label"), "Input",
                                  "Label", "HierarchicalSigmoid");
    auto* bias = ctx.Input<LoDTensor>("Bias");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* pre_out = ctx.Output<LoDTensor>("PreOut");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    // for remote prefetch

    bool is_custom = false;
    if (path) {
      is_custom = true;
    }
    int64_t code_length =
        path ? path->dims()[1] : math::FindLastSet(num_classes - 1);
    int64_t batch_size = in.dims()[0];
    LoDTensor sum;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* pre_out_data = pre_out->mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
    // Not all class(leaf) nodes' path lengths equal code_length, thus init as
    // 0s can avoid out of path's loss.
    pten::funcs::SetConstant<DeviceContext, T> zero;
    zero(dev_ctx, pre_out, static_cast<T>(0.0));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    pten::funcs::RowwiseSum<DeviceContext, T> row_sum;

    std::unique_ptr<math::MatrixBitCodeFunctor<T>> bit_code;
    if (!is_custom) {
      bit_code.reset(new math::MatrixBitCodeFunctor<T>(
          num_classes, label.template data<int64_t>()));
    } else {
      bit_code.reset(new math::MatrixBitCodeFunctor<T>(
          *path, *code, label.template data<int64_t>()));
    }

    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    auto sum_mat = EigenMatrix<T>::From(sum);
    out->mutable_data<T>(ctx.GetPlace());
    auto out_mat = framework::EigenMatrix<T>::From(*out);
    if (bias) {
      bit_code->Add(*bias, pre_out);
    }
    bit_code->Mul(pre_out, w, in);
    // clip to [-40, 40]
    Transform<DeviceContext> trans;
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out->numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    bit_code->Sum(*pre_out, out, static_cast<T>(-1));
    // use softrelu to calculate cross entropy
    pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();
    row_sum(dev_ctx, *pre_out, &sum);
    // TODO(guosheng): Subtract the out of path's loss, since not all
    // class(leaf) nodes' path lengths equal code_length. But it won't break the
    // gradient check since both have the out of path's loss and will cancel out
    // each other.
    out_mat.device(place) = sum_mat + out_mat;
  }
};

template <typename DeviceContext, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& in = GET_DATA_SAFELY(ctx.Input<LoDTensor>("X"), "Input", "X",
                               "HierarchicalSigmoidGrad");
    auto& w = GET_DATA_SAFELY(ctx.Input<LoDTensor>("W"), "Input", "W",
                              "HierarchicalSigmoidGrad");
    auto* path = ctx.Input<LoDTensor>("PathTable");
    auto* code = ctx.Input<LoDTensor>("PathCode");
    auto* in_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    bool is_sparse = ctx.Attr<bool>("is_sparse");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    pten::funcs::SetConstant<DeviceContext, T> zero;
    auto& label = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Label"), "Input",
                                  "Label", "HierarchicalSigmoidGrad");
    auto& pre_out = GET_DATA_SAFELY(ctx.Input<LoDTensor>("PreOut"), "Input",
                                    "PreOut", "HierarchicalSigmoidGrad");
    auto& out_grad = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>(framework::GradVarName("Out")), "Input",
        framework::GradVarName("Out"), "HierarchicalSigmoidGrad");
    LoDTensor pre_out_grad;

    pre_out_grad.mutable_data<T>(pre_out.dims(), ctx.GetPlace());
    in_grad->mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, in_grad, static_cast<T>(0.0));

    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));

    bool is_custom = false;
    if (path) {
      is_custom = true;
    }

    std::unique_ptr<math::MatrixBitCodeFunctor<T>> bit_code;
    if (!is_custom) {
      bit_code.reset(new math::MatrixBitCodeFunctor<T>(
          num_classes, label.template data<int64_t>()));
    } else {
      bit_code.reset(new math::MatrixBitCodeFunctor<T>(
          *path, *code, label.template data<int64_t>()));
    }

    // softrelu derivative

    auto blas = math::GetBlas<DeviceContext, T>(ctx);

    auto* pre_out_grad_data = pre_out_grad.data<T>();
    auto* pre_out_data = pre_out.template data<T>();
    auto n = pre_out.numel();
    blas.VEXP(n, pre_out_data, pre_out_grad_data);
    blas.VINV(n, pre_out_grad_data, pre_out_grad_data);
    for (int64_t i = 0; i < n; ++i) {
      pre_out_grad_data[i] = 1.0 - pre_out_grad_data[i];
    }
    bit_code->Sub(&pre_out_grad);  // the gradient of clip(w * x + b)
    auto* out_grad_data = out_grad.template data<T>();

    int64_t dim0 = pre_out_grad.dims()[0];
    int64_t dim1 = pre_out_grad.dims()[1];
    for (int64_t i = 0; i < dim0; ++i) {
      T tmp = out_grad_data[i];
      blas.SCAL(dim1, tmp, pre_out_grad_data + i * dim1);
    }
    // TODO(guosheng): multiply pre_out_grad with subgradient of clipping to
    // be consistent with the clipping in forward.
    auto* bias_grad = ctx.Output<LoDTensor>(framework::GradVarName("Bias"));
    if (bias_grad) {
      bias_grad->mutable_data<T>(ctx.GetPlace());
      zero(dev_ctx, bias_grad, static_cast<T>(0.0));
      bit_code->AddGrad(pre_out_grad, bias_grad);
    }
    if (!is_sparse) {
      auto* w_grad = ctx.Output<LoDTensor>(framework::GradVarName("W"));
      w_grad->mutable_data<T>(ctx.GetPlace());
      zero(dev_ctx, w_grad, static_cast<T>(0.0));
      bit_code->MulGradWeight(pre_out_grad, w_grad, in);
    } else {
      PADDLE_ENFORCE_NOT_NULL(path,
                              platform::errors::NotFound(
                                  "Custom tree must be set for sparse mode!"));
      framework::Vector<int64_t> real_rows = PathToRows(*path);
      auto* w_grad =
          ctx.Output<pten::SelectedRows>(framework::GradVarName("W"));
      w_grad->set_rows(real_rows);
      // Build a map of id -> row_index to speed up finding the index of one id
      w_grad->set_height(w.dims()[0]);
      auto* w_grad_value = w_grad->mutable_value();
      framework::DDim temp_dim(w.dims());
      temp_dim[0] = real_rows.size();
      w_grad_value->mutable_data<T>(temp_dim, ctx.GetPlace());
      zero(dev_ctx, w_grad_value, static_cast<T>(0.0));
      bit_code->MulGradWeight(pre_out_grad, w_grad, in);
    }
    bit_code->MulGradError(pre_out_grad, w, in_grad);
  }
};

}  // namespace operators
}  // namespace paddle
