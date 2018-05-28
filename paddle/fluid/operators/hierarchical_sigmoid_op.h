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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matrix_bit_code.h"
#include "paddle/fluid/platform/transform.h"
namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
using platform::Transform;

template <typename DeviceContext, typename T>
class HierarchicalSigmoidOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* w = ctx.Input<framework::Tensor>("W");
    auto* ids = ctx.Input<framework::Tensor>("Ids");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* pre_out = ctx.Output<framework::Tensor>("PreOut");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    std::cout << "bias dims: " << bias->dims() << std::endl;
    std::cout << "num_classes: " << num_classes << std::endl;
    std::cout << "bias (c++): [" << std::endl;
    for (size_t i = 0; i < num_classes - 1; ++i)
      std::cout << bias->data<T>()[i] << ",";
    std::cout << "]\n";
    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];
    std::cout << "w (c++): \n [";
    for (size_t i = 0; i < num_classes - 1; ++i)
      for (int j = 0; j < w->dims()[1]; ++j) {
        std::cout << w->data<T>()[i * w->dims()[1] + j] << ",";
      }
    std::cout << "]\n";

    std::cout << "x  (c++): \n [";
    for (int64_t i = 0; i < batch_size; ++i)
      for (int j = 0; j < in->dims()[1]; ++j) {
        std::cout << in->data<T>()[i * in->dims()[1] + j] << ",";
      }
    std::cout << "]\n";
    // framework::Tensor pre_out;
    framework::Tensor sum;
    math::SetConstant<DeviceContext, T> zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto pre_out_data = pre_out->mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
    zero(dev_ctx, pre_out, static_cast<T>(0.0));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    math::RowwiseSum<DeviceContext, T> row_sum;
    math::MatrixBitCodeFunctor<T> bit_code(num_classes, ids->data<int64_t>());

    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    auto sum_mat = EigenMatrix<T>::From(sum);
    out->mutable_data<T>(ctx.GetPlace());
    auto out_mat = framework::EigenVector<T>::Flatten(*out);
    std::cout << "pre_out before bias (c++): \n [";
    for (int i = 0; i < batch_size; ++i)
      for (int j = 0; j < code_length; ++j) {
        std::cout << pre_out->data<T>()[i * code_length + j] << ",";
      }
    std::cout << "]\n";
    if (bias) {
      bit_code.Add(pre_out, *bias);
    }
    std::cout << "pre_out after bias (c++): \n [";
    for (int i = 0; i < batch_size; ++i)
      for (int j = 0; j < code_length; ++j) {
        std::cout << pre_out->data<T>()[i * code_length + j] << ",";
      }
    std::cout << "]\n";
    // for (int64_t i = 0; i < batch_size; ++i) {
    // auto w_i = w->Slice(i, i + 1);
    bit_code.Mul(pre_out, *w, *in);
    std::cout << "pre_out after Mul (c++): \n [";
    for (int i = 0; i < batch_size; ++i)
      for (int j = 0; j < code_length; ++j) {
        std::cout << pre_out->data<T>()[i * code_length + j] << ",";
      }
    std::cout << "]\n";
    // }
    // clip the matrix with (-40, 40)
    Transform<DeviceContext> trans;
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out->numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    bit_code.Sum(*pre_out, out, static_cast<T>(-1));
    std::cout << "out after sum (c++): \n [";
    for (int i = 0; i < out->dims()[0]; i++) {
      std::cout << out->data<T>()[i] << ",";
      std::cout << "]" << std::endl;
    }
    // softrelu with threshold is 40.0
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out->numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();
    std::cout << "pre_out after relu (c++): \n [";
    for (int i = 0; i < batch_size; ++i)
      for (int j = 0; j < code_length; ++j) {
        std::cout << pre_out->data<T>()[i * code_length + j] << ",";
      }
    std::cout << "]\n";
    row_sum(dev_ctx, *pre_out, &sum);
    out_mat.device(place) = sum_mat + out_mat;
    std::cout << "out contrast: [";
    for (int i = 0; i < out->dims()[0]; i++) {
      std::cout << out->data<T>()[i] << ",";
      std::cout << "]" << std::endl;
    }
  }
};

template <typename DeviceContext, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* w = ctx.Input<framework::Tensor>("W");
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* w_grad = ctx.Output<framework::Tensor>(framework::GradVarName("W"));
    auto* bias_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));
    auto* ids = ctx.Input<framework::Tensor>("Ids");
    auto* pre_out = ctx.Input<framework::Tensor>("PreOut");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];
    framework::Tensor pre_out_grad;
    pre_out_grad.mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    // auto& device_ctx = ctx.template device_context<DeviceContext>();
    auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
    auto pre_out_grad_mat = EigenMatrix<T>::From(pre_out_grad);
    // auto out_grad_mat = EigenMatrix<T>::From(*out_grad);
    // init pre_out_grad matrix with {1.0}
    // math::SetConstant<DeviceContext, T> one;
    math::MatrixBitCodeFunctor<T> bit_code(num_classes, ids->data<int64_t>());
    // one(device_ctx, pre_out_grad, static_cast<T>(1.0));
    // softrelu derivative
    /*
    auto dims = out_grad->dims();
    std::cout << "dims out_grad" << dims << std::endl;
    for (int i = 0; i < dims[0]; i++) {
        std::cout << "out_grad: \n[";
        std::cout << out_grad->data<T>()[i]
                      << ",";
        std::cout << "]" << std::endl;
    }
    std::cout << "code_length: " << code_length << std::endl;
    std::cout << "pre_out: \n[";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < code_length; j++) {
            std::cout << pre_out->data<T>()[i * code_length + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "pre_out_grad 1: \n[";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < code_length; j++) {
            std::cout << pre_out_grad.data<T>()[i * code_length + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
    bit_code.OutGrad(&pre_out_grad, *out_grad);
    /*
    std::cout << "pre_out_grad after outgrad: \n[";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < code_length; j++) {
            std::cout << pre_out_grad.data<T>()[i * code_length + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
    pre_out_grad_mat.device(place) =
        pre_out_grad_mat *
        (static_cast<T>(1.0) - static_cast<T>(1.0) / pre_out_mat.exp());
    /*
    std::cout << "pre_out_grad 2: \n[";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < code_length; j++) {
            std::cout << pre_out_grad.data<T>()[i * code_length + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
    bit_code.Sub(&pre_out_grad);
    /*
    std::cout << "pre_out_grad 3: \n[";
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < code_length; j++) {
            std::cout << pre_out_grad.data<T>()[i * code_length + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
    if (bias_grad) {
      bias_grad->mutable_data<T>(ctx.GetPlace());
      /*
      std::cout << "bias grad before  addgrad: \n[";
      for (int i = 0; i < bias_grad->dims()[1]; i++) {
          std::cout << bias_grad->data<T>()[i]<< ",";
      }
      */
      bit_code.AddGrad(pre_out_grad, bias_grad);
    }
    /*
    std::cout << "bias grad after addgrad: \n[";
    for (int i = 0; i < bias_grad->dims()[1]; i++) {
        std::cout << bias_grad->data<T>()[i]<< ",";
    }
    std::cout << "]" << std::endl;
    */
    in_grad->mutable_data<T>(ctx.GetPlace());
    w_grad->mutable_data<T>(ctx.GetPlace());
    /*
    std::cout << "w_grad before mulgradweight: \n[";
    for (int i = 0; i < w_grad->dims()[0]; i++) {
        for (int j = 0; j < w_grad->dims()[1]; j++) {
            std::cout << w_grad->data<T>()[i * w_grad->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "in_grad before mulgraderror: \n[";
    for (int i = 0; i < in_grad->dims()[0]; i++) {
        for (int j = 0; j < in_grad->dims()[1]; j++) {
            std::cout << in_grad->data<T>()[i * in_grad->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    // for (int i = 0; i < batch_size; ++i) {
    // auto w_i = w->Slice(i, i + 1);
    // auto in_i = in->Slice(i, i + 1);
    // auto in_grad_i = in_grad->Slice(i, i + 1);

    std::cout << "w: \n[";
    for (int i = 0; i < w->dims()[0]; i++) {
        for (int j = 0; j < w->dims()[1]; j++) {
            std::cout << w->data<T>()[i * w->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "x: \n[";
    for (int i = 0; i < in->dims()[0]; i++) {
        for (int j = 0; j < in->dims()[1]; j++) {
            std::cout << in->data<T>()[i * in->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
    bit_code.MulGradWeight(pre_out_grad, w_grad, *in);
    bit_code.MulGradError(pre_out_grad, *w, in_grad);
    // }
    /*
    std::cout << "w_grad after mulgradweight: \n[";
    for (int i = 0; i < w_grad->dims()[0]; i++) {
        for (int j = 0; j < w_grad->dims()[1]; j++) {
            std::cout << w_grad->data<T>()[i * w_grad->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "in_grad after mulgraderror: \n[";
    for (int i = 0; i < in_grad->dims()[0]; i++) {
        for (int j = 0; j < in_grad->dims()[1]; j++) {
            std::cout << in_grad->data<T>()[i * in_grad->dims()[1] + j]
                      << ",";
        }
    }
    std::cout << "]" << std::endl;
    */
  }
};

}  // namespace operators
}  // namespace paddle
