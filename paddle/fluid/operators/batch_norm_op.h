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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/norm_utils.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename DeviceContext, typename T>
inline void ResizeToChannelFirst(const framework::ExecutionContext& context,
                                 const Tensor* input,
                                 Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[4];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    in_dims_vec[4] = input->dims()[3];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[3];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
  } else if (dim == 1) {
    transformed_input->Resize(input->dims());

    auto in_dims_vec = framework::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(framework::make_ddim(in_dims_vec));
    transformed_input->mutable_data<T>(context.GetPlace());
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelFirst(const framework::ExecutionContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 4, 1, 2, 3};
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);

  } else if (dim == 2) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 3, 1, 2};
    math::Transpose<DeviceContext, T, 4> trans4;
    trans4(dev_ctx, *input, transformed_input, axis);
  } else if (dim == 1) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 2, 1};
    math::Transpose<DeviceContext, T, 3> trans3;
    trans3(dev_ctx, *input, transformed_input, axis);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelLast(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 2, 3, 4, 1};
    math::Transpose<DeviceContext, T, 5> trans5;
    trans5(dev_ctx, *input, transformed_input, axis);

  } else if (dim == 2) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 2, 3, 1};
    math::Transpose<DeviceContext, T, 4> trans4;
    trans4(dev_ctx, *input, transformed_input, axis);
  } else if (dim == 1) {
    auto& dev_ctx = context.template device_context<DeviceContext>();
    std::vector<int> axis{0, 2, 1};
    math::Transpose<DeviceContext, T, 3> trans3;
    trans3(dev_ctx, *input, transformed_input, axis);
  }
}

class BatchNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class BatchNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class BatchNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T>
class BatchNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override;
};

class BatchNormOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

template <typename DeviceContext, typename T>
class BatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class BatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
