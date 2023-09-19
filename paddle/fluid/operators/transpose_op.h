// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class TransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

// FIXME(zcd): transpose2 adds an intermediate output(XShape) based on
// transpose, the XShape is used to carry the shape and lod of X which
// will be used in transpose_grad, in this way, the framework can reuse
// the memory of X immediately the transpose2_op is finished.
// Considering compatibility issues, we could not fix transpose2_op
class Transpose2Op : public TransposeOp {
 public:
  using TransposeOp::TransposeOp;
  Transpose2Op(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : TransposeOp(type, inputs, outputs, attrs) {}
  void InferShape(framework::InferShapeContext *ctx) const override;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override;
};

class Transpose2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

}  // namespace operators
}  // namespace paddle
