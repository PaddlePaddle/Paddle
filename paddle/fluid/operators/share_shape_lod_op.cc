// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ShareShapeLodOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->IsRuntime() == false,
        "InferShape of ShareShapeLodOp should not be called during runtime.");

    PADDLE_ENFORCE(ctx->HasInputs("X"), "Inputs(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"), "Outputs(Out) must exist");

    const auto& share_shape =
        ctx->Attrs().Get<std::vector<bool>>("share_shape");
    const auto& share_lod = ctx->Attrs().Get<std::vector<bool>>("share_lod");

    size_t n = share_shape.size();
    PADDLE_ENFORCE(n == share_lod.size() && n == ctx->Inputs("X").size() &&
                       n == ctx->Outputs("Out").size(),
                   "Size of Inputs(X) %d, Outputs(Out) %d, Attr(share_shape) "
                   "%d, Attr(share_lod) %d must be the same",
                   ctx->Inputs("X").size(), ctx->Outputs("Out").size(), n,
                   share_lod.size());

    // FIXME(zjl): Hot-fix. Set all outputs be persistable to prevent memory
    // reuse
    auto output_ptrs = ctx->GetOutputVarPtrs("Out");
    for (auto& out : output_ptrs) {
      auto var_desc = boost::get<framework::VarDesc*>(out);
      var_desc->SetPersistable(true);
    }

    std::vector<framework::DDim> output_dims = ctx->GetInputsDim("X");
    for (size_t i = 0; i < n; ++i) {
      if (!share_shape[i]) {
        output_dims[i] = framework::make_ddim({static_cast<int64_t>(0)});
      }

      if (share_lod[i]) {
        ctx->ShareLoD("X", "Out", i, i);
      }
    }

    ctx->SetOutputsDim("Out", output_dims);
  }
};

class ShareShapeLodOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 protected:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    framework::ExecutionContext ctx(
        *this, scope, *(platform::DeviceContextPool::Instance().Get(place)));
    auto x_vars = ctx.MultiInputVar("X");
    auto out_vars = ctx.MultiOutputVar("Out");
    auto& share_shape = ctx.Attr<std::vector<bool>>("share_shape");
    auto& share_lod = ctx.Attr<std::vector<bool>>("share_lod");
    size_t n = share_shape.size();
    PADDLE_ENFORCE(
        n == share_lod.size() && n == x_vars.size() && n == out_vars.size(),
        "Size of Inputs(X) %d, Outputs(Out) %d, Attr(share_shape) %d, "
        "Attr(share_lod) %d must be the same",
        x_vars.size(), out_vars.size(), n, share_lod.size());

    for (size_t i = 0; i < n; ++i) {
      auto& in_tensor = x_vars[i]->Get<framework::LoDTensor>();
      auto* out_tensor = out_vars[i]->GetMutable<framework::LoDTensor>();
      if (share_shape[i]) {
        // In runtime, should not insert dim 0
        out_tensor->Resize(in_tensor.dims());
      }

      if (share_lod[i]) {
        out_tensor->set_lod(in_tensor.lod());
      }
    }
  }
};

class ShareShapeLodOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input LoDTensors").AsDuplicable();
    AddOutput("Out",
              "Output LoDTensors. Number of Outputs(Out) must be the same as "
              "Inputs(X)")
        .AsDuplicable();
    AddAttr<std::vector<bool>>("share_shape",
                               "Whether to share shapes between X and Out");
    AddAttr<std::vector<bool>>("share_lod",
                               "Whether to share lods between X and Out");
    AddComment(R"DOC(
ShareShapeLod Operator.

This Operator would share the shapes or lods between Inputs(X) and
Outputs(Out), without sharing memory. Attr(share_shape) and 
Attr(share_lod) is used to indicated whether to share shape and lod 
between X(i) and Out(i).

This Operator may be useful in some ops whose backward operator 
does not need the forward memory data.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(share_shape_lod, ops::ShareShapeLodOp,
                  ops::ShareShapeLodOpMaker, ops::ShareShapeLodOpInferShape,
                  paddle::framework::EmptyGradOpMaker);
