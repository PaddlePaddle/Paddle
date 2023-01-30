// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type_traits.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

<<<<<<< HEAD
=======
using LoDTensor = paddle::framework::LoDTensor;
using Tensor = paddle::framework::Tensor;

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
namespace paddle {
namespace operators {

class CopyCrossScopeOp : public framework::OperatorBase {
 public:
  CopyCrossScopeOp(const std::string& type,
                   const framework::VariableNameMap& inputs,
                   const framework::VariableNameMap& outputs,
                   const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    int num_micro_scopes = scope.kids().size();
    int num_micro_batches = Attr<int>("num_micro_batches");
    bool ToM = Attr<bool>("to_main_scope");
    PADDLE_ENFORCE_EQ(num_micro_scopes,
                      num_micro_batches,
                      platform::errors::InvalidArgument(
                          "For pipeline, number of micro scopes (%d) should "
                          "be equal to number of micro batches (%d).",
                          num_micro_scopes,
                          num_micro_batches));
    const std::string& id_name = Input("Id");
    auto* id_var = scope.FindVar(id_name);
    PADDLE_ENFORCE_NOT_NULL(
        id_var,
        platform::errors::NotFound("No variable with name %s found.", id_name));
<<<<<<< HEAD
    auto id_tensor = id_var->GetMutable<phi::DenseTensor>();
    auto it = scope.kids().begin();
    phi::DenseTensor cpu_id_tensor;
=======
    auto id_tensor = id_var->GetMutable<LoDTensor>();
    auto it = scope.kids().begin();
    framework::Tensor cpu_id_tensor;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle::framework::TensorCopySync(
        *id_tensor, platform::CPUPlace(), &cpu_id_tensor);
    auto id_value = cpu_id_tensor.data<int64_t>();
    for (auto i = 0; i < *id_value; i++) {
      it++;
    }
    if (it == scope.kids().end()) {
      if (ToM) {
        auto dst_scope = *it;
        const std::string& x_name = Input("X");
        auto* dst_var = dst_scope->FindVar(x_name);
        PADDLE_ENFORCE_NOT_NULL(
            dst_var,
            platform::errors::NotFound(
                "No variable with name %s found in source scope.", x_name));
        auto* main_var = scope.FindVar(x_name);
        PADDLE_ENFORCE_NOT_NULL(
            main_var,
            platform::errors::NotFound(
                "No variable with name %s found in destination scope.",
                x_name));
<<<<<<< HEAD
        auto dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
        auto main_tensor = main_var->GetMutable<phi::DenseTensor>();
=======
        auto dst_tensor = dst_var->GetMutable<LoDTensor>();
        auto main_tensor = main_var->GetMutable<LoDTensor>();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle::framework::TensorCopySync(
            *dst_tensor, main_tensor->place(), main_tensor);
      }
      return;
    }
    auto source_scope = *it;
    it++;
    auto dst_scope = *it;
    const std::string& x_name = Input("X");
    auto* source_var = source_scope->FindVar(x_name);
    PADDLE_ENFORCE_NOT_NULL(
        source_var,
        platform::errors::NotFound(
            "No variable with name %s found in source scope.", x_name));
    auto* dst_var = dst_scope->FindVar(x_name);
    PADDLE_ENFORCE_NOT_NULL(
        dst_var,
        platform::errors::NotFound(
            "No variable with name %s found in destination scope.", x_name));
<<<<<<< HEAD
    auto src_tensor = source_var->GetMutable<phi::DenseTensor>();
    auto dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
=======
    auto src_tensor = source_var->GetMutable<LoDTensor>();
    auto dst_tensor = dst_var->GetMutable<LoDTensor>();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle::framework::TensorCopySync(
        *src_tensor, dst_tensor->place(), dst_tensor);

    if (ToM) {
      auto* main_var = scope.FindVar(x_name);
      PADDLE_ENFORCE_NOT_NULL(
          main_var,
          platform::errors::NotFound(
              "No variable with name %s found in destination scope.", x_name));
<<<<<<< HEAD
      auto main_tensor = main_var->GetMutable<phi::DenseTensor>();
=======
      auto main_tensor = main_var->GetMutable<LoDTensor>();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      paddle::framework::TensorCopySync(
          *dst_tensor, main_tensor->place(), main_tensor);
    }
  }
};

class CopyCrossScopeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), The first input tensor of copy_cross_scope op, which "
             "is copying micro scope.");
    AddInput("Id",
             "(Tensor), The second input tensor of copy_cross_scope op, which "
             "is a id of the current micro scope.");
    AddAttr<bool>("to_main_scope", "Return current scope to main scope.")
        .SetDefault(false);
    AddAttr<int>("num_micro_batches", "Number of micro batches for pipeline.");
    AddComment(R"DOC(
<<<<<<< HEAD
      This op is used by pipeline to copy tensors across micro batch scopes.
      Copy the variable value of the giving Id's micro scope to the micro scope of Id + 1 position.
      If need to copy back to the main scope, using to_main_scope option to copy the variable value of
=======
      This op is used by pipeline to copy tensors across micro batch scopes. 
      Copy the variable value of the giving Id's micro scope to the micro scope of Id + 1 position. 
      If need to copy back to the main scope, using to_main_scope option to copy the variable value of 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      the current micro scope to the main scope.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(copy_cross_scope,
                             ops::CopyCrossScopeOp,
                             ops::CopyCrossScopeOpMaker);
