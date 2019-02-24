// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

static framework::proto::VarType::Type kDefaultDtype =
    static_cast<framework::proto::VarType::Type>(0);

class AllocContinuousSpaceForGradKernel : public framework::OperatorBase {
 public:
  AllocContinuousSpaceForGradKernel(const std::string &type,
                                    const framework::VariableNameMap &inputs,
                                    const framework::VariableNameMap &outputs,
                                    const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &param_var_names = Inputs("Parameters");
    auto &grad_var_names = Outputs("Gradients");
    auto fused_var_name = Output("FusedOutput");

    PADDLE_ENFORCE_GT(param_var_names.size(), static_cast<size_t>(0));
    PADDLE_ENFORCE_EQ(param_var_names.size(), grad_var_names.size());

    size_t mem_size = 0;
    framework::proto::VarType::Type dtype = kDefaultDtype;
    GetMemSizeAndDtype(scope, param_var_names, grad_var_names, &mem_size,
                       &dtype);

    auto fused_var = scope.FindVar(fused_var_name);
    PADDLE_ENFORCE_NOT_NULL(fused_var);
    auto out_tensor = fused_var->GetMutable<framework::LoDTensor>();
    out_tensor->Resize(framework::make_ddim({static_cast<int64_t>(mem_size)}))
        .mutable_data(dev_place, dtype);

    int64_t offset = 0;
    for (size_t i = 0; i < grad_var_names.size(); ++i) {
      auto out_var = scope.FindVar(grad_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(out_var);
      auto out_t = out_var->GetMutable<framework::LoDTensor>();
      int64_t len = out_t->numel();
      auto dim = out_t->dims();
      out_t->ShareDataWith(out_tensor->Slice(offset, offset + len)).Resize(dim);

      offset += len;
      VLOG(10) << "alloc_continuous_space_for_grad: output("
               << grad_var_names[i] << ") ,dim:(" << dim << ")"
               << " Address: " << out_t->data<void>();
    }
  }

  void GetMemSizeAndDtype(const framework::Scope &scope,
                          const std::vector<std::string> &param_var_names,
                          const std::vector<std::string> &grad_var_names,
                          size_t *mem_size,
                          framework::proto::VarType::Type *dtype) const {
    *mem_size = 0;
    for (size_t i = 0; i < grad_var_names.size(); ++i) {
      auto grad_var = scope.FindVar(grad_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(grad_var, "%s", grad_var_names[i]);

      // Only support LoDTensor,
      PADDLE_ENFORCE(grad_var->IsType<framework::LoDTensor>(), "%s",
                     grad_var_names[i]);

      // Note: Assume that the dtype of parameter and gradient are the same.
      // Doesn't get dtype from grad_var in runtime.
      auto param_var = scope.FindVar(param_var_names[i]);
      PADDLE_ENFORCE_NOT_NULL(param_var, "%s", param_var_names[i]);
      auto &p_tensor = param_var->Get<framework::LoDTensor>();

      auto p_dtype = p_tensor.type();
      if (*dtype == kDefaultDtype) {
        PADDLE_ENFORCE_NE(p_dtype, kDefaultDtype, "%s", grad_var_names[i]);
        *dtype = p_dtype;
      }
      PADDLE_ENFORCE_EQ(p_dtype, *dtype, "%s", grad_var_names[i]);

      grad_var->GetMutable<framework::LoDTensor>()->Resize(p_tensor.dims());
      auto size = p_tensor.numel();
      VLOG(10) << "alloc_continuous_space_for_grad: input(" << grad_var_names[i]
               << ") ,dim:(" << p_tensor.dims() << ")";
      PADDLE_ENFORCE_GT(size, 0, "%s", grad_var_names[i]);
      *mem_size += size;
    }
  }
};

class AllocContinuousSpaceForGradKernelMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Parameters", "A set of variables.").AsDuplicable();
    AddOutput("Gradients", "A set of variables.").AsDuplicable();
    AddOutput("FusedOutput", "");
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(alloc_continuous_space_for_grad,
                  paddle::operators::AllocContinuousSpaceForGradKernel,
                  paddle::operators::AllocContinuousSpaceForGradKernelMaker);
