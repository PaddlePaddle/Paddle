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
#include <fstream>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorBase {
 public:
  LoadOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    platform::RecordEvent record_event(Type(), dev_ctx);

    auto filename = Attr<std::string>("file_path");
    std::ifstream fin(filename);
    PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s for load op",
                   filename);

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_name);

    auto *tensor = out_var->GetMutable<framework::LoDTensor>();

    DeserializeFromStream(fin, tensor, *dev_ctx);

    if (platform::is_gpu_place(place)) {
      // copy CPU to GPU
      framework::LoDTensor cpu_tensor;
      cpu_tensor.ShareDataWith(*tensor);
      cpu_tensor.set_lod(tensor->lod());

      // reset tensor
      out_var->Clear();
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->set_lod(cpu_tensor.lod());
      TensorCopy(cpu_tensor, place, *dev_ctx, tensor);
    }
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoadOpProtoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "(Tensor) The tensor need to be loaded");
    AddAttr<std::string>("file_path",
                         "(string) "
                         "Variable will be loaded from \"file_path\".")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddComment(R"DOC(
Load Operator.

Load operator will load a tensor variable from disk file.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);
