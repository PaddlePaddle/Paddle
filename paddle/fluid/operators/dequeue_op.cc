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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
using LoDTensor = paddle::framework::LoDTensor;
using LoDTensorBlockingQueueHolder =
    paddle::operators::reader::LoDTensorBlockingQueueHolder;

namespace paddle {
namespace operators {

class DequeueOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
  DequeueOp(const std::string& type, const framework::VariableNameMap& inputs,
            const framework::VariableNameMap& outputs,
            const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const std::string& queue_name = Attr<std::string>("queue_name");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE_NOT_NULL(
        queue_holder_var,
        "No LoDTensorBlockingQueueHolder variable with name %s found",
        queue_name);
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();
    auto& out_names = Outputs("Out");
    PADDLE_ENFORCE_GT(out_names.size(), 0, "No output set for dequeue op.");
    for (size_t i = 0; i < out_names.size(); ++i) {
      auto out_var = scope.FindVar(out_names[i]);
      VLOG(0) << "out var name:" << out_names[i];
      PADDLE_ENFORCE_NOT_NULL(out_var, "No variable with name %s found",
                              out_names[i]);
      auto* out_tensor = out_var->GetMutable<LoDTensor>();

      std::vector<LoDTensor> lod_tensor_vec;
      bool success = false;
      lod_tensor_vec = queue_holder->GetQueue()->Pop(&success);
      PADDLE_ENFORCE_EQ(success, true, "An error occurs for dequeue op.");
      PADDLE_ENFORCE_EQ(
          lod_tensor_vec.size(), 1,
          "For dequeue op, dequeue one element per call for Pop.");
      for (size_t j = 0; j < lod_tensor_vec.size(); ++j) {
        TensorCopySync(lod_tensor_vec[j], dev_place, out_tensor);
      }
    }
  }
};

class DequeueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<std::string>("queue_name",
                         "Name of the `LoDTensorBlockingQueueHolder` variable");
    AddOutput("Out", "A list of `lod_tensor` to dequeue and assigned.")
        .AsDuplicable();
    AddComment(R"DOC(
			Dequeue operator.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;

REGISTER_OPERATOR(dequeue, ops::DequeueOp, ops::DequeueOpMaker);
