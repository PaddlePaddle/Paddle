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

#include <nccl.h>
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

class GenQueueOp : public framework::OperatorBase {
 public:
  GenQueueOp(const std::string& type, const framework::VariableNameMap& inputs,
             const framework::VariableNameMap& outputs,
             const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put queue var in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    // framework::Scope& local_scope = scope.NewScope();

    std::vector<std::string> queue_names =
        Attr<std::vector<std::string>>("queue_names");
    PADDLE_ENFORCE_GT(queue_names.size(), 0,
                      "The attribute of 'queue_names' for GenQueueOP must "
                      "contain one or more names.");
    int queue_size = Attr<int>("queue_size");
    PADDLE_ENFORCE_GT(queue_names.size(), 0,
                      "The attribute of 'queue_names' for GenQueueOP must "
                      "contain one or more names.");

    // generate queue vars and initialize them
    for (auto var_name : queue_names) {
      // Generate(&local_scope, dev_ctx, var_name, queue_size);
      Generate(&scope, dev_ctx, var_name, queue_size);
    }
  }

 private:
  void Generate(const framework::Scope* scope,
                const platform::DeviceContext& dev_ctx,
                const std::string& var_name, size_t queue_size) const {
    auto var = scope->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(var, "can't find var_name:%s", var_name);
    auto var_ptr = var->GetMutable<reader::LoDTensorBlockingQueueHolder>();
    var_ptr->InitOnce(queue_size);

    VLOG(3) << "generated var: " << var_name;
  }
};

class GenQueueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
GenQueue operator
Generate LodTensorBlockingQueueHolders, and initialize them.
)DOC");
    AddAttr<std::vector<std::string>>(
        "queue_names",
        "['queue_name1', 'queue_name2', ...] "
        "list of names for LodTensorBlockingQueueHolders")
        .SetDefault({});
    AddAttr<int>("queue_size", "queue size").SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_queue, ops::GenQueueOp, ops::GenQueueOpMaker);
