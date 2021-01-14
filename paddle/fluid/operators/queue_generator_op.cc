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

#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {

class QueueGeneratorOp : public framework::OperatorBase {
 public:
  QueueGeneratorOp(const std::string& type,
                   const framework::VariableNameMap& inputs,
                   const framework::VariableNameMap& outputs,
                   const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    std::vector<std::string> names = Attr<std::vector<std::string>>("names");
    PADDLE_ENFORCE_GT(names.size(), 0, platform::errors::InvalidArgument(
                                           "The attribute 'names' for "
                                           "Op(queue_generator) must be set."));

    int capacity = Attr<int>("capacity");
    PADDLE_ENFORCE_GT(capacity, 0,
                      platform::errors::InvalidArgument(
                          "The attribute 'capacity' for Op(queue_generator) "
                          "must be set a positive value, "
                          "but the one received is %d.",
                          capacity));

    // generate queue vars and initialize them
    for (const auto& name : names) {
      GenerateQueue(&scope, name, capacity);
    }
  }

 private:
  void GenerateQueue(const framework::Scope* scope, const std::string& name,
                     size_t capacity) const {
    auto var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound(
                 "Can't find var named '%s' in the global scope.", name));
    auto ptr = var->GetMutable<reader::LoDTensorBlockingQueueHolder>();
    ptr->InitOnce(capacity);

    VLOG(3) << "generated a LodTensorBlockingQueue var named: " << name;
  }
};

class QueueGeneratorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
QueueGenerator operator
Generate and initialize one or more LodTensorBlockingQueueHolders.
)DOC");
    AddAttr<std::vector<std::string>>(
        "names",
        "['name1', 'name2', ...] "
        "list of names for LodTensorBlockingQueueHolders")
        .SetDefault({});
    AddAttr<int>("capacity", "queue capacity").SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(queue_generator, ops::QueueGeneratorOp,
                             ops::QueueGeneratorOpMaker);
