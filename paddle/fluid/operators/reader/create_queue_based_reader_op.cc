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

#include "paddle/fluid/operators/reader/queue_based_reader.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CreateQueueBasedReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 protected:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) return;

    const std::string& queue_name = Input("blocking_queue");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE_NOT_NULL(queue_holder_var,
                            "No MultiDeviceLoDTensorBlockingQueueHolder "
                            "variable with name %s found",
                            queue_name);

    auto* queue_holder = queue_holder_var->template GetMutable<
        std::shared_ptr<MultiDeviceLoDTensorBlockingQueueHolder>>();

    out->Reset((*queue_holder)->CreateNextReader());
  }
};

class CreateQueueBasedReaderOpMaker : public FileReaderMakerBase {
 public:
  void Apply() override {
    AddInput("blocking_queue",
             "Name of the `MultiDeviceLoDTensorBlockingQueueHolder` variable");

    AddComment(R"DOC(
      Create QueueBasedReader to support LoDTensor data feeding in Python side.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_queue_based_reader,
                              reader::CreateQueueBasedReaderOp,
                              reader::CreateQueueBasedReaderOpMaker,
                              paddle::framework::EmptyGradOpMaker);
