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

#include "paddle/fluid/operators/reader/py_array_feed_queue.h"

namespace paddle {
namespace operators {
namespace reader {

class PyArrayReader : public framework::ReaderBase {
 public:
  explicit PyArrayReader(const std::shared_ptr<PyArrayFeedQueue>& queue) {
    PADDLE_ENFORCE(queue != nullptr, "PyArrayFeedQueue must not be null");
    queue_ = queue;
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    *out = queue_->Dequeue();
  }

  void ReInit() override {}

 private:
  std::shared_ptr<PyArrayFeedQueue> queue_;
};

class CreatePyArrayReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const std::string& feeder_name = Attr<std::string>("feeder_name");
    auto* feeder_holder_var = scope.FindVar(feeder_name);
    PADDLE_ENFORCE(feeder_holder_var != nullptr,
                   "No PyArrayFeedQueue variable with name %s found",
                   feeder_name);
    auto* feeder_holder =
        feeder_holder_var->template GetMutable<PyArrayFeedQueueHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new PyArrayReader(feeder_holder->GetFeeder()));
  }
};

class CreatePyArrayReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<std::string>("feeder_name",
                         "Name of the `PyArrayFeedQueueHolder` variable");

    AddComment(R"DOC(
			Create PyArrayReader to accept Python data feeding.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_py_array_reader,
                              reader::CreatePyArrayReaderOp,
                              reader::CreatePyArrayReaderOpMaker);
