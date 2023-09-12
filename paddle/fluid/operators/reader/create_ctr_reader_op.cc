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

#include "paddle/fluid/operators/reader/ctr_reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CreateCTRReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) return;

    const std::string& queue_name = Input("blocking_queue");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE_NOT_NULL(
        queue_holder_var,
        platform::errors::PreconditionNotMet(
            "No LoDTensorBlockingQueueHolder variable with name %s found",
            queue_name));
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();

    auto thread_num = Attr<int>("thread_num");
    auto sparse_slots = Attr<std::vector<std::string>>("sparse_slots");
    auto dense_slot_index = Attr<std::vector<int>>("dense_slot_index");
    auto sparse_slot_index = Attr<std::vector<int>>("sparse_slot_index");
    auto batch_size = Attr<int>("batch_size");
    auto file_type = Attr<std::string>("file_type");
    auto file_format = Attr<std::string>("file_format");
    auto file_list = Attr<std::vector<std::string>>("file_list");
    DataDesc data_desc(batch_size,
                       file_list,
                       file_type,
                       file_format,
                       dense_slot_index,
                       sparse_slot_index,
                       sparse_slots);
    VLOG(1) << data_desc;
    out->Reset(std::make_shared<CTRReader>(
        queue_holder->GetQueue(), thread_num, data_desc));
  }
};

class CreateCTRReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddInput("blocking_queue",
             "Name of the `LoDTensorBlockingQueueHolder` variable");
    AddAttr<int>("thread_num", "the thread num to read data");
    AddAttr<int>("batch_size", "the batch size of read data");
    AddAttr<std::string>("file_type", "plain or gzip").SetDefault("plain");
    AddAttr<std::string>("file_format", "svm or csv").SetDefault("csv");
    AddAttr<std::vector<std::string>>("file_list",
                                      "The list of files that need to read");
    AddAttr<std::vector<int>>(
        "dense_slot_index",
        "the dense slots id that should be extract from file")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "sparse_slot_index",
        "the sparse slots id that should be extract from file")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("sparse_slots",
                                      "the sparse slots id that should be "
                                      "extract from file, used when file "
                                      "format is svm");

    AddComment(R"DOC(
      Create CTRReader to support read ctr data with cpp.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_ctr_reader,
                              reader::CreateCTRReaderOp,
                              reader::CreateCTRReaderOpMaker);
