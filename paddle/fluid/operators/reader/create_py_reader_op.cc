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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CreatePyReaderOp : public framework::OperatorBase {
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
        "No LoDTensorBlockingQueueHolder variable with name %s found",
        queue_name);
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();

    /* Coverting shape_concat and ranks into DDim of each data.
     shape_concat and ranks are shapes and shape ranks of each data.E.g.
     shape_concat = [2,3,4,5,6], ranks = [3,2] means two data whose shapes are
     [2,3,4] and [5,6] respectively. */
    auto& shape_concat = Attr<std::vector<int>>("shape_concat");
    auto& ranks = Attr<std::vector<int>>("ranks");
    int shape_start_index = 0;
    std::vector<framework::DDim> dims;
    for (size_t i = 0; i < ranks.size(); ++i) {
      int shape_end_index = shape_start_index + ranks[i];
      auto shape = std::vector<int>(shape_concat.begin() + shape_start_index,
                                    shape_concat.begin() + shape_end_index);
      dims.push_back(framework::make_ddim(shape));
      shape_start_index = shape_end_index;
    }

    // Converts VarType from int to enum
    auto& dtype_int = Attr<std::vector<int>>("dtypes");
    std::vector<framework::proto::VarType::Type> var_types;
    for (size_t i = 0; i < dtype_int.size(); ++i) {
      var_types.push_back(
          static_cast<framework::proto::VarType::Type>(dtype_int[i]));
    }

    // Converts need_check_feed from int to bool
    auto& need_check_feed_int = Attr<std::vector<int>>("need_check_feed");
    std::vector<bool> need_check_feed;
    for (size_t i = 0; i < need_check_feed_int.size(); ++i) {
      need_check_feed.push_back(static_cast<bool>(need_check_feed_int[i]));
    }
    out->Reset(std::make_shared<PyReader>(queue_holder->GetQueue(), dims,
                                          var_types, need_check_feed));
  }
};

class CreatePyReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddInput("blocking_queue",
             "Name of the `LoDTensorBlockingQueueHolder` variable");

    AddComment(R"DOC(
      Create PyReader to support LoDTensor data feeding in Python side.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_py_reader, reader::CreatePyReaderOp,
                              reader::CreatePyReaderOpMaker);
