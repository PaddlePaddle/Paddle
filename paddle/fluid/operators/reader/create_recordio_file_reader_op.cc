//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/reader_op_registry.h"
#include "paddle/fluid/recordio/scanner.h"

namespace paddle {
namespace operators {
namespace reader {
class RecordIOFileReader : public framework::FileReader {
 public:
  RecordIOFileReader(const std::string& filename,
                     const std::vector<framework::DDim>& shapes)
      : FileReader(shapes),
        scanner_(filename),
        dev_ctx_(*platform::DeviceContextPool::Instance().Get(
            platform::CPUPlace())) {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    *out = framework::ReadFromRecordIO(scanner_, dev_ctx_);
  }

  bool HasNext() const override { return scanner_.HasNext(); }

  void ReInit() override { scanner_.Reset(); }

 private:
  recordio::Scanner scanner_;
  const platform::DeviceContext& dev_ctx_;
};

class CreateRecordIOReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& shape_concat = Attr<std::vector<int>>("shape_concat");
    const auto& ranks = Attr<std::vector<int>>("ranks");
    PADDLE_ENFORCE(!shape_concat.empty() && !ranks.empty());
    PADDLE_ENFORCE_EQ(std::accumulate(ranks.begin(), ranks.end(), 0),
                      int(shape_concat.size()),
                      "The accumulate of all ranks should be equal to the "
                      "shape concat's length.");
    std::vector<framework::DDim> shapes = RestoreShapes(shape_concat, ranks);
    std::string filename = Attr<std::string>("filename");

    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new RecordIOFileReader(filename, shapes));
  }
};

class CreateRecordIOReaderOpMaker : public FileReaderMakerBase {
 public:
  CreateRecordIOReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : FileReaderMakerBase(op_proto, op_checker) {
    AddAttr<std::string>("filename", "The filename of record io reader");
    AddComment(R"DOC(
      CreateRecordIOReader Operator

      Create a reader from a record io file
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_recordio_file_reader,
                              reader::CreateRecordIOReaderOp,
                              reader::CreateRecordIOReaderOpMaker);
