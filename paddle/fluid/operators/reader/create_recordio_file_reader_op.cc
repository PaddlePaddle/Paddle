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
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/recordio/scanner.h"

namespace paddle {
namespace operators {
namespace reader {
template <bool ThreadSafe>
class RecordIOFileReader : public framework::FileReader {
 public:
  explicit RecordIOFileReader(const std::string& filename)
      : scanner_(filename),
        dev_ctx_(*platform::DeviceContextPool::Instance().Get(
            platform::CPUPlace())) {
    if (ThreadSafe) {
      mutex_.reset(new std::mutex());
    }
    LOG(INFO) << "Creating file reader" << filename;
  }

 protected:
  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override {
    platform::LockGuardPtr<std::mutex> guard(mutex_);
    bool ok = framework::ReadFromRecordIO(&scanner_, dev_ctx_, out);
    if (!ok) {
      out->clear();
    }
  }

  void StartImpl() override { scanner_.Reset(); }

 private:
  std::unique_ptr<std::mutex> mutex_;
  recordio::Scanner scanner_;
  const platform::DeviceContext& dev_ctx_;
};

class CreateRecordIOReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    std::string filename = Attr<std::string>("filename");
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();

    out->Reset(std::make_shared<RecordIOFileReader<true>>(filename));
  }
};

class CreateRecordIOReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<std::string>(
        "filename",
        "The filename of record file. This file will given to reader.");
    AddComment(R"DOC(
Open a recordio file and return the reader object. The returned reader object
is thread-safe.

NOTE: This is a very low-level API. It is used for debugging data file or
training. Please use `open_files` instead of this API for production usage.
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

REGISTER_FILE_READER(recordio, reader::RecordIOFileReader<false>);
