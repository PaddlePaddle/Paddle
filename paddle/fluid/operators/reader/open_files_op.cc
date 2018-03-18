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

#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class MultipleReader : public framework::ReaderBase {
 public:
  struct Quota {};

  MultipleReader(const std::vector<std::string>& file_names,
                 const std::vector<framework::DDim>& dims, size_t thread_num)
      : file_names_(file_names), dims_(dims), thread_num_(thread_num) {
    PADDLE_ENFORCE_GT(thread_num_, 0);
    StartNewScheduler();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  bool HasNext() const override;
  void ReInit() override;

 private:
  void StartNewScheduler();
  void ScheduleThreadFunc();
  void PrefetchThreadFunc(std::string file_name);

  std::vector<std::string> file_names_;
  std::vector<framework::DDim> dims_;
  size_t thread_num_;
  framework::Channel<size_t>* waiting_file_idx_;
  framework::Channel<Quota>* thread_quotas_;
  framework::Channel<std::vector<framework::LoDTensor>>* buffer_;
  mutable std::vector<framework::LoDTensor> local_buffer_;
};

void MultipleReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  if (!HasNext()) {
    PADDLE_THROW("There is no next data!");
  }

  if (local_buffer_.empty()) {
    buffer_->Receive(&local_buffer_);
  }
  *out = local_buffer_;
  local_buffer_.clear();
}

bool MultipleReader::HasNext() const {
  return local_buffer_.empty() ? buffer_->Receive(&local_buffer_) : true;
}

void MultipleReader::ReInit() {
  buffer_->Close();
  thread_quotas_->Close();
  waiting_file_idx_->Close();
  local_buffer_.clear();

  StartNewScheduler();
}

void MultipleReader::StartNewScheduler() {
  waiting_file_idx_ = framework::MakeChannel<size_t>(file_names_.size());
  thread_quotas_ = framework::MakeChannel<Quota>(thread_num_);
  buffer_ =
      framework::MakeChannel<std::vector<framework::LoDTensor>>(thread_num_);

  for (size_t i = 0; i < file_names_.size(); ++i) {
    waiting_file_idx_->Send(&i);
  }
  waiting_file_idx_->Close();
  for (size_t i = 0; i < thread_num_; ++i) {
    Quota quota;
    thread_quotas_->Send(&quota);
  }

  std::thread scheduler([this] { ScheduleThreadFunc(); });
  scheduler.detach();
}

void MultipleReader::ScheduleThreadFunc() {
  VLOG(5) << "MultipleReader schedule thread starts.";
  size_t completed_thread_num = 0;
  Quota quota;
  while (thread_quotas_->Receive(&quota)) {
    size_t file_idx;
    if (waiting_file_idx_->Receive(&file_idx)) {
      // Still have files to read. Start a new prefetch thread.
      std::string file_name = file_names_[file_idx];
      std::thread prefetcher(
          [this, file_name] { PrefetchThreadFunc(file_name); });
      prefetcher.detach();
    } else {
      // No more file to read.
      ++completed_thread_num;
      if (completed_thread_num == thread_num_) {
        thread_quotas_->Close();
        buffer_->Close();
        break;
      }
    }
  }
  VLOG(5) << "MultipleReader schedule thread terminates.";
}

void MultipleReader::PrefetchThreadFunc(std::string file_name) {
  VLOG(5) << "The prefetch thread of file '" << file_name << "' starts.";
  std::unique_ptr<framework::ReaderBase> reader =
      CreateReaderByFileName(file_name, dims_);
  while (reader->HasNext()) {
    std::vector<framework::LoDTensor> ins;
    reader->ReadNext(&ins);
    if (!buffer_->Send(&ins)) {
      VLOG(5) << "WARNING: The buffer channel has been closed. The prefetch "
                 "thread of file '"
              << file_name << "' will terminate.";
      break;
    }
  }
  Quota quota;
  thread_quotas_->Send(&quota);
  VLOG(5) << "The prefetch thread of file '" << file_name << "' terminates.";
}

class OpenFilesOp : public framework::OperatorBase {
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
    const auto& file_names = Attr<std::vector<std::string>>("file_names");
    PADDLE_ENFORCE(!file_names.empty(), "No file to be read!");
    const size_t thread_num = Attr<int>("thread_num");

    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new MultipleReader(
        file_names, RestoreShapes(shape_concat, ranks), thread_num));
  }
};

class OpenFilesOpMaker : public FileReaderMakerBase {
 public:
  OpenFilesOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : FileReaderMakerBase(op_proto, op_checker) {
    AddAttr<std::vector<std::string>>("file_names", "Files to be read.");
    AddAttr<int>("thread_num", "The maximal concurrent prefetch thread number.")
        .GreaterThan(0);

    AddComment(R"DOC(
      OpenFiles Operator

      An OpenFilesOp creates a MultipleReader, which is able to 
      read data multi-threaded from multiple files.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(open_files, reader::OpenFilesOp,
                              reader::OpenFilesOpMaker);
