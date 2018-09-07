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

#include <thread>  // NOLINT

#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class MultiFileReader : public framework::ReaderBase {
 public:
  MultiFileReader(const std::vector<std::string>& file_names,
                  const std::vector<framework::DDim>& dims, size_t thread_num,
                  size_t buffer_size)
      : file_names_(file_names), dims_(dims), buffer_size_(buffer_size) {
    prefetchers_.resize(thread_num);
    StartNewScheduler();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  ~MultiFileReader() { EndScheduler(); }

 private:
  void StartNewScheduler();
  void EndScheduler();
  void ScheduleThreadFunc();
  void PrefetchThreadFunc(std::string file_name, size_t thread_idx);

  std::vector<std::string> file_names_;
  std::vector<framework::DDim> dims_;
  std::thread scheduler_;
  std::vector<std::thread> prefetchers_;
  size_t buffer_size_;
  reader::BlockingQueue<size_t>* waiting_file_idx_;
  reader::BlockingQueue<size_t>* available_thread_idx_;
  reader::BlockingQueue<std::vector<framework::LoDTensor>>* buffer_;
};

void MultiFileReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  if (!buffer_->Receive(out)) {
    out->clear();
  }
}

void MultiFileReader::ReInit() {
  EndScheduler();
  StartNewScheduler();
}

void MultiFileReader::StartNewScheduler() {
  size_t thread_num = prefetchers_.size();
  waiting_file_idx_ = new reader::BlockingQueue<size_t>(file_names_.size());
  available_thread_idx_ = new reader::BlockingQueue<size_t>(thread_num);
  buffer_ = new reader::BlockingQueue<std::vector<framework::LoDTensor>>(
      buffer_size_);

  for (size_t i = 0; i < file_names_.size(); ++i) {
    waiting_file_idx_->Send(i);
  }
  waiting_file_idx_->Close();
  for (size_t i = 0; i < thread_num; ++i) {
    available_thread_idx_->Send(i);
  }

  scheduler_ = std::thread([this] { ScheduleThreadFunc(); });
}

void MultiFileReader::EndScheduler() {
  available_thread_idx_->Close();
  buffer_->Close();
  waiting_file_idx_->Close();
  if (scheduler_.joinable()) {
    scheduler_.join();
  }
  delete buffer_;
  delete available_thread_idx_;
  delete waiting_file_idx_;
}

void MultiFileReader::ScheduleThreadFunc() {
  VLOG(5) << "MultiFileReader schedule thread starts.";
  size_t completed_thread_num = 0;
  size_t thread_idx;
  while (available_thread_idx_->Receive(&thread_idx)) {
    std::thread& prefetcher = prefetchers_[thread_idx];
    if (prefetcher.joinable()) {
      prefetcher.join();
    }
    size_t file_idx;
    if (waiting_file_idx_->Receive(&file_idx)) {
      // Still have files to read. Start a new prefetch thread.
      std::string file_name = file_names_[file_idx];
      prefetcher = std::thread([this, file_name, thread_idx] {
        PrefetchThreadFunc(file_name, thread_idx);
      });
    } else {
      // No more file to read.
      ++completed_thread_num;
      if (completed_thread_num == prefetchers_.size()) {
        buffer_->Close();
        break;
      }
    }
  }
  // If users invoke ReInit() when scheduler is running, it will close the
  // 'avaiable_thread_idx_' and prefecther threads have no way to tell scheduler
  // to release their resource. So a check is needed before scheduler ends.
  for (auto& p : prefetchers_) {
    if (p.joinable()) {
      p.join();
    }
  }
  VLOG(5) << "MultiFileReader schedule thread terminates.";
}

void MultiFileReader::PrefetchThreadFunc(std::string file_name,
                                         size_t thread_idx) {
  VLOG(5) << "The prefetch thread of file '" << file_name << "' starts.";
  std::unique_ptr<framework::ReaderBase> reader =
      CreateReaderByFileName(file_name, dims_);
  while (true) {
    std::vector<framework::LoDTensor> ins;
    reader->ReadNext(&ins);
    if (ins.empty()) {
      break;
    }
    try {
      buffer_->Send(std::move(ins));
    } catch (paddle::platform::EnforceNotMet e) {
      VLOG(5) << "WARNING: The buffer channel has been closed. The prefetch "
                 "thread of file '"
              << file_name << "' will terminate.";
      break;
    }
  }

  if (!available_thread_idx_->Send(thread_idx)) {
    VLOG(5) << "WARNING: The available_thread_idx_ channel has been closed. "
               "Fail to send thread_idx.";
  }
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
                      static_cast<int>(shape_concat.size()),
                      "The accumulate of all ranks should be equal to the "
                      "shape concat's length.");
    const auto& file_names = Attr<std::vector<std::string>>("file_names");
    PADDLE_ENFORCE(!file_names.empty(), "No file to be read!");
    const size_t thread_num = Attr<int>("thread_num");
    const size_t buffer_size = Attr<int>("buffer_size");

    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new MultiFileReader(file_names,
                                   RestoreShapes(shape_concat, ranks),
                                   thread_num, buffer_size));
  }
};

class OpenFilesOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<std::vector<std::string>>("file_names", "Files to be read.");
    AddAttr<int>("thread_num", "The maximal concurrent prefetch thread number.")
        .GreaterThan(0);
    AddAttr<int>("buffer_size", "The size of prefetch buffer.").GreaterThan(0);

    AddComment(R"DOC(
      OpenFiles Operator

      An OpenFilesOp creates a MultiFileReader, which is able to
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
