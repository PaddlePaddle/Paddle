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

#include <cmath>
#include <stdexcept>
#include <thread>  // NOLINT
#include "ThreadPool.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class IReaderContainer {
 public:
  virtual ~IReaderContainer() {}
  virtual void AppendReader(
      std::unique_ptr<framework::ReaderBase>&& readers) = 0;
  virtual void Stop() = 0;
  virtual void Start() = 0;
  virtual void ReadNext(std::vector<framework::LoDTensor>* out) = 0;
};

class OrderedReaderContainer : public IReaderContainer {
 public:
  void AppendReader(std::unique_ptr<framework::ReaderBase>&& reader) override {
    pending_.emplace(std::move(reader));
  }

  void Stop() override {
    while (!pending_.empty()) {
      MoveFrontPendingToDone();
    }
  }

  void Start() override { std::swap(done_, pending_); }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    if (!pending_.empty()) {
      pending_.front()->ReadNext(out);
      if (out->empty()) {
        MoveFrontPendingToDone();
        ReadNext(out);
      }
    } else {
      out->clear();
    }
  }

 private:
  void MoveFrontPendingToDone() {
    pending_.front()->Shutdown();
    pending_.front()->Start();
    done_.emplace(move(pending_.front()));
    pending_.pop();
  }

  std::queue<std::unique_ptr<framework::ReaderBase>> pending_;
  std::queue<std::unique_ptr<framework::ReaderBase>> done_;
};

class PreemptiveReaderContainer : public IReaderContainer {
  using ReaderList = std::list<std::unique_ptr<framework::ReaderBase>>;

  struct FutureItem {
    std::vector<framework::LoDTensor> data_;
    ReaderList::iterator reader_it_;
    std::exception_ptr exception_;
  };

  using FutureList = std::list<std::future<FutureItem>>;

 public:
  explicit PreemptiveReaderContainer(size_t thread_num) : pool_(thread_num) {}

  void Stop() override {
    if (!pending_.empty()) {
      for (auto& reader : pending_) {
        reader->Shutdown();
      }
      for (auto& fu : futures_) {
        fu.wait();
      }
      futures_.clear();
      for (auto& reader : pending_) {
        reader->Start();
        done_.emplace_back(std::move(reader));
      }
      pending_.clear();
      bool timeout;
      complete_queue_.PopAll(1000, &timeout);
      PADDLE_ENFORCE(!timeout);
    }
  }

  void Start() override {
    for (auto& reader : done_) {
      AppendReader(std::move(reader));
    }
    done_.clear();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    if (!pending_.empty()) {
      auto future_it = complete_queue_.Pop();
      FutureItem item = future_it->get();
      if (item.exception_) {
        for (auto it = futures_.begin(); it != futures_.end(); ++it) {
          if (it != future_it) {
            it->wait();  // Wait all other threads complete.
          }
        }
        std::rethrow_exception(item.exception_);

      } else if (item.data_.empty()) {  // reader done.
        done_.emplace_back(std::move(*item.reader_it_));
        pending_.erase(item.reader_it_);
        futures_.erase(future_it);
        ReadNext(out);
      } else {
        *out = item.data_;
        // continue read async
        ReadAsync(item.reader_it_, &future_it);
      }
    } else {
      out->clear();
    }
  }

 private:
  void AppendReader(std::unique_ptr<framework::ReaderBase>&& reader) override {
    pending_.emplace_back(std::move(reader));
    auto reader_it = pending_.end();
    --reader_it;

    futures_.emplace_back();
    auto future_it = futures_.end();
    --future_it;

    ReadAsync(reader_it, &future_it);
  }

  void ReadAsync(const ReaderList::iterator& reader_it,
                 FutureList::iterator* future_it_ptr) {
    auto& future_it = *future_it_ptr;
    *future_it = pool_.enqueue([reader_it, future_it, this] {
      try {
        FutureItem item;
        item.reader_it_ = reader_it;
        (*reader_it)->ReadNext(&item.data_);
        if (item.data_.empty()) {
          (*reader_it)->Shutdown();
          (*reader_it)->Start();
        }
        complete_queue_.Push(future_it);
        return item;
      } catch (...) {
        FutureItem item;
        item.exception_ = std::current_exception();
        complete_queue_.Push(future_it);
        return item;
      }
    });
  }

  FutureList futures_;
  ThreadPool pool_;
  framework::BlockingQueue<FutureList::iterator> complete_queue_;
  std::list<std::unique_ptr<framework::ReaderBase>> pending_;
  std::list<std::unique_ptr<framework::ReaderBase>> done_;
};

class MultiFileReader : public framework::ReaderBase {
 public:
  MultiFileReader(const std::vector<std::string>& file_names,
                  std::unique_ptr<IReaderContainer>&& container)
      : container_(std::move(container)) {
    for (auto& fn : file_names) {
      container_->AppendReader(CreateReaderByFileName(fn));
    }
  }

  ~MultiFileReader() { container_->Stop(); }

 protected:
  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override {
    container_->ReadNext(out);
  }
  void ShutdownImpl() override { container_->Stop(); }
  void StartImpl() override { container_->Start(); }

 private:
  std::unique_ptr<IReaderContainer> container_;
};

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
    bool is_test = Attr<bool>("is_test");

    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    std::unique_ptr<IReaderContainer> container;

    if (is_test) {
      container.reset(new OrderedReaderContainer());
    } else {
      container.reset(new PreemptiveReaderContainer(
          static_cast<size_t>(Attr<int>("thread_num"))));
    }

    std::shared_ptr<framework::ReaderBase> reader(
        new MultiFileReader(file_names, std::move(container)));
    auto buffer_size = Attr<int>("buffer_size");
    if (buffer_size > 1) {
      reader = framework::MakeDecoratedReader<BufferedReader>(
          reader, platform::CPUPlace(), buffer_size);
    }
    out->Reset(reader);
  }
};

class OpenFilesOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<std::vector<std::string>>("file_names", "Files to be read.");
    AddAttr<bool>("is_test", "Used for testing data.").SetDefault(false);

    AddComment(R"DOC(
      OpenFiles Operator

      An OpenFilesOp creates a MultiFileReader, which is able to
      read data multi-threaded from multiple files.
    )DOC");
    AddAttr<int>("thread_num",
                 "The maximal concurrent prefetch thread number. Used only "
                 "when is_test = False");
    AddAttr<int>("buffer_size", "The reading buffer of these files.")
        .GreaterThan(0);
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(open_files, reader::OpenFilesOp,
                              reader::OpenFilesOpMaker);
