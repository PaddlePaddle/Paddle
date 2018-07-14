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

#include "ThreadPool.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {
class BufferedReader : public framework::DecoratedReader {
  using TensorVec = std::vector<framework::LoDTensor>;
  using VecFuture = std::future<TensorVec>;

 public:
  BufferedReader(const std::shared_ptr<framework::ReaderBase>& reader,
                 const platform::Place& place, size_t buffer_size)
      : framework::DecoratedReader(reader),
        thread_pool_(1),
        place_(place),
        buffer_size_(buffer_size) {
    AppendFutureToBatchSize();
  }

  ~BufferedReader() override {
    reader_->Shutdown();
    buffer_.clear();
  }

 private:
  void AppendFutureToBatchSize() {
    while (buffer_.size() < buffer_size_) {
      AppendFuture();
    }
  }

  void AppendFuture() {
    buffer_.emplace_back(thread_pool_.enqueue([this] {
      TensorVec cpu_buffer;
      reader_->ReadNext(&cpu_buffer);
      if (platform::is_gpu_place(place_)) {
        TensorVec gpu_buffer;

        for (size_t i = 0; i < cpu_buffer.size(); ++i) {
          gpu_buffer.emplace_back();
          framework::TensorCopySync(cpu_buffer[i], place_, &gpu_buffer.back());
        }

        cpu_buffer = gpu_buffer;
      }
      return cpu_buffer;
    }));
  }

 protected:
  void ShutdownImpl() override {
    reader_->Shutdown();
    buffer_.clear();
  }
  void StartImpl() override {
    reader_->Start();
    AppendFutureToBatchSize();
  }
  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override {
    std::cerr << "Read" << std::endl;
    PADDLE_ENFORCE_EQ(buffer_.size(), buffer_size_);
    *out = buffer_.front().get();
    buffer_.pop_front();
    AppendFuture();
  }

 private:
  ThreadPool thread_pool_;
  platform::Place place_;
  const size_t buffer_size_;
  std::list<VecFuture> buffer_;
};

class CreateDoubleBufferReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) {
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();

    auto place_str = Attr<std::string>("place");
    platform::Place place;
    if (place_str == "AUTO") {
      place = dev_place;
    } else if (place_str == "CPU") {
      place = platform::CPUPlace();
    } else {
      std::istringstream sin(place_str);
      sin.seekg(std::string("CUDA:").size(), std::ios::beg);
      size_t num;
      sin >> num;
      place = platform::CUDAPlace(static_cast<int>(num));
    }

    out->Reset(framework::MakeDecoratedReader<BufferedReader>(underlying_reader,
                                                              place, 2));
  }
};

class CreateDoubleBufferReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
    AddComment(R"DOC(
      CreateDoubleBufferReader Operator

      A double buffer reader takes another reader as its 'underlying reader'.
      It launches another thread to execute the 'underlying reader' asynchronously,
      which prevents reading process from blocking subsequent training.
    )DOC");
    std::unordered_set<std::string> enum_range;
    constexpr size_t kMaxCUDADevs = 128;
    for (size_t i = 0; i < kMaxCUDADevs; ++i) {
      enum_range.insert(string::Sprintf("CUDA:%d", i));
    }
    enum_range.insert("CPU");
    enum_range.insert("AUTO");
    AddAttr<std::string>("place", "The double buffer place")
        .SetDefault("AUTO")
        .InEnum({enum_range});
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
