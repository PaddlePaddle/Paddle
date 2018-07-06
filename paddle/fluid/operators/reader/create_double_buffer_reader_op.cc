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

#include <functional>
#include <thread>  // NOLINT
#include "ThreadPool.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {
template <size_t BufferSize>
class BufferedReader : public framework::DecoratedReader {
 public:
  explicit BufferedReader(const std::shared_ptr<ReaderBase>& reader,
                          platform::Place target_place = platform::CPUPlace())
      : DecoratedReader(reader), pool_(1), place_(target_place) {
    this->ReInit();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    *out = buff_.front().get();
    buff_.pop_front();
    EnqueueJob();
  }

  void ReInit() override {
    buff_.clear();
    for (size_t i = 0; i < BufferSize; ++i) {
      EnqueueJob();
    }
  }

 private:
  void EnqueueJob() {
    buff_.emplace_back(pool_.enqueue(
        std::bind(std::mem_fn(&BufferedReader::ReadUnderlyingReader), this)));
  }

  std::vector<framework::LoDTensor> ReadUnderlyingReader() {
    std::vector<framework::LoDTensor> res;
    reader_->ReadNext(&res);
    if (platform::is_gpu_place(place_)) {
      std::vector<framework::LoDTensor> gpu_res;
      for (auto& cpu : res) {
        gpu_res.emplace_back();
        auto& gpu = gpu_res.back();
        framework::TensorCopySync(cpu, place_, &gpu);
      }
      res = gpu_res;
    }
    return res;
  }

  std::list<std::future<std::vector<framework::LoDTensor>>> buff_;
  ::ThreadPool pool_;
  platform::Place place_;
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

    out->Reset(new BufferedReader<2>(underlying_reader.Get(), place));
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
