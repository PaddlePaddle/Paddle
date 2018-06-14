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

// 'Double buffer' means we shall maintain two batches of input data at the same
// time. So the kCacheSize shoul be at least 2.
static constexpr size_t kCacheSize = 3;
// There will be two bacthes out of the channel during training:
// 1. the one waiting to be sent to the channel
// 2. the one just be received from the channel, which is also being used by
// subsequent operators.
// So the channel size should be kChacheSize - 2
static constexpr size_t kChannelSize = 1;  // kCacheSize - 2

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  explicit DoubleBufferReader(
      const std::shared_ptr<ReaderBase>& reader,
      platform::Place target_place = platform::CPUPlace())
      : DecoratedReader(reader), place_(target_place) {
    cpu_tensor_cache_.resize(kCacheSize);
    gpu_tensor_cache_.resize(kCacheSize);
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place_)) {
      for (size_t i = 0; i < kCacheSize; ++i) {
        ctxs_.emplace_back(new platform::CUDADeviceContext(
            boost::get<platform::CUDAPlace>(place_)));
      }
    }
#endif
    StartPrefetcher();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  ~DoubleBufferReader() { EndPrefetcher(); }

 private:
  void StartPrefetcher() {
    channel_ = new reader::BlockingQueue<size_t>(kChannelSize);
    prefetcher_ = std::thread([this] { PrefetchThreadFunc(); });
  }

  void EndPrefetcher() {
    channel_->Close();
    if (prefetcher_.joinable()) {
      prefetcher_.join();
    }
    delete channel_;
    channel_ = nullptr;
  }

  void PrefetchThreadFunc();

  std::thread prefetcher_;
  reader::BlockingQueue<size_t>* channel_;
  platform::Place place_;
  std::vector<std::vector<framework::LoDTensor>> cpu_tensor_cache_;
  std::vector<std::vector<framework::LoDTensor>> gpu_tensor_cache_;
  std::vector<std::unique_ptr<platform::DeviceContext>> ctxs_;
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

    out->Reset(new DoubleBufferReader(underlying_reader.Get(), place));
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

void DoubleBufferReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  size_t cached_tensor_id;
  if (channel_->Receive(&cached_tensor_id)) {
    if (platform::is_gpu_place(place_)) {
      *out = gpu_tensor_cache_[cached_tensor_id];
    } else {
      // CPU place
      *out = cpu_tensor_cache_[cached_tensor_id];
    }
  } else {
    out->clear();
  }
}

void DoubleBufferReader::ReInit() {
  reader_->ReInit();
  EndPrefetcher();
  StartPrefetcher();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  VLOG(5) << "A new prefetch thread starts.";
  size_t cached_tensor_id = 0;
  while (true) {
    auto& cpu_batch = cpu_tensor_cache_[cached_tensor_id];
    reader_->ReadNext(&cpu_batch);
    if (cpu_batch.empty()) {
      // The underlying reader have no next data.
      break;
    }
    if (platform::is_gpu_place(place_)) {
      auto& gpu_batch = gpu_tensor_cache_[cached_tensor_id];
      gpu_batch.resize(cpu_batch.size());
      for (size_t i = 0; i < cpu_batch.size(); ++i) {
        // TODO(fengjiayi): Use asynchronous TensorCopy instead
        framework::TensorCopySync(cpu_batch[i], place_, &gpu_batch[i]);
        gpu_batch[i].set_lod(cpu_batch[i].lod());
      }
    }
    if (!channel_->Send(cached_tensor_id)) {
      VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                 "prefetch thread will terminate.";
      break;
    }
    ++cached_tensor_id;
    cached_tensor_id %= kCacheSize;
  }
  channel_->Close();
  VLOG(5) << "Prefetch thread terminates.";
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
