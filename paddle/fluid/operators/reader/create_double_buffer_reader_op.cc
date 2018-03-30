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

#include <thread>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

static constexpr size_t kChannelSize = 2;
static constexpr size_t kCacheSize = 4;  // kChannelSize + 2

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  struct Item {
    Item() : ctx_(nullptr) {}

    std::vector<framework::LoDTensor> payloads_;
    platform::DeviceContext* ctx_;
  };

  explicit DoubleBufferReader(
      ReaderBase* reader, platform::Place target_place = platform::CPUPlace())
      : DecoratedReader(reader), place_(target_place) {
#ifdef PADDLE_WITH_CUDA
    for (size_t i = 0; i < kChannelSize + 2; ++i) {
      if (platform::is_gpu_place(place_)) {
        ctxs_.emplace_back(new platform::CUDADeviceContext(
            boost::get<platform::CUDAPlace>(place_)));
      }
    }
#endif
    StartPrefetcher();
  }

  bool HasNext() const override;
  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  void StartPrefetcher() {
    buffer_ = framework::MakeChannel<Item>(kChannelSize);
    prefetcher_ = std::thread([this] { PrefetchThreadFunc(); });
  }

  void EndPrefetcher() {
    buffer_->Close();
    if (prefecther_.joinable()) {
      prefetcher_.join();
    }
    delete buffer_;
    buffer_ = nullptr;
  }

  ~DoubleBufferReader() { EndPrefetcher(); }

 private:
  void PrefetchThreadFunc();

  std::thread prefetcher_;
  framework::Channel<Item>* buffer_;
  platform::Place place_;
  std::vector<std::unique_ptr<platform::DeviceContext>> ctxs_;
  mutable Item local_buffer_;
};

class CreateDoubleBufferReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();

    auto place_str = Attr<std::string>("place");
    platform::Place place;
    if (place_str == "CPU") {
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
 public:
  CreateDoubleBufferReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
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
    AddAttr<std::string>("place", "The double buffer place, default is CPU")
        .SetDefault("CPU")
        .InEnum({enum_range});
  }
};

bool DoubleBufferReader::HasNext() const {
  if (local_buffer_.payloads_.empty()) {
    bool ok = buffer_->Receive(&local_buffer_);
    return ok;
  } else {
    return true;
  }
}

void DoubleBufferReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  if (!HasNext()) {
    PADDLE_THROW("There is no next data!");
  }

  *out = local_buffer_.payloads_;
  local_buffer_.payloads_.clear();
  if (local_buffer_.ctx_) {
    local_buffer_.ctx_->Wait();
  }
}

void DoubleBufferReader::ReInit() {
  reader_->ReInit();
  EndPrefetcher();
  StartPrefetcher();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  VLOG(5) << "A new prefetch thread starts.";
  std::vector<std::vector<framework::LoDTensor>> cpu_tensor_cache(kCacheSize);
  std::vector<std::vector<framework::LoDTensor>> gpu_tensor_cache(kCacheSize);
  size_t cached_tensor_id = 0;

  while (reader_->HasNext()) {
    Item batch;
    auto& cpu_batch = cpu_tensor_cache[cached_tensor_id];
    reader_->ReadNext(&cpu_batch);
    if (platform::is_gpu_place(place_)) {
      auto& gpu_batch = gpu_tensor_cache[cached_tensor_id];
      auto* gpu_ctx = ctxs_[cached_tensor_id].get();
      gpu_batch.resize(cpu_batch.size());
      for (size_t i = 0; i < cpu_batch.size(); ++i) {
        framework::TensorCopy(cpu_batch[i], place_, *gpu_ctx, &gpu_batch[i]);
        gpu_batch[i].set_lod(batch.payloads_[i].lod());
      }
      batch.payload_ = gpu_batch;
      batch.ctx_ = gpu_ctx;
    } else {
      // CPUPlace
      batch.payload_ = cpu_batch;
    }
    ++cached_tensor_id;
    cached_tensor_id %= kCacheSize;

    try {
      buffer_->Send(&batch);
    } catch (paddle::platform::EnforceNotMet e) {
      VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                 "prefetch thread will terminate.";
      break;
    }
  }
  buffer_->Close();
  VLOG(5) << "Prefetch thread terminates.";
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
