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

static constexpr size_t kDoubleBufferSize = 2;

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  explicit DoubleBufferReader(
      ReaderBase* reader, platform::Place target_place = platform::CPUPlace())
      : DecoratedReader(reader), place_(target_place) {
    start_thread();
  }

  void start_thread() {
    buffer_ = framework::MakeChannel<std::vector<framework::LoDTensor>>(
        kDoubleBufferSize);
    std::thread prefetch([this] { PrefetchThreadFunc(); });
    prefetch.detach();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  ~DoubleBufferReader() { buffer_->Close(); }

  bool HasNext() const override;

 private:
  void PrefetchThreadFunc();

  framework::Channel<std::vector<framework::LoDTensor>>* buffer_;
  platform::Place place_;
  mutable std::vector<framework::LoDTensor> local_buffer_;
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

void DoubleBufferReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  out->clear();
  if (local_buffer_.empty()) {
    buffer_->Receive(out);
  } else {
    *out = local_buffer_;
    local_buffer_.clear();
  }
}

void DoubleBufferReader::ReInit() {
  reader_->ReInit();
  buffer_->Close();
  start_thread();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  VLOG(5) << "A new prefetch thread starts.";
  while (reader_->HasNext()) {
    std::vector<framework::LoDTensor> batch;
    reader_->ReadNext(&batch);
    if (platform::is_gpu_place(place_)) {
      std::vector<framework::LoDTensor> gpu_batch;
      gpu_batch.resize(batch.size());
      for (size_t i = 0; i < batch.size(); ++i) {
        framework::TensorCopy(batch[i], place_, &gpu_batch[i]);
        gpu_batch[i].set_lod(batch[i].lod());
      }
    }

    if (!buffer_->Send(&batch)) {
      VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                 "prefetch thread terminates.";
      break;
    }
  }
  buffer_->Close();
}

bool DoubleBufferReader::HasNext() const {
  if (local_buffer_.empty()) {
    bool ok = buffer_->Receive(&local_buffer_);
    return ok;
  } else {
    return true;
  }
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
