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

namespace paddle {
namespace operators {
namespace reader {

class BatchReader : public framework::DecoratedReader {
 public:
  BatchReader(const std::shared_ptr<ReaderBase>& reader, int batch_size)
      : DecoratedReader(reader), batch_size_(batch_size) {
    buffer_.reserve(batch_size_);
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;

 private:
  int batch_size_;
  std::vector<std::vector<framework::LoDTensor>> buffer_;
};

class CreateBatchReaderOp : public framework::OperatorBase {
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
    out->Reset(
        new BatchReader(underlying_reader.Get(), Attr<int>("batch_size")));
  }
};

class CreateBatchReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<int>("batch_size",
                 "How many instances the batch reader yields each time.")
        .GreaterThan(0);
    AddComment(R"DOC(
      CreateBatchReader Operator

      A batch reader takes another reader as its 'underlying reader',
      gathers the underlying reader's outputs and then yields them in batches.
    )DOC");
  }
};

void BatchReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  buffer_.clear();
  buffer_.reserve(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    buffer_.push_back(std::vector<framework::LoDTensor>());
    reader_->ReadNext(&buffer_.back());
    if (buffer_.back().empty()) {
      buffer_.pop_back();
      break;
    }
  }
  // Concat instances
  out->clear();
  if (buffer_.empty()) {
    // if buffer_ is empty, the 'out' will return as an empty vector.
    return;
  }
  int out_num = buffer_[0].size();
  out->reserve(out_num);
  for (int j = 0; j < out_num; ++j) {
    // Merge shape and check date type
    std::type_index batch_type = buffer_[0][j].type();
    framework::DDim batch_shape = buffer_[0][j].dims();
    for (size_t i = 1; i < buffer_.size(); ++i) {
      std::type_index ins_type = buffer_[i][j].type();
      framework::DDim ins_shape = buffer_[i][j].dims();
      PADDLE_ENFORCE_EQ(batch_type, ins_type);
      PADDLE_ENFORCE_EQ(slice_ddim(batch_shape, 1, batch_shape.size()),
                        slice_ddim(ins_shape, 1, ins_shape.size()));
      PADDLE_ENFORCE_GT(ins_shape[0], 0);
      batch_shape[0] += ins_shape[0];
    }

    framework::LoDTensor out_tensor;
    out_tensor.Resize(batch_shape);
    out_tensor.mutable_data(platform::CPUPlace(), batch_type);
    int64_t dst_offset = 0;

    // Merge lod and data
    framework::LoD batch_lod;
    for (size_t i = 0; i < buffer_.size(); ++i) {
      framework::DDim ins_shape = buffer_[i][j].dims();
      framework::LoD ins_lod = buffer_[i][j].lod();
      if (i == 0) {
        batch_lod = ins_lod;
      } else {
        PADDLE_ENFORCE_EQ(batch_lod.size(), ins_lod.size());
        for (size_t level_idx = 0; level_idx < batch_lod.size(); ++level_idx) {
          auto& lod_level = batch_lod[level_idx];
          for (size_t k = 1; k < ins_lod[level_idx].size(); ++k) {
            lod_level.push_back(ins_lod[level_idx][k] + lod_level.back());
          }
        }
      }
      auto dst = out_tensor.Slice(dst_offset, dst_offset + ins_shape[0]);
      TensorCopy(buffer_[i][j], platform::CPUPlace(), &dst);
      dst_offset += ins_shape[0];
    }
    out_tensor.set_lod(batch_lod);
    out->push_back(out_tensor);
  }
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_batch_reader,
                                   ops::CreateBatchReaderOp,
                                   ops::CreateBatchReaderOpMaker);
