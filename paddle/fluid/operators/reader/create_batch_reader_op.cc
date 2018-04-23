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
  BatchReader(ReaderBase* reader, int batch_size)
      : DecoratedReader(reader), batch_size_(batch_size) {}

  LoDTensorListPtr ReadNext() override {
    std::vector<LoDTensorListPtr> batch;

    for (int i = 0; i < batch_size_; ++i) {
      auto ins = reader_->ReadNext();
      if (ins == nullptr) {
        break;
      }
      batch.emplace_back(std::move(ins));
    }
    if (batch.empty()) {
      return nullptr;
    }

    // Concat instances
    size_t out_num = batch[0]->size();
    LoDTensorListPtr out(new LoDTensorListPtr::element_type(out_num));

    //! FIXME(yuyang18): use concat method.
    for (int j = 0; j < out_num; ++j) {
      // Merge shape and check date type
      std::type_index batch_type = batch[0]->at(j).type();
      framework::DDim batch_shape = batch[0]->at(j).dims();
      for (size_t i = 1; i < batch.size(); ++i) {
        std::type_index ins_type = batch[i]->at(j).type();
        framework::DDim ins_shape = batch[i]->at(j).dims();
        PADDLE_ENFORCE_EQ(batch_type, ins_type);
        PADDLE_ENFORCE_EQ(slice_ddim(batch_shape, 1, batch_shape.size()),
                          slice_ddim(ins_shape, 1, ins_shape.size()));
        PADDLE_ENFORCE_GT(ins_shape[0], 0);
        batch_shape[0] += ins_shape[0];
      }

      framework::LoDTensor& out_tensor = out->at(j);
      out_tensor.Resize(batch_shape);
      out_tensor.mutable_data(platform::CPUPlace(), batch_type);
      int64_t dst_offset = 0;

      // Merge lod and data
      framework::LoD batch_lod;
      for (size_t i = 0; i < batch.size(); ++i) {
        framework::DDim ins_shape = batch[i]->at(j).dims();
        framework::LoD ins_lod = batch[i]->at(j).lod();
        if (i == 0) {
          batch_lod = ins_lod;
        } else {
          PADDLE_ENFORCE_EQ(batch_lod.size(), ins_lod.size());
          for (size_t level_idx = 0; level_idx < batch_lod.size();
               ++level_idx) {
            auto& lod_level = batch_lod[level_idx];
            for (size_t k = 1; k < ins_lod[level_idx].size(); ++k) {
              lod_level.push_back(ins_lod[level_idx][k] + lod_level.back());
            }
          }
        }
        auto dst = out_tensor.Slice(dst_offset, dst_offset + ins_shape[0]);
        TensorCopy(batch[i]->at(j), platform::CPUPlace(), &dst);
        dst_offset += ins_shape[0];
      }
      out_tensor.set_lod(batch_lod);
    }
    return out;
  }

 private:
  int batch_size_;
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
    out->Reset(std::unique_ptr<framework::ReaderBase>(
        new BatchReader(underlying_reader.Get(), Attr<int>("batch_size"))));
  }
};

class CreateBatchReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateBatchReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
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

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_batch_reader,
                                   ops::CreateBatchReaderOp,
                                   ops::CreateBatchReaderOpMaker);
