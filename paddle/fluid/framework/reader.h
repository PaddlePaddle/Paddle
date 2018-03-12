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

#pragma once

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor_array.h"

namespace paddle {
namespace framework {

class ReaderBase {
 public:
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;

  virtual void ReInit() = 0;

  virtual bool HasNext() const = 0;

  virtual ~ReaderBase();
};

class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(ReaderBase* reader) : ReaderBase(), reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  void ReInit() override { reader_->ReInit(); }

  bool HasNext() const override { return reader_->HasNext(); }

 protected:
  ReaderBase* reader_;
};

// The ReaderHolder is used as reader' unified wrapper,
// making it easier to access different type reader in Variables.
class ReaderHolder {
 public:
  void Reset(ReaderBase* reader) { reader_.reset(reader); }

  ReaderBase* Get() const { return reader_.get(); }

  void ReadNext(std::vector<LoDTensor>* out) { reader_->ReadNext(out); }
  void ReInit() { reader_->ReInit(); }

  bool HasNext() const { return reader_->HasNext(); }

 private:
  std::unique_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
