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
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace framework {

static constexpr size_t kDoubleBufferSize = 3;

class ReaderBase {
 public:
  explicit ReaderBase(const std::vector<DDim>& shapes) : shapes_(shapes) {
    PADDLE_ENFORCE(!shapes_.empty());
  }
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;
  virtual bool HasNext() const = 0;

  virtual void ReInit() = 0;

  DDim shape(size_t idx) const;
  std::vector<DDim> shapes() const { return shapes_; }
  void set_shapes(const std::vector<DDim>& shapes) { shapes_ = shapes; }

  virtual ~ReaderBase() {}

 protected:
  std::vector<DDim> shapes_;
};

class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& shapes) : ReaderBase(shapes) {}
};

class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(ReaderBase* reader)
      : ReaderBase(reader->shapes()), reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  bool HasNext() const override { return reader_->HasNext(); }

  void ReInit() override { reader_->ReInit(); }

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
  bool HasNext() const { return reader_->HasNext(); }
  void ReInit() { reader_->ReInit(); }

  DDim shape(size_t idx) const { return reader_->shape(idx); }
  std::vector<DDim> shapes() const { return reader_->shapes(); }
  void set_shapes(const std::vector<DDim>& shapes) {
    reader_->set_shapes(shapes);
  }

 private:
  std::unique_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
