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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class ReaderBase {
 public:
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;

  std::vector<std::weak_ptr<ReaderBase>>& GetDecorations() {
    return decorations_;
  }

  virtual std::function<void()> CloseAndGetRestartMethod(bool recursively) = 0;

  void ReInitAllReaders();

  virtual ~ReaderBase();

 protected:
  virtual void Close() {
    PADDLE_ENFORCE(!is_closed_);
    is_closed_ = true;
  }

  virtual void ReStart() {
    PADDLE_ENFORCE(is_closed_);
    is_closed_ = false;
  }

  std::vector<std::weak_ptr<ReaderBase>> decorations_;
  bool is_closed_ = false;
};

class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(const std::shared_ptr<ReaderBase>& reader);

  void ReadNext(std::vector<LoDTensor>* out) final;

  std::function<void()> CloseAndGetRestartMethod(bool recursively) override;

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;

  std::shared_ptr<ReaderBase> reader_;
};

class RootReader : public ReaderBase {
 public:
  RootReader() : ReaderBase() {}

  void ReadNext(std::vector<LoDTensor>* out) override;

  std::function<void()> CloseAndGetRestartMethod(bool recursively) override;

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;
};

class FileReader : public RootReader {
 public:
  explicit FileReader(const std::vector<DDim>& dims);

  void ReadNext(std::vector<LoDTensor>* out) final;

 private:
  std::vector<DDim> dims_;
};

// The ReaderHolder is used as reader' unified wrapper,
// making it easier to access different type reader in Variables.
class ReaderHolder {
 public:
  void Reset(ReaderBase* reader) { reader_.reset(reader); }

  std::shared_ptr<ReaderBase> Get() const { return reader_; }

  void ReadNext(std::vector<LoDTensor>* out) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReadNext(out);
  }

  void ReInitAllReaders() {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReInitAllReaders();
  }

 private:
  std::shared_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
