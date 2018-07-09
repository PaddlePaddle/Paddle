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

enum ReaderStatus { kRunning, kStopped };

class ReaderBase {
 public:
  void ReadNext(std::vector<LoDTensor>* out);

  void Shutdown();

  void Start();

  virtual ~ReaderBase();

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;

  virtual void ShutdownImpl() = 0;

  virtual void StartImpl() = 0;

  ReaderStatus status_{kRunning};
};

class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(const std::shared_ptr<ReaderBase>& reader)
      : ReaderBase(), reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

 protected:
  void ShutdownImpl() override { reader_->Shutdown(); }

  void StartImpl() override { reader_->Start(); }

  std::shared_ptr<ReaderBase> reader_;
};

class FileReader : public ReaderBase {
 public:
  FileReader() : ReaderBase() {}

 protected:
  void ShutdownImpl() override {}

  void StartImpl() override {}
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

  void ResetAll() {
    // TODO(fengjiayi): The interface of reseting all.
  }

  void Shutdown() {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->Shutdown();
  }

  void Start() {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->Start();
  }

 private:
  std::shared_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
