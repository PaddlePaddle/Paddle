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
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class DecoratedReader;
class ReaderBase {
 public:
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;

  virtual void ReInit() = 0;

  virtual ~ReaderBase();

  // Return the readers which are the end of decorating chain. Basically
  // they are readers just before read op.
  std::unordered_set<ReaderBase*> GetEndPoints();

 private:
  friend class DecoratedReader;
  // These methods can be only invoked inside DecoratedReader to record the
  // decorating chain.
  void InsertDecoratedReader(
      const std::shared_ptr<DecoratedReader>& decorated_reader);
  // A set of which readers that decorated this reader.
  std::vector<std::weak_ptr<DecoratedReader>> decorated_readers_;
  std::mutex decorated_readers_mtx_;
};

class DecoratedReader : public ReaderBase,
                        public std::enable_shared_from_this<DecoratedReader> {
 public:
  explicit DecoratedReader(const std::shared_ptr<ReaderBase>& reader)
      : ReaderBase(), reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  void ReInit() override { reader_->ReInit(); }

  void RegisterDecorateChain() {
    reader_->InsertDecoratedReader(shared_from_this());
  }

 protected:
  std::shared_ptr<ReaderBase> reader_;
};

class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& dims);

  void ReadNext(std::vector<LoDTensor>* out) override;

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;

 private:
  std::vector<DDim> dims_;
};

// The ReaderHolder is used as reader' unified wrapper,
// making it easier to access different type reader in Variables.
class ReaderHolder {
 public:
  template <typename T>
  void Reset(const std::shared_ptr<T>& reader) {
    auto reader_base = std::dynamic_pointer_cast<ReaderBase>(reader);
    PADDLE_ENFORCE_NOT_NULL(reader_base);
    reader_ = reader_base;
  }

  const std::shared_ptr<ReaderBase>& Get() const { return reader_; }

  void ReadNext(std::vector<LoDTensor>* out) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReadNext(out);
  }
  void ReInit() {
    PADDLE_ENFORCE_NOT_NULL(reader_);
    reader_->ReInit();
  }

  operator const std::shared_ptr<ReaderBase>&() const { return this->reader_; }

 private:
  std::shared_ptr<ReaderBase> reader_;
};

template <typename T, typename... ARGS>
inline std::shared_ptr<DecoratedReader> MakeDecoratedReader(ARGS&&... args) {
  std::shared_ptr<DecoratedReader> reader(new T(std::forward<ARGS>(args)...));
  reader->RegisterDecorateChain();
  return reader;
}

}  // namespace framework
}  // namespace paddle
