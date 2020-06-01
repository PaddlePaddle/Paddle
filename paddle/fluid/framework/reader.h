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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class ReaderBase {
 public:
  explicit ReaderBase(const std::vector<DDim>& shapes,
                      const std::vector<proto::VarType::Type>& var_types,
                      const std::vector<bool>& need_check_feed)
      : shapes_(shapes),
        var_types_(var_types),
        need_check_feed_(need_check_feed) {
    PADDLE_ENFORCE_EQ(
        shapes_.size(), need_check_feed_.size(),
        platform::errors::InvalidArgument(
            "Construct ReaderBase with mismatched sizes of shapes "
            "and need_check_feed"));
    PADDLE_ENFORCE_EQ(
        var_types_.size(), need_check_feed_.size(),
        platform::errors::InvalidArgument(
            "Construct ReaderBase with mismatched sizes of var_types "
            "and need_check_feed"));
  }

  virtual void ReadNext(std::vector<LoDTensor>* out);

  virtual void Shutdown();

  virtual void Start();

  // Return the readers which are the end of decorating chain. Basically
  // they are readers just before read op.
  std::unordered_set<ReaderBase*> GetEndPoints();

  // Returns the shapes of the fed variables
  const std::vector<DDim>& Shapes() const { return shapes_; }

  // Returns the dtypes of the fed variables
  const std::vector<proto::VarType::Type>& VarTypes() const {
    return var_types_;
  }

  // For Backward compatibility, old fluid.layers.data doesn't check shape.
  // This function returns whether you have the check shape for this Reader.
  const std::vector<bool>& NeedCheckFeed() const { return need_check_feed_; }

  virtual ~ReaderBase();

 protected:
  virtual void ReadNextImpl(std::vector<LoDTensor>* out) {}

  virtual void ShutdownImpl() {}

  virtual void StartImpl() {}

  enum ReaderStatus { kRunning, kStopped };

  ReaderStatus status_{kRunning};

  mutable std::mutex mu_;

  // The shapes of the fed variables.
  std::vector<DDim> shapes_;

  // The dtypes of the fed variables.
  std::vector<proto::VarType::Type> var_types_;

  // Whether to check the shape and dtype of fed variables.
  // For Backward compatibility, variables created by old API fluid.layers.data
  // doesn't check shape but fluid.data checks.
  std::vector<bool> need_check_feed_;

 private:
  friend class DecoratedReader;
  // These methods can be only invoked inside DecoratedReader to record the
  // decorating chain.
  void InsertDecoratedReader(
      const std::shared_ptr<ReaderBase>& decorated_reader);
  // A set of which readers that decorated this reader.
  std::vector<std::weak_ptr<ReaderBase>> decorated_readers_;
};

class DecoratedReader : public ReaderBase,
                        public std::enable_shared_from_this<DecoratedReader> {
 public:
  explicit DecoratedReader(const std::shared_ptr<ReaderBase>& reader)
      : ReaderBase(reader->Shapes(), reader->VarTypes(),
                   reader->NeedCheckFeed()),
        reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(
        reader_,
        platform::errors::InvalidArgument(
            "The underlying reader of DecoratedReader should not be null"));
  }

  void RegisterDecorateChain() {
    reader_->InsertDecoratedReader(shared_from_this());
  }

  ~DecoratedReader();

  const std::shared_ptr<ReaderBase>& UnderlyingReader() const {
    return reader_;
  }

 protected:
  void ShutdownImpl() override {
    VLOG(1) << "ShutdownImpl";
    reader_->Shutdown();
  }

  void StartImpl() override { reader_->Start(); }

  std::shared_ptr<ReaderBase> reader_;
};

// FileReader is just a conceptual class.
class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& shapes,
                      const std::vector<proto::VarType::Type>& var_types,
                      const std::vector<bool>& need_check_feed)
      : ReaderBase(shapes, var_types, need_check_feed) {}
};

// The ReaderHolder is used as reader' unified wrapper,
// making it easier to access different type reader in Variables.
class ReaderHolder {
 public:
  template <typename T>
  void Reset(const std::shared_ptr<T>& reader) {
    auto reader_base = std::dynamic_pointer_cast<ReaderBase>(reader);
    PADDLE_ENFORCE_NOT_NULL(
        reader_base,
        platform::errors::InvalidArgument(
            "The underlying reader of ReaderHolder should not be null"));
    reader_ = reader_base;
  }

  ~ReaderHolder() { VLOG(1) << "~ReaderHolder"; }

  const std::shared_ptr<ReaderBase>& Get() const { return reader_; }

  void ReadNext(std::vector<LoDTensor>* out) {
    PADDLE_ENFORCE_NOT_NULL(
        reader_,
        platform::errors::InvalidArgument(
            "The underlying reader of ReaderHolder should not be null"));
    reader_->ReadNext(out);
  }

  void ResetAll() {
    VLOG(1) << "ResetAll";
    auto end_readers = reader_->GetEndPoints();
    for (auto* reader : end_readers) {
      reader->Shutdown();
    }
    for (auto* reader : end_readers) {
      reader->Start();
    }
  }

  void Shutdown() {
    VLOG(1) << "Shutdown";
    PADDLE_ENFORCE_NOT_NULL(
        reader_,
        platform::errors::InvalidArgument(
            "The underlying reader of ReaderHolder should not be null"));
    reader_->Shutdown();
  }

  void Start() {
    VLOG(1) << "start";
    PADDLE_ENFORCE_NOT_NULL(
        reader_,
        platform::errors::InvalidArgument(
            "The underlying reader of ReaderHolder should not be null"));
    reader_->Start();
  }

  const std::vector<DDim>& Shapes() const { return reader_->Shapes(); }

  const std::vector<proto::VarType::Type>& VarTypes() const {
    return reader_->VarTypes();
  }

  const std::vector<bool>& NeedCheckFeed() const {
    return reader_->NeedCheckFeed();
  }

  void Clear() { reader_.reset(); }

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
