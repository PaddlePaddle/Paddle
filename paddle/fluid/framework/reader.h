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

// file readers

template <typename T>
class RandomDataGenerator : public FileReader {
 public:
  RandomDataGenerator(const std::vector<DDim>& shapes, float min, float max)
      : FileReader(shapes), min_(min), max_(max) {
    PADDLE_ENFORCE_LE(
        min, max, "'min' shouldn't be greater than 'max'.(%f vs %f)", min, max);
    unsigned int seed = std::random_device()();
    engine_.seed(seed);
    dist_ = std::uniform_real_distribution<float>(min_, max_);
  }

  void ReadNext(std::vector<LoDTensor>* out) override {
    out->clear();
    out->reserve(shapes_.size());
    for (const DDim& shape : shapes_) {
      PADDLE_ENFORCE_GE(
          shape.size(), 2,
          "The rank of reader's output data should be 2 at least.(Now it's %d)",
          shape.size());
      LoDTensor out_tensor;
      out_tensor.Resize(shape);
      T* data = out_tensor.mutable_data<T>(platform::CPUPlace());
      int64_t numel = product(shape);
      for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist_(engine_);
      }
      out->push_back(out_tensor);
    }
  }

  bool HasNext() const override { return true; }

  void ReInit() override { return; }

 private:
  float min_;
  float max_;
  std::minstd_rand engine_;
  std::uniform_real_distribution<float> dist_;
};

// decorated readers

class ShuffleReader : public DecoratedReader {
 public:
  ShuffleReader(ReaderBase* reader, int buffer_size)
      : DecoratedReader(reader), buffer_size_(buffer_size), iteration_pos_(0) {
    buffer_.reserve(buffer_size);
  }

  void ReadNext(std::vector<LoDTensor>* out) override;

 private:
  int buffer_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
  size_t iteration_pos_;
};

class BatchReader : public DecoratedReader {
 public:
  BatchReader(ReaderBase* reader, int batch_size)
      : DecoratedReader(reader), batch_size_(batch_size) {
    buffer_.reserve(batch_size_);
  }

  void ReadNext(std::vector<LoDTensor>* out) override;

 private:
  int batch_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
};

// The ReaderHolder is used as readers' unified wrapper,
// making it easier to access different type readers in Variables.
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
