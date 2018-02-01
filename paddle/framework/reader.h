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

#include "paddle/framework/ddim.h"
#include "paddle/framework/lod_tensor.h"

namespace paddle {
namespace framework {

class ReaderBase {
 public:
  virtual std::vector<LoDTensor> ReadNext() = 0;
  virtual bool HasNext() const = 0;

  virtual DDim shape(size_t idx) const = 0;
  virtual std::vector<DDim> shapes() const = 0;

  virtual ~ReaderBase() {}
};

class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& shapes) : shapes_(shapes) {
    PADDLE_ENFORCE(!shapes_.empty());
  }

  DDim shape(size_t idx) const override;
  std::vector<DDim> shapes() const override { return shapes_; }

 protected:
  std::vector<DDim> shapes_;
};

class ReaderDecorator : public ReaderBase {
 public:
  explicit ReaderDecorator(ReaderBase* reader) : reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  bool HasNext() const override { return reader_->HasNext(); }

  DDim shape(size_t idx) const override { return reader_->shape(idx); }
  std::vector<DDim> shapes() const override { return reader_->shapes(); }

 protected:
  ReaderBase* reader_;
};

// file readers

template <typename T>
class RandomReader : public FileReader {
 public:
  RandomReader(const std::vector<DDim>& shapes, float min, float max)
      : FileReader(shapes), min_(min), max_(max) {
    PADDLE_ENFORCE_LE(min, max,
                      "'min' should be less than or equal to 'max'.(%f vs %f)",
                      min, max);
    unsigned int seed = std::random_device()();
    engine_.seed(seed);
    dist_ = std::uniform_real_distribution<float>(min_, max_);
  }

  std::vector<LoDTensor> ReadNext() override {
    std::vector<LoDTensor> res;
    res.reserve(shapes_.size());
    for (const DDim& shape : shapes_) {
      PADDLE_ENFORCE_GE(
          shape.size(), 2,
          "The rank of input data should be 2 at least.(Now it's %d)",
          shape.size());
      LoDTensor out;
      out.Resize(shape);
      T* data = out.mutable_data<T>(platform::CPUPlace());
      int64_t numel = product(shape);
      for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist_(engine_);
      }
      res.push_back(out);
    }
    return res;
  }

  bool HasNext() const override { return true; }

 private:
  float min_;
  float max_;
  std::minstd_rand engine_;
  std::uniform_real_distribution<float> dist_;
};

// decorators

class ShuffleReader : public ReaderDecorator {
 public:
  ShuffleReader(ReaderBase* reader, int buffer_size)
      : ReaderDecorator(reader), buffer_size_(buffer_size), iteration_pos_(0) {
    buffer_.reserve(buffer_size);
  }

  std::vector<LoDTensor> ReadNext() override;

 private:
  int buffer_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
  size_t iteration_pos_;
};

class BatchReader : public ReaderDecorator {
 public:
  BatchReader(ReaderBase* reader, int batch_size)
      : ReaderDecorator(reader), batch_size_(batch_size) {
    buffer_.reserve(batch_size_);
  }

  std::vector<LoDTensor> ReadNext() override;

 private:
  int batch_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
};

class ReaderHolder {
 public:
  void Reset(ReaderBase* reader) { reader_.reset(reader); }

  ReaderBase* Get() const { return reader_.get(); }

  std::vector<LoDTensor> ReadNext() { return reader_->ReadNext(); }
  bool HasNext() const { return reader_->HasNext(); }

  DDim shape(size_t idx) const { return reader_->shape(idx); }
  std::vector<DDim> shapes() const { return reader_->shapes(); }

 private:
  std::unique_ptr<ReaderBase> reader_;
};

}  // namespace framework
}  // namespace paddle
