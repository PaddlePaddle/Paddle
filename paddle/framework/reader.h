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

class Reader {
 public:
  Reader() {}
  explicit Reader(const std::vector<DDim>& shapes) : shapes_(shapes) {}

  virtual std::vector<LoDTensor> ReadNext() = 0;
  virtual bool HasNext() const = 0;

  virtual DDim shape(size_t idx) const;
  virtual std::vector<DDim> shapes() const { return shapes_; }

  virtual ~Reader() {}

 private:
  // set private to prevent directly access in decorators
  // a decorator should access its underlying reader_'s shape, not its own.
  std::vector<DDim> shapes_;
};

// file readers

template <typename T>
class RandomReader : public Reader {
 public:
  RandomReader(const std::vector<DDim>& shapes, float min, float max)
      : Reader(shapes), min_(min), max_(max) {
    PADDLE_ENFORCE_LE(min, max,
                      "'min' should be less than or equal to 'max'.(%f vs %f)",
                      min, max);
  }

  std::vector<LoDTensor> ReadNext() override {
    std::minstd_rand engine;
    unsigned int seed = std::random_device()();
    engine.seed(seed);
    std::uniform_real_distribution<float> dist(min_, max_);

    std::vector<LoDTensor> res;
    res.reserve(shapes().size());
    for (const DDim& shape : shapes()) {
      PADDLE_ENFORCE_GE(
          shape.size(), 2,
          "The rank of input data should be 2 at least.(Now it's %d)",
          shape.size());
      LoDTensor out;
      out.Resize(shape);
      T* data = out.mutable_data<T>(platform::CPUPlace());
      int64_t numel = product(shape);
      for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
      }
      res.push_back(out);
    }
    return res;
  }

  bool HasNext() const override { return true; }

 private:
  float min_;
  float max_;
};

// decorators

class ShuffleReader : public Reader {
 public:
  ShuffleReader(Reader* reader, int buffer_size)
      : reader_(reader), buffer_size_(buffer_size), iteration_pos_(0) {
    buffer_.reserve(buffer_size);
  }
  std::vector<LoDTensor> ReadNext() override;
  bool HasNext() const override { return reader_->HasNext(); }

  DDim shape(size_t idx) const override { return reader_->shape(idx); }
  std::vector<DDim> shapes() const override { return reader_->shapes(); }

 private:
  Reader* reader_;
  int buffer_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
  size_t iteration_pos_;
};

class BatchReader : public Reader {
 public:
  BatchReader(Reader* reader, int batch_size)
      : reader_(reader), batch_size_(batch_size) {}
  std::vector<LoDTensor> ReadNext() override;
  bool HasNext() const override { return reader_->HasNext(); };

  DDim shape(size_t idx) const override { return reader_->shape(idx); }
  std::vector<DDim> shapes() const override { return reader_->shapes(); }

 private:
  Reader* reader_;
  int batch_size_;
  std::vector<std::vector<LoDTensor>> buffer_;
};

}  // namespace framework
}  // namespace paddle
