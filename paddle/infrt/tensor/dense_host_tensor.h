// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>

#include "paddle/infrt/tensor/tensor_metadata.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt {
class Buffer;
}  // namespace infrt

namespace infrt {
namespace tensor {

enum class DeviceKind {
  kCPU = 0,
};

class Tensor {
 public:
  virtual bool IsHostTensor() const = 0;
  virtual ~Tensor() = default;

  const TensorMetadata& metadata() const { return metadata_; }
  TensorMetadata* mutable_metadata() { return &metadata_; }

 protected:
  Tensor() = default;
  void setTensorMetadata(TensorMetadata& metadata) {  // NOLINT
    metadata_ = metadata;
  }
  explicit Tensor(const TensorMetadata& metadata) : metadata_(metadata) {}
  explicit Tensor(TensorMetadata&& metadata) : metadata_(std::move(metadata)) {}

 private:
  TensorMetadata metadata_;
};

class HostTensor : public Tensor {
 public:
  bool IsHostTensor() const override { return true; }

  virtual ~HostTensor() {}

 protected:
  HostTensor() = default;
  explicit HostTensor(const TensorMetadata& metadata) : Tensor(metadata) {}
  explicit HostTensor(TensorMetadata&& metadata)
      : Tensor(std::move(metadata)) {}
};

// TODO(Superjomn) Replace the hlir/framework/Tensor with this.
/**
 * DenseTensor is a dense tensor, it holds a TensorShape and a buffer.
 */
class DenseHostTensor : public HostTensor {
 public:
  DenseHostTensor() = default;
  DenseHostTensor(const TensorShape& shape, DType dtype);
  DenseHostTensor(std::initializer_list<int64_t>&& list, DType dtype);

  void Init(const std::vector<int64_t>& shape, DType dtype);
  const TensorShape& shape() const;
  TensorShape* mutable_shape();

  DType dtype() const;

  const Buffer* buffer() const;

  void* raw_data() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const DenseHostTensor& instance);

  virtual ~DenseHostTensor();

 private:
  // TODO(Superjomn) Discard the dependency of the Buffer in infrtcore or create
  // a general buffer in common.
  std::shared_ptr<Buffer> buffer_;
};

}  // namespace tensor
}  // namespace infrt
