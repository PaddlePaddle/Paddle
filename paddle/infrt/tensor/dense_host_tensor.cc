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

#include "paddle/infrt/tensor/dense_host_tensor.h"

#include <llvm/Support/raw_os_ostream.h>

#include "paddle/infrt/common/buffer.h"

namespace infrt::tensor {

DenseHostTensor::DenseHostTensor(std::initializer_list<int64_t>&& list,
                                 DType dtype)
    : DenseHostTensor(TensorShape(list), dtype) {}

DenseHostTensor::DenseHostTensor(const TensorShape& shape, DType dtype)
    : HostTensor(TensorMetadata{dtype, shape}) {
  CHECK(metadata().IsValid()) << "Tensor construct get invalid metadata";
  buffer_.reset(new infrt::Buffer(infrt::common::DefaultHostTarget()));
  buffer_->ResizeLazy(dtype.GetHostSize() * shape.GetNumElements());
}

const TensorShape& DenseHostTensor::shape() const { return metadata().shape; }
TensorShape* DenseHostTensor::mutable_shape() {
  return &mutable_metadata()->shape;
}

void DenseHostTensor::Init(const std::vector<int64_t>& shape, DType dtype) {
  auto shape_array = llvm::ArrayRef<int64_t>(shape.data(), shape.size());
  auto metadata = TensorMetadata(dtype, shape_array);
  setTensorMetadata(metadata);
  buffer_.reset(new infrt::Buffer(infrt::common::DefaultHostTarget()));
  buffer_->ResizeLazy(dtype.GetHostSize() * metadata.shape.GetNumElements());
}

const infrt::Buffer* DenseHostTensor::buffer() const { return buffer_.get(); }

template <typename T>
void DisplayArray(std::ostream& os, T* data, int num_elements) {
  for (int i = 0; i < num_elements - 1; i++) os << data[i] << ", ";
  if (num_elements > 0) os << data[num_elements - 1];
}

std::ostream& operator<<(std::ostream& os, const DenseHostTensor& instance) {
  CHECK(instance.metadata().IsValid())
      << "Cann't print tensor with invalid metadata";
  llvm::raw_os_ostream oos(os);
  oos << "tensor: ";
  oos << "shape=";
  oos << instance.shape();
  oos << ", values=[";

  oos.flush();

  if (instance.metadata().dtype == GetDType<float>()) {
    auto* data = reinterpret_cast<float*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<double>()) {
    auto* data = reinterpret_cast<double*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<int32_t>()) {
    auto* data = reinterpret_cast<int32_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else if (instance.metadata().dtype == GetDType<int64_t>()) {
    auto* data = reinterpret_cast<int64_t*>(instance.buffer()->data()->memory);
    DisplayArray(os, data, instance.shape().GetNumElements());
  } else {
    LOG(FATAL) << "Not supported dtype [" << instance.metadata().dtype.name()
               << " " << static_cast<int>(instance.metadata().dtype.kind())
               << "] in print";
  }

  os << "]";

  return os;
}

DenseHostTensor::~DenseHostTensor() {}

void* DenseHostTensor::raw_data() const { return buffer_->data()->memory; }

}  // namespace infrt::tensor
