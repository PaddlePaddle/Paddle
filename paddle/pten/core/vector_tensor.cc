/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdlib.h>

#include "paddle/pten/core/vector_tensor.h"

namespace pten {

VectorTensor::VectorTensor(const std::vector<int64_t>& vec)
    : data_type_(DataType::INT64), size_(vec.size()) {
  data_ = malloc(size_ * paddle::experimental::SizeOf(data_type_));
  auto* data = static_cast<int64_t*>(data_);
  for (auto i = 0; i < size_; i++) {
    data[i] = vec[i];
  }
}

VectorTensor::VectorTensor(const int64_t* date_value, int64_t n)
    : data_type_(DataType::INT64), size_(n) {
  data_ = malloc(size_ * paddle::experimental::SizeOf(data_type_));
  auto* data = static_cast<int64_t*>(data_);
  for (auto i = 0; i < size_; i++) {
    data[i] = date_value[i];
  }
}

VectorTensor::VectorTensor(const int32_t* date_value, int64_t n)
    : data_type_(DataType::INT32), size_(n) {
  data_ = malloc(size_ * paddle::experimental::SizeOf(data_type_));
  auto* data = static_cast<int32_t*>(data_);
  for (auto i = 0; i < size_; i++) {
    data[i] = date_value[i];
  }
}

VectorTensor::VectorTensor(const DenseTensor& dense_tensor)
    : data_type_(dense_tensor.data_type()), size_(dense_tensor.numel()) {
  PADDLE_ENFORCE_EQ(TransToPtenBackend(dense_tensor.place()),
                    Backend::CPU,
                    paddle::platform::errors::InvalidArgument(
                        "The VectorTensor only supports DenseTensor on CPU, "
                        "but now DenseTensor is on %s.",
                        TransToPtenBackend(dense_tensor.place())));
  PADDLE_ENFORCE_EQ(
      dense_tensor.dims().size(),
      1,
      paddle::platform::errors::InvalidArgument(
          "The VectorTensor only supports DenseTensor with 1 dimension, "
          "but now DenseTensor has %d dimensions.",
          dense_tensor.dims().size()));
  data_ = malloc(size_ * paddle::experimental::SizeOf(data_type_));
  // TODO(zhangyunfei) replace switch-case with data_type_visit
  switch (data_type_) {
    case DataType::INT32:
      AssignData<int32_t>(dense_tensor.data<int32_t>());
      break;
    case DataType::INT64:
      AssignData<int64_t>(dense_tensor.data<int64_t>());
      break;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Data type error. Currently, The data type of VectorTensor "
          "only supports int32 and int64, but now received %s.",
          data_type_));
  }
}

VectorTensor::VectorTensor(const std::vector<DenseTensor>& vec)
    : size_(vec.size()) {
  if (size_ == 0) {
    data_type_ = DataType::INT64;
    return;
  }

  data_type_ = vec[0].data_type();
  switch (data_type_) {
    case DataType::INT32: {
      auto* data = static_cast<int32_t*>(data_);
      for (auto i = 0; i < size_; i++) {
        PADDLE_ENFORCE_EQ(
            vec[i].data_type(),
            data_type_,
            paddle::platform::errors::InvalidArgument(
                "The data_type of tensors in the list isn't consistent."
                "the first tensor is %s, but %dth tensor is %s.",
                data_type_,
                i,
                vec[i].data_type()));
        data[i] = *vec[i].data<int32_t>();
      }
      break;
    }
    case DataType::INT64: {
      auto* data = static_cast<int64_t*>(data_);
      for (auto i = 0; i < size_; i++) {
        PADDLE_ENFORCE_EQ(
            vec[i].data_type(),
            data_type_,
            paddle::platform::errors::InvalidArgument(
                "The data_type of tensors in the list isn't consistent."
                "the first tensor is %s, but %dth tensor is %s.",
                data_type_,
                i,
                vec[i].data_type()));
        data[i] = *vec[i].data<int64_t>();
      }
      break;
    }
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Data type error. Currently, The data type of VectorTensor "
          "only supports int32 and int64, but now received %s.",
          data_type_));
  }
}

VectorTensor::VectorTensor(const VectorTensor& other)
    : data_type_(other.data_type_), size_(other.size_) {
  auto bytes = size_ * paddle::experimental::SizeOf(data_type_);
  data_ = malloc(bytes);
  memcpy(data_, other.data_, bytes);
}

VectorTensor::VectorTensor(VectorTensor&& other)
    : data_type_(other.data_type_), size_(other.size_), data_(other.data_) {
  other.data_ = nullptr;
}

VectorTensor::~VectorTensor() {
  free(data_);
  data_ = nullptr;
}

template <typename T>
const T* VectorTensor::data() const {
  PADDLE_ENFORCE(
      (data_type_ == paddle::experimental::CppTypeToDataType<T>::Type()),
      paddle::platform::errors::PreconditionNotMet(
          "The type of data we are trying to retrieve does not match the "
          "type of data currently contained in the container."));
  return static_cast<const T*>(data_);
}

#define DATA_MEMBER_FUNC_INSTANTIATION(dtype) \
  template const dtype* VectorTensor::data() const;

DATA_MEMBER_FUNC_INSTANTIATION(int32_t);
DATA_MEMBER_FUNC_INSTANTIATION(int64_t);

#undef DATA_MEMBER_FUNC_INSTANTIATION

DDim GetDimFromVectorTensor(const VectorTensor& vector_tensor) {
  if (vector_tensor.numel() == 0) {
    return paddle::framework::make_ddim({0});
  }
  switch (vector_tensor.data_type()) {
    case DataType::INT32:
      return DDim(vector_tensor.data<int32_t>(), vector_tensor.numel());
      break;
    case DataType::INT64:
      return DDim(vector_tensor.data<int64_t>(), vector_tensor.numel());
      break;
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Data type error. The type of data we are trying to retrieve "
          "as a dim must be int32 or int64, but now received %s.",
          vector_tensor.data_type()));
  }
}

}  // namespace pten
