// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda_fp16.h>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/phi/core/enforce.h"

namespace paddle::platform {

// Some trt base classes lack of the destructor.
// We use a assisted class to fix this.
struct DeleteHelper {
 protected:
  virtual ~DeleteHelper() {}
};

template <typename T>
inline void SerializeValue(void** buffer, T const& value);

template <typename T>
inline void DeserializeValue(void const** buffer,
                             size_t* buffer_size,
                             T* value);

namespace details {

template <typename T, class Enable = void>
struct Serializer {};

template <typename T>
struct Serializer<
    T,
    typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_enum<T>::value || std::is_pod<T>::value ||
                            std::is_same<T, half>::value>::type> {
  static size_t SerializedSize(T const& value) { return sizeof(T); }

  static void Serialize(void** buffer, T const& value) {
    std::memcpy(*buffer, &value, sizeof(T));
    reinterpret_cast<char*&>(*buffer) += sizeof(T);
  }

  static void Deserialize(void const** buffer, size_t* buffer_size, T* value) {
    assert(*buffer_size >= sizeof(T));
    std::memcpy(value, *buffer, sizeof(T));
    reinterpret_cast<char const*&>(*buffer) += sizeof(T);
    *buffer_size -= sizeof(T);
  }
};

template <>
struct Serializer<const char*> {
  static size_t SerializedSize(const char* value) { return strlen(value) + 1; }

  static void Serialize(void** buffer, const char* value) {
    std::strcpy(static_cast<char*>(*buffer), value);  // NOLINT
    reinterpret_cast<char*&>(*buffer) += strlen(value) + 1;
  }

  static void Deserialize(void const** buffer,
                          size_t* buffer_size,
                          const char** value) {
    *value = static_cast<char const*>(*buffer);
    size_t data_size = strnlen(*value, *buffer_size) + 1;
    assert(*buffer_size >= data_size);
    reinterpret_cast<char const*&>(*buffer) += data_size;
    *buffer_size -= data_size;
  }
};

template <typename T>
struct Serializer<
    std::vector<T>,
    typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_enum<T>::value || std::is_pod<T>::value ||
                            std::is_same<T, half>::value>::type> {
  static size_t SerializedSize(std::vector<T> const& value) {
    return sizeof(value.size()) + value.size() * sizeof(T);
  }

  static void Serialize(void** buffer, std::vector<T> const& value) {
    SerializeValue(buffer, value.size());
    size_t nbyte = value.size() * sizeof(T);
    std::memcpy(*buffer, value.data(), nbyte);
    reinterpret_cast<char*&>(*buffer) += nbyte;
  }

  static void Deserialize(void const** buffer,
                          size_t* buffer_size,
                          std::vector<T>* value) {
    size_t size;
    DeserializeValue(buffer, buffer_size, &size);
    value->resize(size);
    size_t nbyte = value->size() * sizeof(T);
    PADDLE_ENFORCE_GE(*buffer_size,
                      nbyte,
                      common::errors::InvalidArgument(
                          "Insufficient data in buffer, expect contains %d "
                          "byte, but actually only contains %d byte.",
                          *buffer_size,
                          nbyte));
    std::memcpy(value->data(), *buffer, nbyte);
    reinterpret_cast<char const*&>(*buffer) += nbyte;
    *buffer_size -= nbyte;
  }
};

}  // namespace details

template <typename T>
inline size_t SerializedSize(T const& value) {
  return details::Serializer<T>::SerializedSize(value);
}

template <typename T>
inline void SerializeValue(void** buffer, T const& value) {
  return details::Serializer<T>::Serialize(buffer, value);
}

template <typename T>
inline void DeserializeValue(void const** buffer,
                             size_t* buffer_size,
                             T* value) {
  return details::Serializer<T>::Deserialize(buffer, buffer_size, value);
}

template <typename T>
inline void SerializeCudaPointer(void** buffer, T* value, int size) {
  cudaMemcpy((*buffer), value, size * sizeof(T), cudaMemcpyDeviceToHost);
  reinterpret_cast<char*&>(*buffer) += size * sizeof(T);
}

}  // namespace paddle::platform
