/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/extended_tensor.h"

namespace phi {

int64_t ExtendedTensor::numel() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `numel` method."));
}

const DDim& ExtendedTensor::dims() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `dims` method."));
}

const Place& ExtendedTensor::place() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `place` method."));
}

DataType ExtendedTensor::dtype() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `dtype` method."));
}

DataLayout ExtendedTensor::layout() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `layout` method."));
}

bool ExtendedTensor::valid() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `valid` method."));
}

bool ExtendedTensor::initialized() const {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `initialized` method."));
}

void* ExtendedTensor::AllocateFrom(Allocator* allocator,
                                   DataType dtype,
                                   size_t requested_size,
                                   bool fake_alloc) {
  PADDLE_THROW(phi::errors::Unavailable(
      "ExtendedTensor does not support `AllocateFrom` method."));
}

}  // namespace phi
