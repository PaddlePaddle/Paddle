/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "capi_private.h"
#include "vector.h"

using paddle::capi::cast;

extern "C" {

paddle_ivector paddle_ivector_create_none() {
  return new paddle::capi::CIVector();
}

paddle_ivector paddle_ivector_create(int* array,
                                     uint64_t size,
                                     bool copy,
                                     bool useGPU) {
  auto ptr = new paddle::capi::CIVector();
  if (copy) {
    ptr->vec = paddle::IVector::create(size, useGPU);
    ptr->vec->copyFrom(array, size);
  } else {
    ptr->vec = paddle::IVector::create(array, size, useGPU);
  }
  return ptr;
}

paddle_error paddle_ivector_destroy(paddle_ivector ivec) {
  if (ivec == nullptr) return kPD_NULLPTR;
  delete cast<paddle::capi::CIVector>(ivec);
  return kPD_NO_ERROR;
}

paddle_error paddle_ivector_get(paddle_ivector ivec, int** buffer) {
  if (ivec == nullptr || buffer == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  *buffer = v->vec->getData();
  return kPD_NO_ERROR;
}

paddle_error paddle_ivector_resize(paddle_ivector ivec, uint64_t size) {
  if (ivec == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  v->vec->resize(size);
  return kPD_NO_ERROR;
}

paddle_error paddle_ivector_get_size(paddle_ivector ivec, uint64_t* size) {
  if (ivec == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  *size = v->vec->getSize();
  return kPD_NO_ERROR;
}
}
