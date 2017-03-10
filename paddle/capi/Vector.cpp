/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

using paddle::capi::cast;

extern "C" {

PD_IVector PDIVecCreateNone() { return new paddle::capi::CIVector(); }

PD_IVector PDIVectorCreate(int* array, uint64_t size, bool copy, bool useGPU) {
  auto ptr = new paddle::capi::CIVector();
  if (copy) {
    ptr->vec = paddle::IVector::create(size, useGPU);
    ptr->vec->copyFrom(array, size);
  } else {
    ptr->vec = paddle::IVector::create(array, size, useGPU);
  }
  return ptr;
}

PD_Error PDIVecDestroy(PD_IVector ivec) {
  if (ivec == nullptr) return kPD_NULLPTR;
  delete cast<paddle::capi::CIVector>(ivec);
  return kPD_NO_ERROR;
}

PD_Error PDIVectorGet(PD_IVector ivec, int** buffer) {
  if (ivec == nullptr || buffer == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  *buffer = v->vec->getData();
  return kPD_NO_ERROR;
}

PD_Error PDIVectorResize(PD_IVector ivec, uint64_t size) {
  if (ivec == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  v->vec->resize(size);
  return kPD_NO_ERROR;
}

PD_Error PDIVectorGetSize(PD_IVector ivec, uint64_t* size) {
  if (ivec == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  *size = v->vec->getSize();
  return kPD_NO_ERROR;
}
}
