/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "PaddleAPI.h"

#include "paddle/parameter/Argument.h"

struct ArgumentsPrivate {
  std::vector<paddle::Argument> outputs;

  inline paddle::Argument& getArg(size_t idx) throw(RangeError) {
    if (idx < outputs.size()) {
      return outputs[idx];
    } else {
      RangeError e;
      throw e;
    }
  }

  template <typename T>
  std::shared_ptr<T>& cast(void* rawPtr) const {
    return *(std::shared_ptr<T>*)(rawPtr);
  }
};

size_t Arguments::getSlotNum() const { return m->outputs.size(); }

Arguments* Arguments::createArguments(size_t slotNum) {
  auto args = new Arguments();
  args->m->outputs.resize(slotNum);
  return args;
}

void Arguments::resize(size_t slotNum) { m->outputs.resize(slotNum); }

Matrix* Arguments::getSlotValue(size_t idx) const throw(RangeError) {
  auto& a = m->getArg(idx);
  return Matrix::createByPaddleMatrixPtr(&a.value);
}

Arguments::Arguments() : m(new ArgumentsPrivate()) {}

Arguments::~Arguments() { delete m; }

Arguments* Arguments::createByPaddleArgumentVector(void* ptr) {
  auto p = (std::vector<paddle::Argument>*)(ptr);
  auto args = new Arguments();
  args->m->outputs = *p;
  return args;
}

IVector* Arguments::getSlotIds(size_t idx) const throw(RangeError) {
  auto& a = m->getArg(idx);
  return IVector::createByPaddleVectorPtr(&a.ids);
}

Matrix* Arguments::getSlotIn(size_t idx) const throw(RangeError) {
  auto& a = m->getArg(idx);
  return Matrix::createByPaddleMatrixPtr(&a.in);
}

void Arguments::setSlotValue(size_t idx, Matrix* mat) throw(RangeError) {
  auto& a = m->getArg(idx);
  a.value = m->cast<paddle::Matrix>(mat->getSharedPtr());
}

void Arguments::setSlotIn(size_t idx, Matrix* mat) throw(RangeError) {
  auto& a = m->getArg(idx);
  a.in = m->cast<paddle::Matrix>(mat->getSharedPtr());
}

void Arguments::setSlotIds(size_t idx, IVector* vec) throw(RangeError) {
  auto& a = m->getArg(idx);
  auto& v = m->cast<paddle::IVector>(vec->getSharedPtr());
  a.ids = v;
}

template <typename T1>
static inline void doCopyFromSafely(std::shared_ptr<T1>& dest,
                                    std::shared_ptr<T1>& src) {
  if (src) {
    if (dest) {
      dest->copyFrom(*src);
    } else {
      dest = src;
    }
  }
}

IVector* Arguments::getSlotSequenceStartPositions(size_t idx) const
    throw(RangeError) {
  auto& a = m->getArg(idx);
  return IVector::createByPaddleVectorPtr(
    &a.sequenceStartPositions->getMutableVector(false));
}

void Arguments::setSlotSequenceStartPositions(size_t idx,
                                              IVector* vec) throw(RangeError) {
  auto& a = m->getArg(idx);
  auto& v = m->cast<paddle::IVector>(vec->getSharedPtr());
  a.sequenceStartPositions = std::make_shared<paddle::ICpuGpuVector>(v);
}

IVector* Arguments::getSlotSequenceDim(size_t idx) const throw(RangeError) {
  auto& a = m->getArg(idx);
  return IVector::createByPaddleVectorPtr(&a.cpuSequenceDims);
}

void Arguments::setSlotSequenceDim(size_t idx, IVector* vec) throw(RangeError) {
  auto& a = m->getArg(idx);
  a.cpuSequenceDims = m->cast<paddle::IVector>(vec->getSharedPtr());
}

int64_t Arguments::getBatchSize(size_t idx) const throw(RangeError) {
  auto& a = m->getArg(idx);
  return a.getBatchSize();
}

void* Arguments::getInternalArgumentsPtr() const { return &m->outputs; }
