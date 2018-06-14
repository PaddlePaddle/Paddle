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

#include "PaddleAPI.h"

#include "paddle/math/Vector.h"

#include <cstring>

struct IVectorPrivate {
  paddle::IVectorPtr vec;
};

IVector::IVector() : m(new IVectorPrivate()) {}

IVector* IVector::createZero(size_t sz, bool useGpu) {
  auto v = new IVector();
  v->m->vec = paddle::IVector::create(sz, useGpu);
  v->m->vec->zeroMem();
  return v;
}

IVector* IVector::create(const std::vector<int>& data, bool useGpu) {
  auto v = new IVector();
  v->m->vec = paddle::IVector::create(data.size(), useGpu);
  v->m->vec->copyFrom(data.data(), data.size());
  return v;
}

IVector* IVector::createVectorFromNumpy(int* data,
                                        int dim,
                                        bool copy,
                                        bool useGpu) throw(UnsupportError) {
  if (useGpu) {
    /// if use gpu only copy=true is supported
    if (!copy) {
      throw UnsupportError("Gpu mode only supports copy=True");
    }
    return IVector::createGpuVectorFromNumpy(data, dim);
  } else {
    return IVector::createCpuVectorFromNumpy(data, dim, copy);
  }
}

IVector* IVector::createCpuVectorFromNumpy(int* data, int dim, bool copy) {
  auto v = new IVector();
  if (copy) {
    v->m->vec = paddle::IVector::create(dim, false);
    v->m->vec->copyFrom(data, dim);
  } else {
    v->m->vec = paddle::IVector::create(data, dim, false);
  }
  return v;
}

IVector* IVector::createGpuVectorFromNumpy(int* data, int dim) {
  auto v = new IVector();
  v->m->vec = paddle::IVector::create(dim, true);
  v->m->vec->copyFrom(data, dim);
  return v;
}

bool IVector::isGpu() const {
  return dynamic_cast<paddle::GpuIVector*>(m->vec.get()) != nullptr;
}

IntArray IVector::getData() const {
  if (this->isGpu()) {
    int* src = m->vec->getData();
    size_t len = m->vec->getSize();
    int* dest = new int[len];
    hl_memcpy_device2host(dest, src, len * sizeof(int));
    return IntArray(dest, len, true);
  } else {
    return IntArray(m->vec->getData(), m->vec->getSize());
  }
}

int& IVector::operator[](const size_t idx) throw(RangeError, UnsupportError) {
  if (this->isGpu()) {
    UnsupportError e;
    throw e;
  } else {
    if (idx >= m->vec->getSize()) {
      RangeError e;
      throw e;
    }
  }
  return m->vec->getData()[idx];
}

const int& IVector::operator[](const size_t idx) const
    throw(RangeError, UnsupportError) {
  return (*const_cast<IVector*>(this))[idx];
}

IVector* IVector::createByPaddleVectorPtr(void* ptr) {
  auto* p = (paddle::IVectorPtr*)ptr;
  if ((*p) != nullptr) {
    IVector* vec = new IVector();
    vec->m->vec = *p;
    return vec;
  } else {
    return nullptr;
  }
}

IVector::~IVector() { delete m; }

void* IVector::getSharedPtr() const { return &m->vec; }

size_t IVector::getSize() const { return m->vec->getSize(); }

void IVector::toNumpyArrayInplace(int** data, int* dim1) throw(UnsupportError) {
  auto v = std::dynamic_pointer_cast<paddle::CpuIVector>(m->vec);
  if (v) {
    *data = v->getData();
    *dim1 = v->getSize();
  } else {
    throw UnsupportError();
  }
}

void IVector::copyToNumpyArray(int** view_m_data, int* dim1) {
  *dim1 = m->vec->getSize();
  *view_m_data = new int[*dim1];
  if (auto cpuVec = dynamic_cast<paddle::CpuIVector*>(m->vec.get())) {
    std::memcpy(*view_m_data, cpuVec->getData(), sizeof(int) * (*dim1));
  } else if (auto gpuVec = dynamic_cast<paddle::GpuIVector*>(m->vec.get())) {
    hl_memcpy_device2host(
        *view_m_data, gpuVec->getData(), sizeof(int) * (*dim1));
  } else {
    LOG(INFO) << "Unexpected situation";
  }
}

void IVector::copyFromNumpyArray(int* data, int dim) {
  m->vec->resize(dim);
  m->vec->copyFrom(data, dim);
}

struct VectorPrivate {
  paddle::VectorPtr vec;

  void safeAccessData(const size_t idx,
                      const std::function<void(float&)>& func) const
      throw(RangeError, UnsupportError) {
    auto cpuVec = std::dynamic_pointer_cast<const paddle::CpuVector>(vec);
    if (cpuVec != nullptr) {
      if (idx < vec->getSize()) {
        func(vec->getData()[idx]);
      } else {
        throw RangeError();
      }
    } else {
      throw UnsupportError();
    }
  }
};

Vector::Vector() : m(new VectorPrivate()) {}

Vector::~Vector() { delete m; }

Vector* Vector::createZero(size_t sz, bool useGpu) {
  auto retVec = new Vector();
  retVec->m->vec = paddle::Vector::create(sz, useGpu);
  retVec->m->vec->zero();
  return retVec;
}

Vector* Vector::create(const std::vector<float>& data, bool useGpu) {
  auto retVec = new Vector();
  retVec->m->vec = paddle::Vector::create(data.size(), useGpu);
  retVec->m->vec->copyFrom(data.data(), data.size());
  return retVec;
}

Vector* Vector::createByPaddleVectorPtr(void* ptr) {
  auto& v = *(paddle::VectorPtr*)(ptr);
  if (v == nullptr) {
    return nullptr;
  } else {
    auto retVec = new Vector();
    retVec->m->vec = v;
    return retVec;
  }
}

Vector* Vector::createVectorFromNumpy(float* data,
                                      int dim,
                                      bool copy,
                                      bool useGpu) throw(UnsupportError) {
  if (useGpu) {
    /// if use gpu only copy=True is supported
    if (!copy) {
      throw UnsupportError("Gpu mode only supports copy=True");
    }
    return Vector::createGpuVectorFromNumpy(data, dim);
  } else {
    return Vector::createCpuVectorFromNumpy(data, dim, copy);
  }
}

Vector* Vector::createCpuVectorFromNumpy(float* data, int dim, bool copy) {
  CHECK_GT(dim, 0);
  auto retVec = new Vector();
  if (copy) {
    retVec->m->vec = paddle::Vector::create((size_t)dim, false);
    retVec->m->vec->copyFrom(data, dim);
  } else {
    retVec->m->vec = paddle::Vector::create(data, (size_t)dim, false);
  }
  return retVec;
}

Vector* Vector::createGpuVectorFromNumpy(float* data, int dim) {
  CHECK_GT(dim, 0);
  auto retVec = new Vector();
  retVec->m->vec = paddle::Vector::create((size_t)dim, true);
  retVec->m->vec->copyFrom(data, (size_t)dim);
  return retVec;
}

void Vector::toNumpyArrayInplace(float** view_data,
                                 int* dim1) throw(UnsupportError) {
  auto v = std::dynamic_pointer_cast<paddle::CpuVector>(m->vec);
  if (v != nullptr) {
    *view_data = v->getData();
    *dim1 = (int)v->getSize();
  } else {
    throw UnsupportError();
  }
}

void Vector::copyToNumpyArray(float** view_m_data, int* dim1) {
  *dim1 = m->vec->getSize();
  *view_m_data = new float[*dim1];
  if (auto cpuVec = dynamic_cast<paddle::CpuVector*>(m->vec.get())) {
    std::memcpy(*view_m_data, cpuVec->getData(), sizeof(float) * (*dim1));
  } else if (auto gpuVec = dynamic_cast<paddle::GpuVector*>(m->vec.get())) {
    hl_memcpy_device2host(
        *view_m_data, gpuVec->getData(), sizeof(float) * (*dim1));
  } else {
    LOG(INFO) << "Unexpected situation";
  }
}

void Vector::copyFromNumpyArray(float* data, int dim) {
  m->vec->resize(dim);
  m->vec->copyFrom(data, dim);
}

FloatArray Vector::getData() const {
  if (this->isGpu()) {
    float* src = m->vec->getData();
    size_t len = m->vec->getSize();
    float* dest = new float[len];
    hl_memcpy_device2host(dest, src, len * sizeof(float));
    FloatArray ret_val(dest, len);
    ret_val.needFree = true;
    return ret_val;
  } else {
    FloatArray ret_val(m->vec->getData(), m->vec->getSize());
    return ret_val;
  }
}

void Vector::copyFrom(Vector* src) throw(RangeError) {
  if (src->m->vec->getSize() != m->vec->getSize()) {
    throw RangeError();
  }
  m->vec->copyFrom(*src->m->vec);
}

bool Vector::isGpu() const {
  return std::dynamic_pointer_cast<paddle::GpuVector>(m->vec) != nullptr;
}

float Vector::get(const size_t idx) const throw(RangeError, UnsupportError) {
  float r;
  m->safeAccessData(idx, [&](float& o) { r = o; });
  return r;
}

void Vector::set(const size_t idx, float val) throw(RangeError,
                                                    UnsupportError) {
  m->safeAccessData(idx, [&](float& o) { o = val; });
}

size_t Vector::getSize() const { return m->vec->getSize(); }

void* Vector::getSharedPtr() { return &m->vec; }
