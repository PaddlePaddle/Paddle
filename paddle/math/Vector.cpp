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

#include "Vector.h"
#include "paddle/utils/Util.h"

#include <memory>
#include "Matrix.h"
#include "hl_gpu.h"
#include "hl_table_apply.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Thread.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(size_t size, bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuVectorT<T>>(size);
  } else {
    return std::make_shared<CpuVectorT<T>>(size);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::createParallelVector(
    size_t size, bool useGpu, SyncThreadPool* pool) {
  if (!useGpu && FLAGS_trainer_count > 1 && FLAGS_enable_parallel_vector &&
      size >= (size_t)FLAGS_enable_parallel_vector) {
    return std::make_shared<ParallelCpuVectorT<T>>(
        size, pool ? pool : getGlobalSyncThreadPool());
  } else {
    return create(size, useGpu);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(T* data,
                                               size_t size,
                                               bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuVectorT<T>>(size, data);
  } else {
    return std::make_shared<CpuVectorT<T>>(size, data);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(size_t size,
                                               MemoryHandlePtr memoryHandle,
                                               size_t offset) {
  if (auto cpuMemHandle =
          std::dynamic_pointer_cast<CpuMemoryHandle>(memoryHandle)) {
    return std::make_shared<CpuVectorT<T>>(size, cpuMemHandle, offset);
  } else if (auto gpuMemHandle =
                 std::dynamic_pointer_cast<GpuMemoryHandle>(memoryHandle)) {
    return std::make_shared<GpuVectorT<T>>(size, gpuMemHandle, offset);
  } else {
    LOG(FATAL) << "Wrong";
    return NULL;
  }
}

template <>
MatrixPtr VectorT<real>::toOneHotSparseMatrix(size_t idRange, bool useGpu) {
  LOG(FATAL) << "Wrong for real vector";
  return nullptr;
}

template <>
MatrixPtr VectorT<int>::toOneHotSparseMatrix(size_t idRange, bool useGpu) {
  size_t height = getSize();
  size_t width = idRange;
  MatrixPtr mat = Matrix::createSparseMatrix(
      height, idRange, height, NO_VALUE, SPARSE_CSR, false, useGpu);

  CpuIVector cpuIds(height);
  cpuIds.copyFrom(*this);
  int* idData = cpuIds.getData();

  for (decltype(height) i = 0; i < height; i++) {
    const unsigned int id = idData[i];
    CHECK_LT(id, width);
    mat->setRow(i, 1, &id, nullptr);
  }
  return mat;
}

template <class T>
GpuVectorT<T>::GpuVectorT(size_t size)
    : VectorT<T>(size,
                 std::make_shared<GpuMemoryHandle>(sizeof(T) * size),
                 0, /* offset = 0 */
                 true /* useGpu = true */) {}

template <class T>
T GpuVectorT<T>::getElement(size_t i) const {
  T elem = 0;
  hl_memcpy_device2host(&elem, const_cast<T*>(&this->getData()[i]), sizeof(T));
  return elem;
}
template <class T>
void GpuVectorT<T>::setElement(size_t i, const T& value) {
  hl_memcpy_host2device(&this->getData()[i], const_cast<T*>(&value), sizeof(T));
}

template <class T>
T* GpuVectorT<T>::getPoint(const uint64_t beginPos) {
  LOG(FATAL) << "Not implemented" << beginPos;
  return NULL;
}

template <>
int GpuVectorT<int>::getAbsSum() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
int GpuVectorT<int>::getSum() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
real GpuVectorT<real>::getAbsSum() {
  real* A = this->getData();
  real sum = 0;
  hl_vector_abs_sum(A, &sum, this->getSize());
  return sum;
}

template <>
real GpuVectorT<real>::getSum() {
  real* A = this->getData();
  real sum = 0;
  hl_vector_sum(A, &sum, this->getSize());
  return sum;
}

template <>
int GpuVectorT<int>::getMax() {
  CpuIVector cpuIVec = CpuIVector(this->getSize());
  copyTo(&cpuIVec);
  return cpuIVec.getMax();
}

template <>
int GpuVectorT<int>::getAbsMax() {
  CpuIVector cpuIVec = CpuIVector(this->getSize());
  copyTo(&cpuIVec);
  return cpuIVec.getAbsMax();
}

template <class T>
void GpuVectorT<T>::isEqualTo(const VectorT<T>& b, const T& value) {
  BaseMatrixT<T>::isEqualTo((BaseMatrixT<T>&)b, value);
}

template <class T>
void GpuVectorT<T>::selectFrom(const VectorT<T>& src, const VectorT<int>& ids) {
#ifndef PADDLE_ONLY_CPU
  hl_vector_select_from<T>(this->getData(),
                           this->getSize(),
                           src.getData(),
                           src.getSize(),
                           ids.getData(),
                           ids.getSize());
#endif
}

template <class Func>
real gpuRowFunc(Func f, GpuVector& v) {
  static ThreadLocal<std::unique_ptr<CpuVectorT<real>>> local;
  if (!*local) {
    (*local).reset(new CpuVector(1));
  }
  real* A = v.getData();
  f(A, (*local)->getData(), 1, v.getSize());
  return (*local)->getData()[0];
}

template <>
real GpuVectorT<real>::getMax() {
  return gpuRowFunc(hl_matrix_row_max, *this);
}

template <>
real GpuVectorT<real>::getAbsMax() {
  return std::max(gpuRowFunc(hl_matrix_row_max, *this),
                  -gpuRowFunc(hl_matrix_row_min, *this));
}

template <>
int GpuVectorT<int>::getMin() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
real GpuVectorT<real>::getMin() {
  return gpuRowFunc(hl_matrix_row_min, *this);
}

template <class T>
T GpuVectorT<T>::get(size_t pos) {
  T val = (T)0;
  hl_memcpy_device2host((void*)&val, (void*)(this->getData() + pos), sizeof(T));
  return val;
}

template <class T>
void GpuVectorT<T>::histogram(std::ostream& os, int type) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::zeroMem() {
  BaseMatrixT<T>::zero();
}

template <class T>
void GpuVectorT<T>::reset(const T& value) {
  BaseMatrixT<T>::assign(value);
}

template <class T>
void GpuVectorT<T>::fillSequence() {
  LOG(FATAL) << "not implemented";
}

template <class T>
void GpuVectorT<T>::copyFrom(const VectorT<T>& src) {
  src.copyTo(this);
}

template <class T>
void GpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  CHECK_EQ(src.getSize(), this->getSize());
  hl_memcpy_async((void*)this->getData(),
                  (void*)src.getData(),
                  sizeof(T) * this->getSize(),
                  stream);
}

template <class T>
void GpuVectorT<T>::copyFrom(const T* gpuSrc, size_t size) {
  CHECK(gpuSrc != NULL);
  CHECK_LE(size, this->size_);

  hl_memcpy((void*)this->getData(), (void*)gpuSrc, sizeof(T) * size);
}

template <class T>
void GpuVectorT<T>::copyFrom(const T* gpuSrc, size_t size, hl_stream_t stream) {
  CHECK(gpuSrc != NULL);
  CHECK_LE(size, this->size_);

  hl_memcpy_async(
      (void*)this->getData(), (void*)gpuSrc, sizeof(T) * size, stream);
}

template <class T>
void GpuVectorT<T>::copyTo(CpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());

  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(T) * this->getSize());
}

template <class T>
void GpuVectorT<T>::copyTo(GpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());

  hl_memcpy_device2device((void*)dest->getData(),
                          (void*)this->getData(),
                          sizeof(T) * this->getSize());
}

template <>
void GpuVectorT<int>::rand() {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<int>::print(std::ostream& os, size_t num) const {
  IVectorPtr dest = IVector::create(this->size_, false);
  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(int) * this->getSize());
  dest->print(os, num);
}

template <>
void GpuVectorT<real>::print(std::ostream& os, size_t num) const {
  VectorPtr dest = Vector::create(this->size_, false);
  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(int) * this->getSize());
  dest->print(os, num);
}

template <>
void GpuVectorT<int>::printOneElement(std::ostream& os, size_t idx) const {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<real>::printOneElement(std::ostream& os, size_t idx) const {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<int>::rand() {
  LOG(FATAL) << "Not implemented";
}
template <>
void GpuVectorT<real>::rand(size_t classNum) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::rand(size_t classNum) {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<real>::rand() {
  VectorPtr cPtr = Vector::create(this->size_, false);
  cPtr->rand();

  hl_memcpy_host2device(data_, cPtr->getData(), this->size_ * sizeof(real));
}

template <>
void GpuVectorT<int>::rand(size_t classNum) {
  IVectorPtr cPtr = IVector::create(this->size_, false);
  cPtr->rand(classNum);

  hl_memcpy_host2device(data_, cPtr->getData(), this->size_ * sizeof(int));
}

template <>
void CpuVectorT<int>::rand(size_t classNum) {
  size_t size = this->getSize();
  int* data = this->getData();
  for (size_t i = 0; i < size; i++) {
    data[i] =
        std::min(classNum - 1,
                 size_t(::rand() * (1. / ((double)RAND_MAX + 1)) * classNum));
  }
}

template <>
void CpuVectorT<real>::rand() {
  size_t size = this->getSize();
  real* data = this->getData();
  for (size_t i = 0; i < size; i++) {
    data[i] = ::rand() * (1. / (double)RAND_MAX);
    // data[ii] = ((temp > RAND_MAX/2)? 1 : -1) *
    // sqrt( abs((temp-RAND_MAX/2))/(double(RAND_MAX))/2048 );
  }
}

template <class T>
void CpuVectorT<T>::randnorm(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void CpuVectorT<T>::uniform(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::randnorm(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::uniform(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::randnorm(real mean, real std) {
  size_t size = this->getSize();
  real* data = this->getData();
  unsigned int* seed = ThreadLocalRand::getSeed();
  auto rand1 = [&]() { return (1. + ::rand_r(seed)) * (1. / (1. + RAND_MAX)); };
  for (size_t i = 0; i < size - 1; i += 2) {
    real r1 = rand1();
    r1 = std::sqrt(-2 * std::log(r1));
    real r2 = rand1();
    data[i] = mean + std * r1 * cos(2 * M_PI * r2);
    data[i + 1] = mean + std * r1 * sin(2 * M_PI * r2);
  }
  real r1 = rand1();
  r1 = std::sqrt(-2 * std::log(r1));
  real r2 = rand1();
  data[size - 1] = mean + std * r1 * cos(2 * M_PI * r2);
}

template <>
void CpuVectorT<real>::uniform(real left, real right) {
  size_t size = this->getSize();
  real* data = this->getData();
  real range = right - left;
  unsigned int* seed = ThreadLocalRand::getSeed();
  auto rand1 = [&]() { return ::rand_r(seed) * (1. / (1. + RAND_MAX)); };
  for (size_t i = 0; i < size; ++i) {
    data[i] = rand1() * range + left;
  }
}

template <>
void GpuVectorT<real>::randnorm(real mean, real std) {
  CpuVector cpuVec = CpuVector(this->getSize());
  cpuVec.randnorm(mean, std);

  hl_memcpy_host2device(
      data_, cpuVec.getData(), this->getSize() * sizeof(real));
}

template <>
void GpuVectorT<real>::uniform(real left, real right) {
  CpuVector cpuVec = CpuVector(this->getSize());
  cpuVec.uniform(left, right);

  hl_memcpy_host2device(
      data_, cpuVec.getData(), this->getSize() * sizeof(real));
}

template <class T>
CpuVectorT<T>::CpuVectorT(size_t size)
    : VectorT<T>(size,
                 std::make_shared<CpuMemoryHandle>(sizeof(T) * size),
                 0, /* offset = 0 */
                 false /* useGpu = false */) {}

template <class T>
CpuVectorT<T>::CpuVectorT(const VectorT<T>& src)
    : VectorT<T>(src.getSize(),
                 src.getMemoryHandle(),
                 0, /* offset = 0 */
                 false /* useGpu = false */) {
  if (typeid(*this->memoryHandle_.get()) != typeid(CpuMemoryHandle)) {
    this->memoryHandle_ =
        std::make_shared<CpuMemoryHandle>(sizeof(T) * this->getSize());
    this->data_ = reinterpret_cast<T*>(this->memoryHandle_->getBuf());
  }
  src.copyTo(this);
}

template <class T>
T CpuVectorT<T>::getAbsSum() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (A[i] > 0) ? A[i] : -A[i];
  }
  return sum;
}

// cannot use above version, due to precision issue of float
template <>
real CpuVectorT<real>::getAbsSum() {
  const real* A = this->getData();
  size_t size = this->getSize();
  double sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (A[i] > 0) ? A[i] : -A[i];
  }
  return sum;
}

template <class T>
T CpuVectorT<T>::getSum() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

template <>
real CpuVectorT<real>::getSum() {
  const real* A = this->getData();
  size_t size = this->getSize();
  double sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

template <class T>
T CpuVectorT<T>::get(size_t pos) {
  return this->getData()[pos];
}

template <class T>
T CpuVectorT<T>::getMax() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = A[0];
  for (size_t i = 1; i < size; i++) {
    if (res < A[i]) res = A[i];
  }
  return res;
}

template <class T>
T CpuVectorT<T>::getAbsMax() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = std::abs(A[0]);
  for (size_t i = 1; i < size; i++) {
    if (res < std::abs(A[i])) res = std::abs(A[i]);
  }
  return res;
}

template <class T>
T CpuVectorT<T>::getMin() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = A[0];
  for (size_t i = 1; i < size; i++) {
    if (res > A[i]) res = A[i];
  }
  return res;
}

template <class T>
void CpuVectorT<T>::isEqualTo(const VectorT<T>& b, const T& value) {
  size_t size = this->getSize();
  CHECK_EQ(b.getSize(), size);

  const T* B = b.getData();
  T* A = this->getData();
  for (size_t i = 0; i < size; i++) {
    A[i] = (B[i] == value);
  }
}

template <class T>
void CpuVectorT<T>::selectFrom(const VectorT<T>& src, const VectorT<int>& ids) {
  size_t size = this->getSize();
  CHECK_EQ(ids.getSize(), size);

  const int* indices = ids.getData();
  const T* B = src.getData();
  T* A = this->getData();
  for (size_t i = 0; i < size; i++) {
    int index = indices[i];
    CHECK_LT(index, (int)src.getSize());
    A[i] = B[index];
  }
}

static int getSignAndExponentOfFloat(float a) {
  uint32_t* pa = reinterpret_cast<uint32_t*>(&a);
  return *pa >> 23;
}

template <class T>
void CpuVectorT<T>::histogram(std::ostream& os, int type) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::histogram(std::ostream& os, int type) {
  int counters[512];
  memset(counters, 0, sizeof(counters));
  int counterZero = 0;

  const real* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    if (A[i] == 0.0f) {
      ++counterZero;
    } else {
      ++counters[getSignAndExponentOfFloat(A[i])];
    }
  }

  int64_t sum = 0;
  float sizeNonZero = size - counterZero;
  os << "zero:" << counterZero;
  for (int i = 0; i < 256; i++) {
    int counter = counters[i];
    if (counter) {
      os << " 2^" << i - 127 << ":" << counter / sizeNonZero * 100 << "%";
      sum += counter * (i - 127);
    }
  }
  for (int i = 0; i < 256; i++) {
    int counter = counters[i + 256];
    if (counter) {
      os << " -2^" << i - 127 << ":" << counter / sizeNonZero * 100 << "%";
      sum += counter * (i - 127);
    }
  }
  os << ", nonzero_exponent_avg=" << sum / sizeNonZero;
}

template <class T>
void CpuVectorT<T>::zeroMem() {
  memset(this->getData(), 0, sizeof(T) * this->getSize());
}

template <class T>
void CpuVectorT<T>::reset(const T& value) {
  T* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    A[i] = value;
  }
}

template <class T>
void CpuVectorT<T>::fillSequence() {
  T* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    A[i] = i;
  }
}

template <class T>
void CpuVectorT<T>::copyFrom(const VectorT<T>& src) {
  src.copyTo(this);
}

template <class T>
void CpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  if (typeid(src) == typeid(GpuVectorT<T>)) {
    hl_memcpy_async((void*)this->getData(),
                    (void*)src.getData(),
                    sizeof(T) * this->getSize(),
                    stream);
  } else {
    src.copyTo(this);
  }
}

template <class T>
void CpuVectorT<T>::copyFrom(const T* hostSrc, size_t size) {
  CHECK(hostSrc != NULL);
  CHECK_LE(size, this->size_);
  memcpy(this->data_, hostSrc, sizeof(T) * size);
}

template <class T>
void CpuVectorT<T>::copyFrom(const T* hostSrc,
                             size_t size,
                             hl_stream_t stream) {
  (void)stream;

  CHECK(hostSrc != NULL);
  CHECK_LE(size, this->size_);
  memcpy(this->data_, hostSrc, sizeof(T) * size);
}

template <class T>
void CpuVectorT<T>::copyTo(CpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());
  memcpy(dest->getData(), this->getData(), sizeof(T) * this->getSize());
}

template <class T>
void CpuVectorT<T>::copyTo(GpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());
  hl_memcpy_host2device((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(T) * this->getSize());
}

template <>
void CpuVectorT<real>::print(std::ostream& os, size_t num) const {
  size_t w = size_ < num ? size_ : num;
  os << "[";
  for (size_t i = 0; i < w; ++i) {
    os << data_[i] << " ";
  }
  os << "]" << std::endl;
}

template <>
void CpuVectorT<int>::print(std::ostream& os, size_t num) const {
  size_t w = size_ < num ? size_ : num;
  os << "[";
  for (size_t i = 0; i < w; ++i) {
    os << (int)data_[i] << " ";
  }
  os << "]" << std::endl;
}

template <>
void CpuVectorT<real>::printOneElement(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, size_);
  os << data_[idx] << ";";
}

template <>
void CpuVectorT<int>::printOneElement(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, size_);
  os << (int)data_[idx] << ";";
}

template <class T>
void ParallelCpuVectorT<T>::parallelExec(ExecFunc func) {
  LOG(FATAL) << "Not implemented";
}

template <>
void ParallelCpuVectorT<real>::parallelExec(ExecFunc func) {
  pool_->exec([this, func](int tid, size_t numThreads) {
    auto interval = calcSplitArrayInterval(
        this->getSize(), (size_t)tid, numThreads, 8LU /*for avx*/);
    // setup sub bufs
    CpuVector subVec(0, nullptr);
    subVec.subVecFrom(*this, interval);
    func(subVec);
  });
}

template <class T>
void ParallelCpuVectorT<T>::exec(SyncThreadPool::JobFunc func) {
  LOG(FATAL) << "Not implemented";
}

template <>
void ParallelCpuVectorT<real>::exec(SyncThreadPool::JobFunc func) {
  pool_->exec(func);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(size_t size, bool useGpu) : sync_(nullptr) {
  if (!useGpu) {
    cpuVectorT_ = std::make_shared<CpuVectorT<T>>(size);
  } else {
    gpuVectorT_ = std::make_shared<GpuVectorT<T>>(size);
  }
  setSync(useGpu);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(const std::shared_ptr<VectorT<T>>& src)
    : sync_(nullptr) {
  bool useGpu = src->useGpu();
  if (useGpu) {
    gpuVectorT_ = src;
  } else {
    cpuVectorT_ = src;
  }
  setSync(useGpu);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(size_t size, T* data, bool useGpu)
    : sync_(nullptr) {
  if (!useGpu) {
    cpuVectorT_ = std::make_shared<CpuVectorT<T>>(size, data);
    setSync(DATA_AT_CPU);
  } else {
    gpuVectorT_ = std::make_shared<GpuVectorT<T>>(size, data);
    setSync(DATA_AT_GPU);
  }
}

template <class T>
std::shared_ptr<CpuGpuVectorT<T>> CpuGpuVectorT<T>::create(size_t size,
                                                           bool useGpu) {
  return std::make_shared<CpuGpuVectorT<T>>(size, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::resize(size_t size, bool useGpu) {
  if (useGpu) {
    CHECK(gpuVectorT_) << "gpuVectorT_ is null";
    // If memoryHandle_ is nullptr,
    // the data may be owned by the caller when it was constructed.
    // It should not resize for this case.
    if (gpuVectorT_->getMemoryHandle()) {
      gpuVectorT_->resize(size);
    } else {
      CHECK_EQ(gpuVectorT_->getSize(), size);
    }
  } else {
    CHECK(cpuVectorT_) << "cpuVectorT_ is null";
    // If memoryHandle_ is nullptr,
    // the data may be owned by the caller when it was constructed.
    // It should not resize for this case.
    if (cpuVectorT_->getMemoryHandle()) {
      cpuVectorT_->resize(size);
    } else {
      CHECK_EQ(cpuVectorT_->getSize(), size);
    }
  }
  setSync(useGpu);
}

template <class T>
void CpuGpuVectorT<T>::resizeOrCreate(std::shared_ptr<CpuGpuVectorT<T>>& vec,
                                      size_t size,
                                      bool useGpu) {
  if (vec) {
    vec->resize(size, useGpu);
  } else {
    vec = create(size, useGpu);
  }
}

template <class T>
void CpuGpuVectorT<T>::resizeOrCreate(size_t size, bool useGpu) {
  if (useGpu && (!gpuVectorT_)) {
    gpuVectorT_ = VectorT<T>::create(size, true);
  } else if ((!useGpu) && (!cpuVectorT_)) {
    cpuVectorT_ = VectorT<T>::create(size, false);
  } else {
    CHECK((useGpu && gpuVectorT_) || (!useGpu && cpuVectorT_));
    this->resize(size, useGpu);
  }
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(CpuGpuVectorT<T>& src,
                                size_t offset,
                                size_t size)
    : sync_(nullptr) {
  CHECK_LE(offset + size, static_cast<size_t>(src.getSize()));
#ifndef PADDLE_ONLY_CPU
  SyncedFlag* flag = src.getSync();
  if (*flag == DATA_AT_CPU) {
    src.copyToGpu();  // will set synchronous data between CPU and GPU
  } else if (*flag == DATA_AT_GPU) {
    src.copyToCpu();  // will set synchronous data between CPU and GPU
  }
#endif
  auto cMemHandle = (src.getVector(false))->getMemoryHandle();
  cpuVectorT_ = std::make_shared<CpuVectorT<T>>(
      size, std::dynamic_pointer_cast<CpuMemoryHandle>(cMemHandle), offset);
#ifndef PADDLE_ONLY_CPU
  auto gMemHandle = (src.getVector(true))->getMemoryHandle();
  gpuVectorT_ = std::make_shared<GpuVectorT<T>>(
      size, std::dynamic_pointer_cast<GpuMemoryHandle>(gMemHandle), offset);
  src.setSync(SYNCED);
#endif
  setSync(src.getSync());
}

template <class T>
std::shared_ptr<const VectorT<T>> CpuGpuVectorT<T>::getVector(
    bool useGpu) const {
  auto* self = const_cast<CpuGpuVectorT<T>*>(this);
  if (useGpu) {
    self->copyToGpu();
    return std::const_pointer_cast<const VectorT<T>>(gpuVectorT_);
  } else {
    self->copyToCpu();
    return std::const_pointer_cast<const VectorT<T>>(cpuVectorT_);
  }
}

template <class T>
std::shared_ptr<VectorT<T>>& CpuGpuVectorT<T>::getMutableVector(bool useGpu) {
  setSync(useGpu);
  if (useGpu) {
    copyToGpu();
    return gpuVectorT_;
  } else {
    copyToCpu();
    return cpuVectorT_;
  }
}

template <class T>
const T* CpuGpuVectorT<T>::getData(bool useGpu) const {
  auto self = const_cast<CpuGpuVectorT<T>*>(this);
  if (useGpu) {
    self->copyToGpu();
    return gpuVectorT_->getData();
  } else {
    self->copyToCpu();
    return cpuVectorT_->getData();
  }
}

// Operation will change data and need to reset sync_ & syncFlag_.
#define MUTABLE_VECTOR_OP(OP, useGpu, args...) \
  do {                                         \
    setSync(useGpu);                           \
    if (useGpu) {                              \
      copyToGpu();                             \
      return gpuVectorT_->OP(args);            \
    } else {                                   \
      copyToCpu();                             \
      return cpuVectorT_->OP(args);            \
    }                                          \
  } while (0)

template <class T>
T* CpuGpuVectorT<T>::getMutableData(bool useGpu) {
  MUTABLE_VECTOR_OP(getData, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::zeroMem(bool useGpu) {
  MUTABLE_VECTOR_OP(zeroMem, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::fillSequence(bool useGpu) {
  MUTABLE_VECTOR_OP(fillSequence, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::setElement(size_t i, const T& value, bool useGpu) {
  MUTABLE_VECTOR_OP(setElement, useGpu, i, value);
}

template <class T>
T CpuGpuVectorT<T>::getElement(size_t i) const {
  switch (*this->getSync()) {
    case SYNCED:
    case DATA_AT_CPU:
      return cpuVectorT_->getElement(i);
      break;
    case DATA_AT_GPU:
      return gpuVectorT_->getElement(i);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  auto cVec = dynamic_cast<const CpuVectorT<T>*>(&src);
  auto gVec = dynamic_cast<const GpuVectorT<T>*>(&src);
  if (cVec) {
    copyToCpu(cVec->getData(), cVec->getSize(), stream);
  } else if (gVec) {
    copyToGpu(gVec->getData(), gVec->getSize(), stream);
  } else {
    LOG(FATAL) << "Invalid type of src";
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const T* data, size_t size, bool useGpu) {
  if (useGpu) {
    copyToGpu(data, size);
  } else {
    copyToCpu(data, size);
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const T* data,
                                size_t size,
                                hl_stream_t stream,
                                bool useGpu) {
  if (useGpu) {
    copyToGpu(data, size, stream);
  } else {
    copyToCpu(data, size, stream);
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(CpuGpuVectorT<T>& src,
                                size_t offset,
                                size_t size,
                                bool useGpu,
                                hl_stream_t stream) {
  if (useGpu) {
    VectorT<T>::resizeOrCreate(gpuVectorT_, size, true);
    gpuVectorT_->copyFrom(src.getData(true) + offset, size, stream);
  } else {
    VectorT<T>::resizeOrCreate(cpuVectorT_, size, false);
    cpuVectorT_->copyFrom(src.getData(false) + offset, size, stream);
  }
  setSync(useGpu);
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(CpuGpuVectorT<T>& src, hl_stream_t stream) {
  switch (*src.getSync()) {
    case DATA_AT_CPU:
      copyFrom(*(src.getVector(false)), stream);
      break;
    case DATA_AT_GPU:
      copyFrom(*(src.getVector(true)), stream);
      break;
    case SYNCED:
      copyFrom(*(src.getVector(false)), stream);
      copyFrom(*(src.getVector(true)), stream);
      setSync(SYNCED);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyToCpu() {
  switch (*this->getSync()) {
    case DATA_AT_GPU:
      CHECK(gpuVectorT_);
      this->resizeOrCreate(gpuVectorT_->getSize(), false);
      cpuVectorT_->copyFrom(*gpuVectorT_, HPPL_STREAM_DEFAULT);
      setSync(SYNCED);
      break;
    case DATA_AT_CPU:
    case SYNCED:
      CHECK(cpuVectorT_);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyToGpu() {
  switch (*this->getSync()) {
    case DATA_AT_CPU:
      CHECK(cpuVectorT_);
      this->resizeOrCreate(cpuVectorT_->getSize(), true);
      gpuVectorT_->copyFrom(*cpuVectorT_, HPPL_STREAM_DEFAULT);
      setSync(SYNCED);
      break;
    case DATA_AT_GPU:
    case SYNCED:
      CHECK(gpuVectorT_);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template class VectorT<real>;
template class VectorT<int>;
template class CpuVectorT<real>;
template class CpuVectorT<int>;
template class GpuVectorT<real>;
template class GpuVectorT<int>;
template class CpuGpuVectorT<real>;
template class CpuGpuVectorT<int>;

}  // namespace paddle
