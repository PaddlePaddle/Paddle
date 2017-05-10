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

#pragma once

#include <cmath>
#include <memory>

#include <hl_gpu.h>

#include "BaseMatrix.h"
#include "MemoryHandle.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Thread.h"

namespace paddle {

template <class T>
class GpuVectorT;
template <class T>
class CpuVectorT;

template <class T>
class BaseVector;

class SyncThreadPool;

class Matrix;

template <class T>
class BaseVector : public BaseMatrixT<T> {
public:
  BaseVector(size_t size, T* data, bool useGpu)
      : BaseMatrixT<T>(1, size, data, false, useGpu), size_(this->width_) {}

  ~BaseVector() {}

protected:
  size_t& size_;
};

/**
 * Copy or assignemnt constructor will share the data as opposed to making a
 * copy of the original data. To make a copy of the orinal data, use copyFrom()
 * instead.
 */
template <class T>
class VectorT : public BaseVector<T> {
protected:
  VectorT(size_t size, MemoryHandlePtr memoryHandle, size_t offset, bool useGpu)
      : BaseVector<T>(size,
                      reinterpret_cast<T*>(memoryHandle->getBuf()) + offset,
                      useGpu) {
    memoryHandle_ = memoryHandle;
  }

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  VectorT(size_t size, T* data, bool useGpu)
      : BaseVector<T>(size, data, useGpu) {}

public:
  virtual ~VectorT() {}

  static std::shared_ptr<VectorT<T>> create(size_t size, bool useGpu);

  static std::shared_ptr<VectorT<T>> create(T* data, size_t size, bool useGpu);

  static std::shared_ptr<VectorT<T>> create(size_t size,
                                            MemoryHandlePtr memoryHandle,
                                            size_t offset = 0);

  // owner can set SyncThreadPool,
  // if not set, will use globalSyncThreadPool,
  // which can be used in main thread only.
  static std::shared_ptr<VectorT<T>> createParallelVector(
      size_t size, bool useGpu, SyncThreadPool* pool = nullptr);

  size_t getSize() const { return this->size_; }
  const T* getData() const { return this->data_; }
  T* getData() { return this->data_; }

  virtual void zeroMem() = 0;
  // set all elements to value
  virtual void reset(const T& value) = 0;
  // fill data by 0, 1, 2, ...
  virtual void fillSequence() = 0;

  MemoryHandlePtr getMemoryHandle() const { return memoryHandle_; }

  /**
   * resizing to a big vector will not preserve old values.
   */
  void resize(size_t newSize) {
    if (!memoryHandle_ || newSize * sizeof(T) > memoryHandle_->getAllocSize()) {
      memoryHandle_ = newMemory(newSize * sizeof(T));
      this->data_ = reinterpret_cast<T*>(memoryHandle_->getBuf());
    }
    this->size_ = newSize;
  }

  static void resizeOrCreate(std::shared_ptr<VectorT<T>>& vec,
                             size_t size,
                             bool useGpu) {
    if (vec) {
      vec->resize(size);
    } else {
      vec = create(size, useGpu);
    }
  }

  virtual MemoryHandlePtr newMemory(size_t size) = 0;

  /**
   * form sub vector from *src*, shallow copy
   */
  void subVecFrom(const VectorT<T>& src, size_t start, size_t size) {
    CHECK_EQ(BaseVector<T>::useGpu_, src.useGpu_);
    CHECK_LT(start, src.size_);
    CHECK_LE(start + size, src.size_);

    BaseVector<T>::size_ = size;
    BaseVector<T>::data_ = const_cast<T*>(src.data_) + start;
  }

  std::shared_ptr<VectorT<T>> subVec(size_t start, size_t size) {
    CHECK_LE(start + size, static_cast<size_t>(getSize()));
    return VectorT<T>::create(getData() + start, size, BaseVector<T>::useGpu_);
  }

  /**
   * form sub vector from *src*, shallow copy
   */
  void subVecFrom(const T* src, size_t start, size_t size) {
    BaseVector<T>::size_ = size;
    BaseVector<T>::data_ = const_cast<T*>(src) + start;
  }

  /**
   * form sub vector from *src*, shallow copy
   * in *interval* [interval.first, interval.second)
   */
  void subVecFrom(const VectorT<T>& src, std::pair<size_t, size_t> interval) {
    subVecFrom(src, interval.first, interval.second - interval.first);
  }

  /**
   * convert the vector to a sparse one_hot matrix of width idRange
   * only applies to IVector
   */
  std::shared_ptr<Matrix> toOneHotSparseMatrix(size_t idRange, bool useGpu);

  /**
   * This function will crash if the size of src and dest is different.
   */
  virtual void copyFrom(const VectorT<T>& src) = 0;

  /**
   * If use_gpu, this function will push the copy-task to the specifed-stream
   * and return immediately.
   *
   * If not use GPU, this function is same as
   * the copyFrom(const VectorT<T>& src), which use stream HPPL_STREAM_DEFAULT.
   */
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream) = 0;

  /**
   * copy size elements from src
   *
   * If this is GpuVector, src can be cpu or gpu memory
   *
   * If this is CpuVector, src is assumed to be cpu memory
   */
  virtual void copyFrom(const T* src, size_t size) = 0;

  /**
   * copy size elements from src
   *
   * If this is GpuVector, src can be cpu or gpu memory
   *
   * If this is CpuVector, src is assumed to be cpu memory,
   */
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream) = 0;

  /**
   * exec a func in single/multi thread
   */
  virtual void exec(SyncThreadPool::JobFunc func) { func(0, 1); }

  /// Get the buffer point with beginPos
  virtual T* getPoint(const uint64_t beginPos) = 0;

  /// Get the value for the i'th element
  virtual T getElement(size_t i) const = 0;
  virtual void setElement(size_t i, const T& value) = 0;

  //----------  math operations ----------------

  // sum of the absolute value of each elements
  virtual T getAbsSum() = 0;

  virtual T getSum() = 0;
  virtual T getMax() = 0;
  virtual T getAbsMax() = 0;
  virtual T getMin() = 0;

  /// element-wise calc:  this = (b == value)
  virtual void isEqualTo(const VectorT<T>& b, const T& value) = 0;

  /// select elements indexed by *ids* from vector *src*
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids) = 0;

  enum HistogramType {
    HISTOGRAM_EXPONENT = 0,
  };

  /**
   * @brief  print histogram of vector values
   *
   * @note   only exponent histogram supported currently
   */
  virtual void histogram(std::ostream& os, int type = HISTOGRAM_EXPONENT) = 0;

  /// generate uniform random value for each element
  virtual void rand() = 0;
  /**
   * generate uniform random value for each element,
   * data range is from 0 to (classes - 1).
   */
  virtual void rand(size_t classes) = 0;

  /**
   * Debug use only. Very inefficient for GPU vector.
   * get the value at pos.
   */
  virtual T get(size_t pos) = 0;

  /**
   * generate univariate Gaussian distributed random numbers
   * with given mean and standardDeviation.
   */
  virtual void randnorm(real mean, real standardDeviation) = 0;

  /**
   * generate uniform distributed random numbers
   * with given range.
   */
  virtual void uniform(real left, real right) = 0;

  /// print the first "num" elements of the Vector
  virtual void print(std::ostream& os, size_t num) const = 0;

  /// print the "idx" element of the Vector
  virtual void printOneElement(std::ostream& os, size_t idx) const = 0;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    if (BaseVector<T>::useGpu_) {
      TensorGpuApply<T>(*this, expr);
    } else {
      TensorCpuApply<T>(*this, expr);
    }
  }

protected:
  friend class GpuVectorT<T>;
  friend class CpuVectorT<T>;
  virtual void copyTo(CpuVectorT<T>* dest) const = 0;
  virtual void copyTo(GpuVectorT<T>* dest) const = 0;
  MemoryHandlePtr memoryHandle_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const VectorT<T>& vec) {
  vec.print(os, vec.getSize());
  return os;
}

template <class T>
class GpuVectorT : public VectorT<T> {
public:
  explicit GpuVectorT(size_t size);
  GpuVectorT(size_t size, GpuMemHandlePtr memHandle, size_t offset)
      : VectorT<T>(size, memHandle, offset, true) {}

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  GpuVectorT(size_t size, T* data) : VectorT<T>(size, data, true) {}

  virtual MemoryHandlePtr newMemory(size_t size) {
    return std::make_shared<GpuMemoryHandle>(size);
  }
  virtual void zeroMem();
  virtual void reset(const T& value);
  virtual void fillSequence();

  virtual void copyFrom(const T* src, size_t size);
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream);
  virtual void copyFrom(const VectorT<T>& src);
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream);
  virtual T getElement(size_t i) const;
  virtual void setElement(size_t i, const T& value);
  virtual T* getPoint(const uint64_t beginPos);

  virtual T getAbsSum();
  virtual T getSum();
  virtual T getMax();
  virtual T getAbsMax();
  virtual T getMin();
  virtual void isEqualTo(const VectorT<T>& b, const T& value);
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids);
  virtual void histogram(std::ostream& os, int type);
  virtual void rand();
  virtual void rand(size_t classes);
  virtual void randnorm(real mean, real standardDeviation);
  virtual void uniform(real left, real right);
  virtual T get(size_t pos);
  virtual void print(std::ostream& os, size_t num) const;
  virtual void printOneElement(std::ostream& os, size_t idx) const;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorGpuApply<T>(*this, expr);
  }

protected:
  virtual void copyTo(CpuVectorT<T>* dest) const;
  virtual void copyTo(GpuVectorT<T>* dest) const;
};

template <class T>
class CpuVectorT : public VectorT<T> {
public:
  explicit CpuVectorT(size_t size);
  CpuVectorT(size_t size, MemoryHandlePtr memoryHandle, size_t offset)
      : VectorT<T>(size, memoryHandle, offset, false) {}

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  CpuVectorT(size_t size, T* data) : VectorT<T>(size, data, false) {}

  /**
   * If src is a CpuVector, the new CpuVector will share the data with src
   *
   * If src is a GpuVector, the new CpuVector will copy data from src
   */
  explicit CpuVectorT(const VectorT<T>& src);

  virtual MemoryHandlePtr newMemory(size_t size) {
    return std::make_shared<CpuMemoryHandle>(size);
  }

  virtual void zeroMem();
  virtual void reset(const T& value);
  virtual void fillSequence();
  virtual void copyFrom(const T* src, size_t size);
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream);
  virtual void copyFrom(const VectorT<T>& src);
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream);
  virtual void copyTo(CpuVectorT<T>* dest) const;
  virtual void copyTo(GpuVectorT<T>* dest) const;

  /// Get the buffer point with beginPos
  virtual T* getPoint(const uint64_t beginPos) {
    return this->getData() + beginPos;
  }

  virtual T getElement(size_t i) const { return this->getData()[i]; }
  virtual void setElement(size_t i, const T& value) {
    this->getData()[i] = value;
  }

  virtual T getAbsSum();
  virtual T getSum();
  virtual T getMax();
  virtual T getAbsMax();
  virtual T getMin();
  virtual void isEqualTo(const VectorT<T>& b, const T& value);
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids);
  virtual void histogram(std::ostream& os, int type);
  virtual void rand();
  virtual void rand(size_t classes);
  virtual void randnorm(real mean, real standardDeviation);
  virtual void uniform(real left, real right);
  virtual T get(size_t pos);
  virtual void print(std::ostream& os, size_t num) const;
  virtual void printOneElement(std::ostream& os, size_t idx) const;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorCpuApply<T>(*this, expr);
  }
};

template <class T>
class ParallelCpuVectorT : public CpuVectorT<T> {
public:
  ParallelCpuVectorT(size_t size, SyncThreadPool* pool)
      : CpuVectorT<T>(size), pool_(pool) {}

  virtual void zeroMem() {
    parallelExec([](CpuVectorT<T>& vec) { vec.CpuVectorT<T>::zeroMem(); });
  }
  virtual void randnorm(real mean, real standardDeviation) {
    parallelExec([=](CpuVectorT<T>& vec) {
      vec.CpuVectorT<T>::randnorm(mean, standardDeviation);
    });
  }
  virtual void uniform(real left, real right) {
    parallelExec(
        [=](CpuVectorT<T>& vec) { vec.CpuVectorT<T>::uniform(left, right); });
  }

  virtual void exec(SyncThreadPool::JobFunc jobFunc);

private:
  typedef std::function<void(CpuVectorT<T>& vec)> ExecFunc;
  void parallelExec(ExecFunc func);
  SyncThreadPool* pool_;
};

/**
 * A class to do conversion between CpuVector and GpuVector automatically.
 */
template <class T>
class CpuGpuVectorT {
public:
  /**
   * @brief An enum type of SyncedFlag using to
   *        mark data memory is in CPU or GPU.
   *
   * DATA_AT_CPU: data is located in CPU.
   *
   * DATA_AT_GPU: data is located in GPU.
   *
   * SYNCED: data is located in CPU and GPU simultaneously.
   */
  enum SyncedFlag { DATA_AT_CPU = 0, DATA_AT_GPU = 1, SYNCED = 2 };

  /**
   * @brief A constructor, create cpuVectorT_ or gpuVectorT_.
   *
   * @param[in] size    data size.
   * @param[in] useGpu  use gpu or not.
   */
  explicit CpuGpuVectorT(size_t size, bool useGpu);

  /**
   * @brief A constructor, create CpuGpuVectorT by VectorT.
   *
   * If src is CpuVector, cpuVectorT_ is shared data with src.
   *
   * If src is GpuVector, gpuVectorT_ is shared data with src.
   */
  explicit CpuGpuVectorT(const std::shared_ptr<VectorT<T>>& src);

  /**
   * @brief A constructor.
   *
   * If useGpu is true, data should be located in device and
   * create gpuVectorT_ with data.
   *
   * If useGpu is false, data should be located in host and
   * create cpuVectorT_ with data.
   *
   * @note Data is owned by the caller and should be valid during
   *       the life of this vector.
   *       Caller is responsible for release the memory.
   */
  CpuGpuVectorT(size_t size, T* data, bool useGpu);

  CpuGpuVectorT(CpuGpuVectorT<T>& src, size_t offset, size_t size);

  virtual ~CpuGpuVectorT() {}

  static std::shared_ptr<CpuGpuVectorT<T>> create(size_t size, bool useGpu);

  /**
   * @brief resize vector.
   *
   * If useGpu is true, resize gpuVectorT_ and set syncFlag_ to DATA_AT_GPU,
   *
   * otherwise resize cpuVectorT_ and set syncFlag_ to DATA_AT_CPU.
   */
  void resize(size_t size, bool useGpu);

  /**
   * @brief resize or create CpuGpuVectorT.
   */
  static void resizeOrCreate(std::shared_ptr<CpuGpuVectorT<T>>& vec,
                             size_t size,
                             bool useGpu);

  /**
   * @brief return a const cpuVectorT_ or gpuVectorT_.
   *
   * If useGpu is true, return gpuVectorT_.
   *
   * If useGpu is false, return cpuVectorT_.
   *
   * @note Caller should not change the data.
   *       If caller changes const attribute,
   *       should set syncFlag_.
   */
  std::shared_ptr<const VectorT<T>> getVector(bool useGpu) const;

  /**
   * @brief return a const cpuVectorT_ or gpuVectorT_.
   *
   * @note: This interface will change syncFlag_, so if you will
   *        not change the data, you should call getVector.
   */
  std::shared_ptr<VectorT<T>>& getMutableVector(bool useGpu);

  /**
   * @brief return const T* data.
   *
   * If useGpu is true, return device data.
   *
   * If useGpu is false, return host data.
   */
  const T* getData(bool useGpu) const;

  // TODO(yuyang18): Make getData more c++ style.
  //  inline T* getData(bool useGpu) {
  //    return getMutableData(useGpu);
  //  }

  T* getMutableData(bool useGpu);

  /**
   * If useGpu is true, gpuVectorT_->Op().
   *
   * If useGpu is false, cpuVectorT_->Op().
   *
   * Op is zeroMem, fillSequence, ...
   */
  void zeroMem(bool useGpu);
  void fillSequence(bool useGpu);
  void setElement(size_t i, const T& value, bool useGpu);

  /**
   * @brief return i-th element.
   */
  T getElement(size_t i) const;

  /**
   * @brief return vector size.
   */
  size_t getSize() const {
    size_t size = 0;
    switch (*sync_) {
      case SYNCED:
      case DATA_AT_CPU:
        size = cpuVectorT_->getSize();
        break;
      case DATA_AT_GPU:
        size = gpuVectorT_->getSize();
        break;
      default:
        LOG(FATAL) << "Not support";
        break;
    }
    return size;
  }

  /// copy data to cpuVectorT_.
  inline void copyToCpu(const T* data, size_t size) {
    this->resizeOrCreate(size, false);
    cpuVectorT_->copyFrom(data, size);
    setSync(DATA_AT_CPU);
  }
  /// copy data to cpuVectorT_ using specifed-stream.
  inline void copyToCpu(const T* data, size_t size, hl_stream_t stream) {
    this->resizeOrCreate(size, false);
    cpuVectorT_->copyFrom(data, size, stream);
    setSync(DATA_AT_CPU);
  }

  /// copy data to gpuVectorT_.
  inline void copyToGpu(const T* data, size_t size) {
    this->resizeOrCreate(size, true);
    gpuVectorT_->copyFrom(data, size);
    setSync(DATA_AT_GPU);
  }
  /// copy data to gpuVectorT_ using specifed-stream.
  inline void copyToGpu(const T* data, size_t size, hl_stream_t stream) {
    this->resizeOrCreate(size, true);
    gpuVectorT_->copyFrom(data, size, stream);
    setSync(DATA_AT_GPU);
  }

  /**
   * @brief copy from src using specifed-stream.
   *
   * If src is CpuVectorT, copy to cpuVectorT_.
   *
   * If src is GpuVectorT, copy to gpuVectorT_.
   */
  void copyFrom(const VectorT<T>& src, hl_stream_t stream);

  /**
   * @brief copy data.
   *
   * If useGpu is false, copy host data to cpuVectorT_.
   *
   * If useGpu is true, copy device data to gpuVectorT_.
   *
   * @note  data address should consistent with useGpu.
   */
  void copyFrom(const T* data, size_t size, bool useGpu);
  void copyFrom(const T* data, size_t size, hl_stream_t stream, bool useGpu);

  /**
   * @brief copy from (src + offset) using specifed-stream.
   */
  void copyFrom(CpuGpuVectorT<T>& src,
                size_t offset,
                size_t size,
                bool useGpu,
                hl_stream_t stream);

  /**
   * @brief copy from src using specifed-stream.
   */
  void copyFrom(CpuGpuVectorT<T>& src, hl_stream_t stream);

  /**
   * @brief return sync_.
   */
  inline SyncedFlag* getSync() const { return sync_; }

  /**
   * @brief set sync_.
   */
  inline void setSync(SyncedFlag* sync) { sync_ = sync; }

  inline void setSync(SyncedFlag syncFlag) {
    if (sync_) {
      *sync_ = syncFlag;
    } else {
      syncFlag_ = syncFlag;
      sync_ = &syncFlag_;
    }
  }

  inline void setSync(bool useGpu) {
    SyncedFlag flag = useGpu ? DATA_AT_GPU : DATA_AT_CPU;
    setSync(flag);
  }

protected:
  void resizeOrCreate(size_t size, bool useGpu);

  /**
   * @brief copy between cpuVectorT_ and gpuVectorT_.
   *
   * If syncFlag_ is DATA_AT_CPU and SYNCED, do nothing.
   *
   * If syncFlag_ is DATA_AT_GPU, copy gpuVectorT_ to cpuVectorT_
   *   and set syncFlag_ to SYNCED.
   */
  void copyToCpu();

  /**
   * @brief copy between cpuVectorT_ and gpuVectorT_.
   *
   * If syncFlag_ is DATA_AT_GPU and SYNCED, do nothing.
   *
   * If syncFlag_ is DATA_AT_CPU, copy cpuVectorT_ to gpuVectorT_
   *   and set syncFlag_ to SYNCED.
   */
  void copyToGpu();

  /// host pointer.
  std::shared_ptr<VectorT<T>> cpuVectorT_;
  /// device pointer.
  std::shared_ptr<VectorT<T>> gpuVectorT_;
  /// specify current data address.
  SyncedFlag syncFlag_;
  SyncedFlag* sync_;
};

typedef VectorT<real> Vector;
typedef CpuVectorT<real> CpuVector;
typedef GpuVectorT<real> GpuVector;

typedef VectorT<int> IVector;
typedef CpuVectorT<int> CpuIVector;
typedef GpuVectorT<int> GpuIVector;

typedef std::shared_ptr<Vector> VectorPtr;
typedef std::shared_ptr<CpuVector> CpuVectorPtr;
typedef std::shared_ptr<GpuVector> GpuVectorPtr;

typedef std::shared_ptr<IVector> IVectorPtr;
typedef std::shared_ptr<CpuIVector> CpuIVectorPtr;
typedef std::shared_ptr<GpuIVector> GpuIVectorPtr;

typedef CpuGpuVectorT<real> CpuGpuVector;
typedef CpuGpuVectorT<int> ICpuGpuVector;
typedef std::shared_ptr<CpuGpuVector> CpuGpuVectorPtr;
typedef std::shared_ptr<ICpuGpuVector> ICpuGpuVectorPtr;

}  // namespace paddle
