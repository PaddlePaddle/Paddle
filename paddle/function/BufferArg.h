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

#pragma once

#include <glog/logging.h>

#include "TensorShape.h"
#include "TensorType.h"
#include "paddle/math/Matrix.h"

namespace paddle {

enum BufferType {
  TENSOR_UNKNOWN = 0,
  TENSOR_NORMAL = 1,
  TENSOR_SEQUENCE_ID = 2,
  TENSOR_SEQUENCE_DATA = 3,
  TENSOR_SPARSE = 4
};

class BufferArg;
class SequenceArg;
class SparseMatrixArg;

/**
 * \brief BufferArg used as the argument type of Function.
 *
 * The arguments of the Paddle Function have four Buffer types.
 * 1. BufferArg for a dense Buffer of any dimension.
 * 2. SequenceIdArg for a Buffer of sequence start positions.
 * 3. SequenceArg for a Buffer of sequence data.
 * 4. SparseMatrixArg for a Buffer of sparse matrix.
 *
 * Buffer shape
 * For most buffers, the first dimension `shape()[0]` represents
 * the size of the mini-batch.
 *
 * Buffer argType
 * There is an ArgType property for the BufferArg used as Function Output.
 * Whether the result of the Function calculation is assigned to the
 * output Buffer or added to the output Buffer is determined by the
 * argType_ property of the output BufferArg.
 */

// ArgType is only used by output BufferArg.
// For input argument, argType_ is ignored.
// For output argument, need to set the argType_ of the BufferArg.
enum ArgType {
  UNSPECIFIED = 0,
  ASSIGN_TO = 1,
  ADD_TO = 2,
};
class BufferArg {
 public:
  void setArgType(ArgType argType) { argType_ = argType; }

  ArgType getArgType() const { return argType_; }

 public:
  BufferArg(ValueType valueType,
            const TensorShape& shape,
            ArgType argType = UNSPECIFIED)
      : buf_(nullptr), valueType_(valueType), shape_(shape), argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
  }

  BufferArg(void* buf,
            ValueType valueType,
            const TensorShape& shape,
            ArgType argType = UNSPECIFIED)
      : buf_(buf), valueType_(valueType), shape_(shape), argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
  }

  BufferArg(void* buf, ValueType valueType) : buf_(buf), valueType_(valueType) {
    bufferType_ = TENSOR_NORMAL;
  }

  BufferArg(const Matrix& matrix, ArgType argType = UNSPECIFIED)
      : buf_(
            const_cast<void*>(reinterpret_cast<const void*>(matrix.getData()))),
        valueType_(DataType<real>::value),
        shape_(2),
        argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
    shape_.setDim(0, matrix.getHeight());
    shape_.setDim(1, matrix.getWidth());
  }

  BufferArg(const Matrix& matrix,
            const TensorShape& shape,
            ArgType argType = UNSPECIFIED)
      : buf_(
            const_cast<void*>(reinterpret_cast<const void*>(matrix.getData()))),
        valueType_(DataType<real>::value),
        shape_(shape),
        argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
    CHECK_EQ(matrix.getElementCnt(), shape.getElements());
  }

  BufferArg(const Vector& vector, ArgType argType = UNSPECIFIED)
      : buf_(
            const_cast<void*>(reinterpret_cast<const void*>(vector.getData()))),
        valueType_(DataType<real>::value),
        shape_(1),
        argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
    shape_.setDim(0, vector.getSize());
  }

  BufferArg(const IVector& vector, ArgType argType = UNSPECIFIED)
      : buf_(
            const_cast<void*>(reinterpret_cast<const void*>(vector.getData()))),
        valueType_(VALUE_TYPE_INT32),
        shape_(1),
        argType_(argType) {
    bufferType_ = TENSOR_NORMAL;
    shape_.setDim(0, vector.getSize());
  }

  template <DeviceType DType>
  typename Tensor<real, DType>::Matrix matrix() const {
    CHECK(buf_);
    CHECK(valueType_ == DataType<real>::value);
    // CHECK(deviceType_ == DType);
    CHECK_EQ((size_t)2, shape_.ndims());
    return typename Tensor<real, DType>::Matrix(
        reinterpret_cast<real*>(buf_), shape_[0], shape_[1]);
  }

  template <typename VType, DeviceType DType>
  typename Tensor<VType, DType>::Vector vector() const {
    CHECK(buf_);
    CHECK(valueType_ == DataType<VType>::value);
    // CHECK(deviceType_ == DType);
    CHECK_EQ((size_t)1, shape_.ndims());
    return typename Tensor<VType, DType>::Vector(
        shape_[0], reinterpret_cast<VType*>(buf_));
  }

  virtual ~BufferArg() {}

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(buf_);
  }

  void* data() const { return buf_; }
  ValueType valueType() const { return valueType_; }
  BufferType bufferType() const { return bufferType_; }
  const TensorShape& shape() const { return shape_; }
  bool isSparseArg() const { return TENSOR_SPARSE == bufferType_; }
  bool isSequenceArg() const { return TENSOR_SEQUENCE_DATA == bufferType_; }
  virtual size_t numElements() const { return shape_.getElements(); }

  const SequenceArg& sequence() const;
  const SparseMatrixArg& sparse() const;

 protected:
  void* buf_;
  ValueType valueType_;
  TensorShape shape_;
  BufferType bufferType_{TENSOR_UNKNOWN};
  ArgType argType_{UNSPECIFIED};
  // TODO(tianbing), add deviceType_
  // leading dimensions. The size is dims_.size()
  // Dims lds_;
};

// sequence start positions in a mini-batch of sequences
// shape_.ndims() == 1
// valueType_ = int32
// if a < b then value_.buf_[a] < value_.buf_[b]
class SequenceIdArg : public BufferArg {
 public:
  SequenceIdArg(const TensorShape& shape, ArgType argType = UNSPECIFIED)
      : BufferArg(VALUE_TYPE_INT32, shape, argType) {
    bufferType_ = TENSOR_SEQUENCE_ID;
    CHECK_EQ(shape_.ndims(), 1UL);
    CHECK_GE(shape_[0], 1UL);
    numSeqs_ = shape_[0] - 1;
  }

  SequenceIdArg(void* buf,
                const TensorShape& shape,
                ArgType argType = UNSPECIFIED)
      : BufferArg(buf, VALUE_TYPE_INT32, shape, argType) {
    bufferType_ = TENSOR_SEQUENCE_ID;
    CHECK_EQ(shape_.ndims(), 1UL);
    numSeqs_ = shape_[0] - 1;
  }

  SequenceIdArg(const IVector& vector) : BufferArg(vector) {
    bufferType_ = TENSOR_SEQUENCE_ID;
    numSeqs_ = shape_[0] - 1;
  }

  ~SequenceIdArg() {}

  size_t numSeqs() const { return numSeqs_; }

 private:
  size_t numSeqs_;
};

// sequences data
// For mini-batch calculate,
// one batch can contain more than one sequence of data.
// SequenceArg can be used to represent sequences that contain multiple
// unequal lengths.
class SequenceArg : public BufferArg {
 public:
  SequenceArg(ValueType valueType,
              const TensorShape& shape,
              ArgType argType = UNSPECIFIED)
      : BufferArg(valueType, shape, argType),
        startPositions_(TensorShape({shape[0]})) {
    bufferType_ = TENSOR_SEQUENCE_DATA;
  }

  SequenceArg(void* buf,
              ValueType valueType,
              const TensorShape& shape,
              const SequenceIdArg& startPositions,
              ArgType argType = UNSPECIFIED)
      : BufferArg(buf, valueType, shape, argType),
        startPositions_(startPositions) {
    bufferType_ = TENSOR_SEQUENCE_DATA;
  }

  SequenceArg(const Matrix& matrix,
              const IVector& vector,
              ArgType argType = UNSPECIFIED)
      : BufferArg(matrix, argType), startPositions_(vector) {
    bufferType_ = TENSOR_SEQUENCE_DATA;
  }

  ~SequenceArg() {}

  void* getIdBuf() const { return startPositions_.data(); }
  size_t numSeqs() const { return startPositions_.numSeqs(); }
  SequenceIdArg& getSequenceId() { return startPositions_; }
  const SequenceIdArg& getSequenceId() const { return startPositions_; }

 private:
  SequenceIdArg startPositions_;
};

// sparse matrix
// valueType_ == float or double
// shape_.ndims() == 2
class SparseMatrixArg : public BufferArg {
 public:
  SparseMatrixArg(void* buf,
                  ValueType valueType,
                  const TensorShape& shape,
                  const BufferArg& row,
                  const BufferArg& col,
                  size_t nnz,
                  SparseFormat format,
                  SparseValueType type,
                  ArgType argType = UNSPECIFIED)
      : BufferArg(buf, valueType, shape, argType),
        row_(row),
        col_(col),
        nnz_(nnz),
        format_(static_cast<SparseDataFormat>(format)),
        type_(static_cast<SparseDataType>(type)) {
    bufferType_ = TENSOR_SPARSE;
    CHECK((valueType == VALUE_TYPE_FLOAT) || (valueType == VALUE_TYPE_DOUBLE));
    CHECK_EQ(shape_.ndims(), 2UL);
    CHECK_EQ(row_.shape().ndims(), 1UL);
    CHECK_EQ(col_.shape().ndims(), 1UL);
    if (format_ == T_SPARSE_CSR) {
      CHECK_EQ(nnz, col.shape()[0]);
    } else if (format_ == T_SPARSE_CSC) {
      CHECK_EQ(nnz, row.shape()[0]);
    }
  }

  SparseMatrixArg(ValueType valueType,
                  const TensorShape& shape,
                  size_t nnz,
                  SparseFormat format,
                  SparseValueType type,
                  ArgType argType = UNSPECIFIED)
      : BufferArg(valueType, shape, argType),
        row_(BufferArg(nullptr, VALUE_TYPE_INT32)),
        col_(BufferArg(nullptr, VALUE_TYPE_INT32)),
        nnz_(nnz),
        format_(static_cast<SparseDataFormat>(format)),
        type_(static_cast<SparseDataType>(type)) {
    bufferType_ = TENSOR_SPARSE;
    CHECK((valueType == VALUE_TYPE_FLOAT) || (valueType == VALUE_TYPE_DOUBLE));
    CHECK_EQ(shape_.ndims(), 2UL);

    /// len of row_ : height + 1 (CSR) or nnz (CSC), buf_ == nullptr
    row_ = (format_ == T_SPARSE_CSR
                ? BufferArg(VALUE_TYPE_INT32, TensorShape{shape_[0] + 1})
                : BufferArg(VALUE_TYPE_INT32, TensorShape{nnz}));
    /// len of col_ :  width + 1 (CSC) or nnz (CSR), buf_ == nullptr
    col_ = (format_ == T_SPARSE_CSR
                ? BufferArg(VALUE_TYPE_INT32, TensorShape{nnz})
                : BufferArg(VALUE_TYPE_INT32, TensorShape{shape_[1] + 1}));
  }

  SparseMatrixArg(const CpuSparseMatrix& sparse, ArgType argType = UNSPECIFIED);

  SparseMatrixArg(const GpuSparseMatrix& sparse, ArgType argType = UNSPECIFIED);

  template <DeviceType DType>
  typename Tensor<real, DType>::SparseMatrix SparseMatrix() const {
    CHECK(buf_);
    CHECK(valueType_ == DataType<real>::value);
    // CHECK(deviceType_ == DType);
    CHECK_EQ(2UL, shape_.ndims());
    return typename Tensor<real, DType>::SparseMatrix(
        reinterpret_cast<real*>(buf_),
        reinterpret_cast<int*>(row_.data()),
        reinterpret_cast<int*>(col_.data()),
        shape_[0],
        shape_[1],
        nnz_,
        static_cast<SparseValueType>(type_),
        static_cast<SparseFormat>(format_),
        false);
  }

  ~SparseMatrixArg() {}

  void* getRowBuf() const { return row_.data(); }

  void* getColBuf() const { return col_.data(); }

  size_t nnz() const { return nnz_; }

  size_t numElements() const override { return nnz_; }

  SparseDataFormat dataFormat() const { return format_; }

  SparseDataType dataType() const { return type_; }

 private:
  BufferArg row_;
  BufferArg col_;
  size_t nnz_;
  SparseDataFormat format_;
  SparseDataType type_;
};

}  // namespace paddle
