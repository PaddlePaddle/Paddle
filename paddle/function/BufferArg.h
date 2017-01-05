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

#include <glog/logging.h>

#include "TensorShape.h"
#include "TensorType.h"
#include "paddle/math/CpuSparseMatrix.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {

enum BufferType {
  TENSOR_NORMAL = 0,
  TENSOR_SEQUENCE_ID = 1,
  TENSOR_SEQUENCE_DATA = 2,
  TENSOR_SPARSE = 3
};

enum SparseDataType {
  SPARSE_NO_VALUE = 0,  // do not need value pointer, all values are 1
  SPARSE_FLOAT_VALUE = 1
};

enum SparseDataFormat { SPARSE_CSR_FORMAT = 0, SPARSE_CSC_FORMAT = 1 };

/**
 * BufferArg used as the argument type for Function.
 */
class BufferArg;
class SequenceArg;
class SparseMatrixArg;
typedef std::shared_ptr<BufferArg> BufferArgPtr;

// an array of arbitrary dimensions
class BufferArg {
public:
  BufferArg(void* buf, ValueType valueType, const TensorShape& shape)
      : buf_(buf), valueType_(valueType), shape_(shape) {}

  BufferArg(void* buf, ValueType valueType)
      : buf_(buf), valueType_(valueType) {}

  BufferArg(const Matrix& matrix)
      : buf_((void*)matrix.getData()),
        valueType_(DataType<real>::value),
        shape_(2) {
    shape_.setDim(0, matrix.getHeight());
    shape_.setDim(1, matrix.getWidth());
  }

  BufferArg(const Matrix& matrix, const TensorShape& shape)
      : buf_((void*)matrix.getData()),
        valueType_(DataType<real>::value),
        shape_(shape) {
    CHECK_EQ(matrix.getElementCnt(), shape.getElements());
  }

  BufferArg(const Vector& vector)
      : buf_((void*)vector.getData()),
        valueType_(DataType<real>::value),
        shape_(1) {
    shape_.setDim(0, vector.getSize());
  }

  BufferArg(const IVector& vector)
      : buf_((void*)vector.getData()), valueType_(VALUE_TYPE_INT32), shape_(1) {
    shape_.setDim(0, vector.getSize());
  }

  template <DeviceType DType>
  typename Tensor<real, DType>::Matrix matrix() const {
    CHECK(buf_);
    CHECK(valueType_ == DataType<real>::value);
    // CHECK(deviceType_ == DType);
    CHECK_EQ(2, shape_.ndims());
    return typename Tensor<real, DType>::Matrix(
        reinterpret_cast<real*>(buf_), shape_[0], shape_[1]);
  }

  template <typename VType, DeviceType DType>
  typename Tensor<VType, DType>::Vector vector() const {
    CHECK(buf_);
    CHECK(valueType_ == DataType<VType>::value);
    // CHECK(deviceType_ == DType);
    CHECK_EQ(1, shape_.ndims());
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

  const SequenceArg& sequence() const;
  const SparseMatrixArg& sparse() const;

protected:
  void* buf_;
  ValueType valueType_;
  TensorShape shape_;
  BufferType bufferType_;
  // leading dimensions. The size is dims_.size()
  // Dims lds_;
};

// sequence start positions in a mini-batch of sequences
// shape_.ndims() == 1
// valueType_ = int32
// if a < b than value_.buf_[a] < value_.buf_[b]
class SequenceIdArg : public BufferArg {
public:
  SequenceIdArg(void* buf, const TensorShape& shape)
      : BufferArg(buf, VALUE_TYPE_INT32, shape) {
    CHECK_EQ(shape_.ndims(), 1);
    numSeqs_ = shape_[0] - 1;
  }

  SequenceIdArg(const IVector& vector) : BufferArg(vector) {
    numSeqs_ = shape_[0] - 1;
  }

  ~SequenceIdArg() {}

  size_t numSeqs() const { return numSeqs_; }

private:
  size_t numSeqs_;
};

// sequence data
class SequenceArg : public BufferArg {
public:
  SequenceArg(void* buf,
              ValueType valueType,
              const TensorShape& shape,
              const SequenceIdArg& startPositions)
      : BufferArg(buf, valueType, shape), startPositions_(startPositions) {}

  SequenceArg(const Matrix& matrix, const IVector& vector)
      : BufferArg(matrix), startPositions_(vector) {}

  ~SequenceArg() {}

  void* getIdBuf() const { return startPositions_.data(); }
  size_t numSeqs() const { return startPositions_.numSeqs(); }

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
                  SparseDataFormat format,
                  SparseDataType type)
      : BufferArg(buf, valueType, shape),
        row_(row),
        col_(col),
        nnz_(nnz),
        format_(format),
        type_(type) {
    CHECK((valueType == VALUE_TYPE_FLOAT) || (valueType == VALUE_TYPE_DOUBLE));
    CHECK_EQ(shape_.ndims(), 2);
    CHECK_EQ(row_.shape().ndims(), 1);
    CHECK_EQ(col_.shape().ndims(), 1);
    if (format == SPARSE_CSR_FORMAT) {
      CHECK_EQ(nnz, col.shape()[0]);
    } else if (format == SPARSE_CSC_FORMAT) {
      CHECK_EQ(nnz, row.shape()[0]);
    }
  }

  SparseMatrixArg(const CpuSparseMatrix& sparse)
      : BufferArg(sparse),
        row_((void*)sparse.getRows(), VALUE_TYPE_INT32),
        col_((void*)sparse.getCols(), VALUE_TYPE_INT32) {}

  SparseMatrixArg(const GpuSparseMatrix& sparse)
      : BufferArg(sparse),
        row_((void*)sparse.getRows(), VALUE_TYPE_INT32),
        col_((void*)sparse.getCols(), VALUE_TYPE_INT32) {}

  ~SparseMatrixArg() {}

  void* getRowBuf() const { return row_.data(); }

  void* getColBuf() const { return col_.data(); }

  size_t nnz() const { return nnz_; }

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
