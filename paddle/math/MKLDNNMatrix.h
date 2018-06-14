/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "Matrix.h"
#include "mkldnn.hpp"
#include "paddle/parameter/Parameter.h"

namespace paddle {

class MKLDNNMatrix;
typedef std::shared_ptr<MKLDNNMatrix> MKLDNNMatrixPtr;

#define CHECK_PRIMITIVE_DESC_EQ(MAT, PD, ...)                        \
  CHECK(MAT) << " can not be empty.";                                \
  CHECK(MAT->getPrimitiveDesc() == PD)                               \
      << #MAT "->getPrimitiveDesc() and " #PD " should be equal.\n " \
      << "" __VA_ARGS__;

/**
 * @brief MKLDNN Matrix.
 *
 */
class MKLDNNMatrix : public CpuMatrix, public mkldnn::memory {
 public:
  MKLDNNMatrix(CpuMatrixPtr m, mkldnn::memory::primitive_desc pd)
      : CpuMatrix(m->getData(), m->getHeight(), m->getWidth(), false),
        mkldnn::memory(pd, m->getData()),
        m_(m) {}

  ~MKLDNNMatrix() {}

  /**
   * Create MKLDNNMatrix from a MatrixPtr and memory primitive_desc
   */
  static MKLDNNMatrixPtr create(mkldnn::memory::primitive_desc pd,
                                MatrixPtr m = nullptr);

  /**
   * Create MKLDNNMatrix from a MatrixPtr and memory details info
   */
  static MKLDNNMatrixPtr create(
      mkldnn::memory::dims dims,
      mkldnn::memory::format fmt,
      mkldnn::engine& eg,
      MatrixPtr m = nullptr,
      mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32);

  /**
   * Create primitive descriptor.
   * default with f32 dtype
   */
  static mkldnn::memory::primitive_desc createPrimitiveDesc(
      const mkldnn::memory::dims dims,
      const mkldnn::memory::format& fmt,
      const mkldnn::engine& eg,
      const mkldnn::memory::data_type& dtype = mkldnn::memory::data_type::f32) {
    return mkldnn::memory::primitive_desc(memory::desc(dims, dtype, fmt), eg);
  }

  /**
   * Create Memory descriptor.
   * default with any format and f32 dtype
   */
  static mkldnn::memory::desc createMemoryDesc(
      const mkldnn::memory::dims dims,
      const mkldnn::memory::format& fmt = mkldnn::memory::format::any,
      const mkldnn::memory::data_type& dtype = mkldnn::memory::data_type::f32) {
    return mkldnn::memory::desc(dims, dtype, fmt);
  }

  /**
   * Create reorder primitive.
   * Create a mkldnn::reorder handle for converting src MKLDNNMatrix to dst.
   * checkData: whether to check the data handle of src and dst.
   *            if true, it will check the data and do not allow them equal;
   *            otherwise, it will not check them, then the reorder created
   *            may have inplace buffer.
   *            Do not set false, if you can not guarantee the inplace logical
   *            would work with your reorder.
   */
  static std::shared_ptr<mkldnn::reorder> createReorder(
      const MKLDNNMatrixPtr& src,
      const MKLDNNMatrixPtr& dst,
      bool checkData = true);

  void copyFrom(const Matrix& src) {
    // TODO(TJ): reorder data if this format is not nchw or x
    m_->copyFrom(src);
  }

  void copyTo(Matrix& dst) {
    // TODO(TJ): reorder data if this format is not nchw or x
    dst.copyFrom(*m_);
  }

 public:
  /**
   * Reorder this MKLDNNMatrix from other format.
   * Support inplace reorder.
   * @note: this function would only reorder the data layout.
   *        will NOT change this original dim or format info
   */
  void reorderDataFrom(const MKLDNNMatrixPtr& m,
                       memory::format srcFmt,
                       memory::dims targetDim);

  /**
   * Reorder this MKLDNNMatrix to other format.
   * Support inplace reorder.
   * @note: this function would only reorder the data layout.
   *        will NOT change the dst dim or format info
   */
  void reorderDataTo(const MKLDNNMatrixPtr& m,
                     memory::format dstFmt,
                     memory::dims targetDim);

  /**
   * Dimensionality reduction.
   * Change format "nchw --> nc" or "oihw --> oi" if the h and w are both 1
   */
  void downSpatial();

  /**
   * set the memory data handle.
   * Caution: This will not check the buffer size of the data,
   *          it should be coverd by user.
   */
  void setData(real* data) {
    set_data_handle(data);
    CpuMatrix::setData(data);
    m_.reset();
  }

  /**
   * override the CpuMatrix::resize
   */
  void resize(size_t newHeight, size_t newWidth) override {
    m_->resize(newHeight, newWidth);
    if (data_ == m_->getData() && elementCnt_ == newHeight * newWidth) {
      return;
    }
    CpuMatrix::setData(data_);
    height_ = newHeight;
    width_ = newWidth;
    elementCnt_ = newHeight * newWidth;
    stride_ = width_;
    auto pd = mkldnn::memory::primitive_desc(
        mkldnn::memory::desc({(int)newHeight, (int)newWidth},
                             getDtype(),
                             mkldnn::memory::format::nc),
        getEngine());
    resetMKLDNNMemory(pd, data_);
  }

  /**
   * override Matrix::getData
   * check data before return
   */
  real* getData() override {
    CHECK_EQ((void*)data_, get_data_handle());
    return data_;
  }

  const real* getData() const override {
    CHECK_EQ((void*)data_, get_data_handle());
    return data_;
  }

  /**
   * Get primitive descriptor.
   */
  mkldnn::memory::primitive_desc getPrimitiveDesc() {
    return this->get_primitive_desc();
  }

  /**
   * Get memory descriptor.
   */
  mkldnn::memory::desc getMemoryDesc() { return getPrimitiveDesc().desc(); }

  /**
   * Get dimensions.
   */
  mkldnn::memory::dims getDims() {
    mkldnn::memory::desc md = getMemoryDesc();
    const int* src = md.data.dims;
    int ndims = md.data.ndims;
    mkldnn::memory::dims dst;
    dst.resize(ndims);
    for (int i = 0; i < ndims; ++i) {
      dst[i] = src[i];
    }
    return dst;
  }

  /**
   * Get format.
   */
  mkldnn::memory::format getFormat() {
    return (mkldnn::memory::format)(getMemoryDesc().data.format);
  }

  /**
   * Get memory data type.
   */
  mkldnn::memory::data_type getDtype() {
    return (mkldnn::memory::data_type)(getMemoryDesc().data.data_type);
  }

  /**
   * Get engine.
   */
  mkldnn::engine getEngine() { return getPrimitiveDesc().get_engine(); }

 protected:
  /**
   * Do reorder once.
   * Can support inplace.
   */
  void reorderOnce(void* srcData,
                   void* dstData,
                   memory::format srcFmt,
                   memory::format dstFmt,
                   memory::dims dm);
  /**
   * reset this MKLDNN Memory from primitve desc
   */
  void resetMKLDNNMemory(memory::primitive_desc pd, real* data) {
    mkldnn_primitive_t result;
    mkldnn::error::wrap_c_api(
        mkldnn_primitive_create(&result, pd.get(), nullptr, nullptr),
        "could not create a memory primitive");
    reset(result);
    set_data_handle(data);
  }

 private:
  // save the CpuMatrixPtr in case the buffer released outside
  CpuMatrixPtr m_;
};

}  // namespace paddle
