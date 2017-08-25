/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

/**
 * @brief MKLDNN Matrix.
 *
 */
class MKLDNNMatrix : public CpuMatrix, public mkldnn::memory {
public:
  MKLDNNMatrix(real* data,
               size_t height,
               size_t width,
               mkldnn::memory::primitive_desc pd)
      : CpuMatrix(data, height, width, false), mkldnn::memory(pd, data) {}

  ~MKLDNNMatrix() {}

  /**
   * Create MKLDNNMatrix from a MatrixPtr and memory primitive_desc
   */
  static MKLDNNMatrixPtr create(MatrixPtr m, mkldnn::memory::primitive_desc pd);

  /**
   * Create MKLDNNMatrix from a MatrixPtr and memory details info
   */
  static MKLDNNMatrixPtr create(
      MatrixPtr m,
      mkldnn::memory::dims dims,
      mkldnn::memory::format fmt,
      mkldnn::engine& eg,
      mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32);

public:
  /**
   * Reorder this MKLDNNMatrix from other format.
   * Support inplace reorder
   * Pay attention: this function would only reorder the data layout.
   *                will NOT change this original dim or format info
   */
  void reorderDataFrom(const MKLDNNMatrixPtr& m,
                       memory::format srcFmt,
                       memory::dims targetDim);

  /**
   * Reorder this MKLDNNMatrix to other format.
   * Support inplace reorder
   * Pay attention: this function would only reorder the data layout.
   *                will NOT change the dst dim or format info
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
   * Update the memory data handle.
   * Caution: This will not check the buffer size of the data,
   *          it should be coverd by user.
   */
  void updateData(void* data) { set_data_handle(data); }

  /**
   * Get primitive descriptor.
   */
  mkldnn::memory::primitive_desc getPD() { return this->get_primitive_desc(); }

  /**
   * Get memory descriptor.
   */
  mkldnn::memory::desc getMD() { return getPD().desc(); }

  /**
   * Get dimensions.
   */
  mkldnn::memory::dims getDims() {
    mkldnn::memory::desc md = getMD();
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
    return (mkldnn::memory::format)(getMD().data.format);
  }

  /**
   * Get memory data type.
   */
  mkldnn::memory::data_type getDtype() {
    return (mkldnn::memory::data_type)(getMD().data.data_type);
  }

  /**
   * Get engine.
   */
  mkldnn::engine getEngine() { return getPD().get_engine(); }

protected:
  /**
   * Do once reorder supported inplace.
   */
  void reorderOnce(void* srcData,
                   void* dstData,
                   memory::format srcFmt,
                   memory::format dstFmt,
                   memory::dims dm);
};

}  // namespace paddle
