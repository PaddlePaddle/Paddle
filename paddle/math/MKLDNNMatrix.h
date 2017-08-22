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

static const std::map<mkldnn::memory::format, PARAM_FORMAT> PARAM_FOARMAT_MAP =
    {{mkldnn::memory::format::oi, PARAM_FORMAT_MKLDNN_OI}};

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

  MKLDNNMatrix(size_t height, size_t width, mkldnn::memory::primitive_desc pd)
      : CpuMatrix(height, width, false), mkldnn::memory(pd) {
    set_data_handle(CpuMatrix::getData());
  }

  ~MKLDNNMatrix() {}

  static MKLDNNMatrixPtr create(
      const MatrixPtr& m,
      mkldnn::memory::dims dims,
      mkldnn::memory::format fmt,
      mkldnn::engine& eg,
      mkldnn::memory::data_type dtype = mkldnn::memory::data_type::f32);

  /**
   * Get primitive descriptor.
   */
  mkldnn::memory::primitive_desc getPD() { return this->get_primitive_desc(); }

  /**
   * Get memory descriptor.
   */
  mkldnn::memory::desc getMD() { return getPD().desc(); }

  /**
   * Get dims.
   */
  mkldnn::memory::dims getDims() {
    mkldnn::memory::dims dst;
    int* src = getMD().data.dims;
    int ndims = getMD().data.ndims;
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
   * Update the memory data handle.
   * Caution: This will not check the buffer size of the data,
   *          it should be coverd by user.
   */
  void updateData(void* data) { set_data_handle(data); }
};

}  // namespace paddle
