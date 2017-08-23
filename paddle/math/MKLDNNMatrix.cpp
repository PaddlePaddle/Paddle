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

#include "MKLDNNMatrix.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

MKLDNNMatrixPtr MKLDNNMatrix::create(MatrixPtr m, memory::primitive_desc pd) {
  memory::desc md = pd.desc();
  size_t ndims = md.data.ndims;
  int* dims = md.data.dims;
  CHECK(ndims > 0) << "Input dims should not be empty";
  size_t cnts = 1;
  for (size_t i = 0; i < ndims; ++i) {
    cnts *= dims[i];
  }

  if (m == nullptr) {
    size_t height = dims[0];
    size_t width = cnts / dims[0];
    // LOG(INFO) << height << "," << width;
    m = Matrix::create(height, width, false, false);
  }

  CHECK(m) << " Matrix should not be empty";
  CpuMatrixPtr cpuMatrix = std::dynamic_pointer_cast<CpuMatrix>(m);
  CHECK(cpuMatrix) << "Only support create from CPU matrix yet";

  CHECK_EQ(cnts, m->getElementCnt()) << "Count size does not match";
  size_t width = m->getWidth();
  size_t height = m->getHeight();
  real* data = m->getData();
  return std::make_shared<MKLDNNMatrix>(data, height, width, pd);
}

MKLDNNMatrixPtr MKLDNNMatrix::create(MatrixPtr m,
                                     memory::dims dims,
                                     memory::format fmt,
                                     engine& eg,
                                     mkldnn::memory::data_type dtype) {
  memory::desc md = memory::desc(dims, dtype, fmt);
  memory::primitive_desc pd = memory::primitive_desc(md, eg);
  return create(m, pd);
}

void MKLDNNMatrix::downSpatial() {
  int fmt = getFormat();
  if (!(fmt == memory::format::nchw || fmt == memory::format::oihw)) {
    // only support nchw and oihw yet, later can support more like nhwc, ihwo
    return;
  }

  memory::dims srcDims = getDims();
  const int H = 2, W = 3;
  if (srcDims[H] != 1 || srcDims[W] != 1) {
    // can not down spatial
    return;
  }

  memory::dims dstDims = memory::dims{srcDims[0], srcDims[1]};
  memory::format dstFmt;
  switch (fmt) {
    case memory::format::nchw:
      dstFmt = memory::format::nc;
      break;
    case memory::format::oihw:
      dstFmt = memory::format::oi;
      break;
    default:
      LOG(FATAL) << "unsupported format";
  }
  memory::desc md = memory::desc(dstDims, getDtype(), dstFmt);
  memory::primitive_desc pd = memory::primitive_desc(md, getEngine());
  void* data = getData();
  mkldnn_primitive_t result;
  mkldnn::error::wrap_c_api(
      mkldnn_primitive_create(&result, pd.get(), nullptr, nullptr),
      "could not create a memory primitive");
  reset(result);
  set_data_handle(data);
}

}  // namespace paddle
