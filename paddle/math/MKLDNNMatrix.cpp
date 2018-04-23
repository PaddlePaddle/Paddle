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

#include "MKLDNNMatrix.h"

using namespace mkldnn;  // NOLINT

namespace paddle {

MKLDNNMatrixPtr MKLDNNMatrix::create(memory::primitive_desc pd, MatrixPtr m) {
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
    m = Matrix::create(height, width, false, false);
  }
  CHECK(m) << " Matrix should not be empty";

  CpuMatrixPtr cpuMatrix = std::dynamic_pointer_cast<CpuMatrix>(m);
  CHECK(cpuMatrix) << "Only support create from CPU matrix yet";
  CHECK_EQ(cpuMatrix->getElementCnt(), cnts) << "Count size does not match";
  return std::make_shared<MKLDNNMatrix>(cpuMatrix, pd);
}

MKLDNNMatrixPtr MKLDNNMatrix::create(memory::dims dims,
                                     memory::format fmt,
                                     engine& eg,
                                     MatrixPtr m,
                                     mkldnn::memory::data_type dtype) {
  return create(createPrimitiveDesc(dims, fmt, eg, dtype), m);
}

std::shared_ptr<reorder> MKLDNNMatrix::createReorder(const MKLDNNMatrixPtr& src,
                                                     const MKLDNNMatrixPtr& dst,
                                                     bool checkData) {
  if (src == dst || src->getPrimitiveDesc() == dst->getPrimitiveDesc()) {
    return nullptr;
  }

  if (checkData && (src->getData() == dst->getData())) {
    LOG(FATAL) << "can not create reorder with inplace data";
    return nullptr;
  }

  memory::dims srcDims = src->getDims();
  memory::dims dstDims = dst->getDims();
  CHECK_EQ(srcDims.size(), dstDims.size());
  for (size_t i = 0; i < srcDims.size(); ++i) {
    CHECK_EQ(srcDims[i], dstDims[i]);
  }
  return std::make_shared<reorder>(*src, *dst);
}

void MKLDNNMatrix::reorderDataFrom(const MKLDNNMatrixPtr& m,
                                   memory::format srcFmt,
                                   memory::dims targetDim) {
  memory::format dstFmt = getFormat();
  if (srcFmt == dstFmt) {
    return;
  }
  CHECK_EQ(getElementCnt(), m->getElementCnt()) << "size should equal";
  reorderOnce(getData(), m->getData(), srcFmt, dstFmt, targetDim);
}

void MKLDNNMatrix::reorderDataTo(const MKLDNNMatrixPtr& m,
                                 memory::format dstFmt,
                                 memory::dims targetDim) {
  memory::format srcFmt = getFormat();
  if (srcFmt == dstFmt) {
    return;
  }
  CHECK_EQ(getElementCnt(), m->getElementCnt()) << "size should equal";
  reorderOnce(getData(), m->getData(), srcFmt, dstFmt, targetDim);
}

void MKLDNNMatrix::reorderOnce(void* srcData,
                               void* dstData,
                               memory::format srcFmt,
                               memory::format dstFmt,
                               memory::dims dm) {
  CHECK(srcData);
  CHECK(dstData);
  MatrixPtr tmpSrc;
  if (dstData == srcData) {
    // inplace data
    size_t sz = 1;
    for (size_t i = 0; i < dm.size(); ++i) {
      sz *= dm[i];
    }
    tmpSrc = Matrix::create(sz, 1, false, false);
    tmpSrc->copyFrom((real*)srcData, sz);
    srcData = tmpSrc->getData();
  }

  auto dtype = this->getDtype();
  auto srcMD = memory::desc(dm, dtype, srcFmt);
  auto dstMD = memory::desc(dm, dtype, dstFmt);

  auto eg = this->getEngine();
  auto src = memory(memory::primitive_desc(srcMD, eg), srcData);
  auto dst = memory(memory::primitive_desc(dstMD, eg), dstData);

  auto r = reorder(src, dst);
  stream(stream::kind::eager).submit({r}).wait();
}

void MKLDNNMatrix::downSpatial() {
  int fmt = getFormat();
  if (!(fmt == memory::format::nchw || fmt == memory::format::oihw)) {
    // only support nchw and oihw yet, later can support more like nhwc, ihwo
    return;
  }

  // TODO(TJ): change H(height) and W(width) if support nhwc or more
  const int H = 2, W = 3;
  memory::dims srcDims = getDims();
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
  resetMKLDNNMemory(pd, data_);
}

}  // namespace paddle
