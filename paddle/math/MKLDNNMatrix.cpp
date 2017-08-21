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

MKLDNNMatrixPtr MKLDNNMatrix::create(const MatrixPtr& m,
                                     memory::dims dims,
                                     memory::format fmt,
                                     engine& eg,
                                     mkldnn::memory::data_type dtype) {
  CpuMatrixPtr cpuM = std::dynamic_pointer_cast<CpuMatrix>(m);
  CHECK(cpuM) << "Only support create from CPU matrix yet";

  size_t ndims = dims.size();
  CHECK(ndims > 0) << "Input dims should not be empty";
  size_t cnt = 1;
  for (size_t i = 0; i < ndims; ++i) {
    cnt *= dims[i];
  }
  CHECK_EQ(cnt, m->getElementCnt()) << "Count size does not match";

  size_t width = m->getWidth();
  size_t height = m->getHeight();
  real* data = m->getData();

  memory::desc md = memory::desc(dims, dtype, fmt);
  memory::primitive_desc pd = memory::primitive_desc(md, eg);
  return std::make_shared<MKLDNNMatrix>(data, height, width, pd);
}

}  // namespace paddle
