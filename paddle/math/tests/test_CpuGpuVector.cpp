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

#ifndef PADDLE_ONLY_CPU

#include <gtest/gtest.h>
#include "paddle/math/Vector.h"
#include "paddle/utils/Util.h"
#include "test_matrixUtil.h"

using namespace paddle;  // NOLINT

TEST(CpuGpuVector, getData) {
  size_t size = 500;
  hl_stream_t stream(HPPL_STREAM_DEFAULT);
  CpuVectorPtr cpuVec = std::make_shared<CpuVector>(size);
  GpuVectorPtr gpuVec = std::make_shared<GpuVector>(size);
  cpuVec->uniform(0.0, 10.0);
  gpuVec->copyFrom(*cpuVec, stream);
  hl_stream_synchronize(stream);

  CpuGpuVectorPtr vec = std::make_shared<CpuGpuVector>(gpuVec);
  auto a = vec->getData(false);
  auto b = cpuVec->getData();
  hl_stream_synchronize(stream);
  checkDataEqual(a, b, size);
}

TEST(CpuGpuVector, subCreate) {
  size_t size1 = 1024;
  size_t offset = 100;
  size_t size2 = 500;
  hl_stream_t stream(HPPL_STREAM_DEFAULT);
  CpuGpuVectorPtr v1 = std::make_shared<CpuGpuVector>(size1, /*useGpu*/ false);
  auto vec = v1->getMutableVector(false);
  vec->uniform(0.0, 10.0);
  auto v2 = std::make_shared<CpuGpuVector>(*v1, offset, size2);
  CHECK_EQ(*v1->getSync(), *v2->getSync());

  // check subVec equal
  checkDataEqual(v1->getData(false) + offset, v2->getData(false), size2);

  CpuVectorPtr v1Check = std::make_shared<CpuVector>(size1);
  CpuVectorPtr v2Check = std::make_shared<CpuVector>(size2);
  v1Check->copyFrom(*(v1->getVector(true)), stream);
  v2Check->copyFrom(*(v2->getVector(true)), stream);
  hl_stream_synchronize(stream);

  checkDataEqual(v2->getData(false), v2Check->getData(), size2);
  checkDataEqual(v1Check->getData() + offset, v2Check->getData(), size2);

  CpuVectorPtr noise = std::make_shared<CpuVector>(size2);
  noise->uniform(0.0, 1.0);
  auto v = v2->getMutableVector(false);  // will change header
  // add noise to subVec
  v->add(*noise);

  // check v1_cpu_data == v2_cpu_data
  checkDataEqual(v1->getData(false) + offset, v2->getData(false), size2);

  v1Check->copyFrom(*(v1->getVector(true)), stream);
  v2Check->copyFrom(*(v2->getVector(true)), stream);
  hl_stream_synchronize(stream);

  // check v1_gpu_data == v2_gpu_data
  checkDataEqual(v1Check->getData() + offset, v2Check->getData(), size2);
}

#endif
