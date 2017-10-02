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

#include <gtest/gtest.h>
#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/hostdevice.h"
#include "paddle/platform/transform.h"

template <typename T>
class Scale {
 public:
  explicit Scale(const T& scale) : scale_(scale) {}

  HOSTDEVICE T operator()(const T& a) const { return a * scale_; }

 private:
  T scale_;
};

template <typename T>
class Multiply {
 public:
  HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};

TEST(Transform, CPUUnary) {
  using namespace paddle::platform;
  CPUDeviceContext ctx;
  float buf[4] = {0.1, 0.2, 0.3, 0.4};
  Transform<paddle::platform::CPUPlace> trans;
  trans(ctx, buf, buf + 4, buf, Scale<float>(10));
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(buf[i], static_cast<float>(i + 1), 1e-5);
  }
}

TEST(Transform, GPUUnary) {
  using namespace paddle::platform;
  using namespace paddle::memory;
  GPUPlace gpu0(0);
  CUDADeviceContext ctx(gpu0);
  float cpu_buf[4] = {0.1, 0.2, 0.3, 0.4};
  float* gpu_buf = static_cast<float*>(Alloc(gpu0, sizeof(float) * 4));
  Copy(gpu0, gpu_buf, CPUPlace(), cpu_buf, sizeof(cpu_buf));
  Transform<paddle::platform::GPUPlace> trans;
  trans(ctx, gpu_buf, gpu_buf + 4, gpu_buf, Scale<float>(10));
  ctx.Wait();
  Copy(CPUPlace(), cpu_buf, gpu0, gpu_buf, sizeof(cpu_buf));
  Free(gpu0, gpu_buf);
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(cpu_buf[i], static_cast<float>(i + 1), 1e-5);
  }
}

TEST(Transform, CPUBinary) {
  using namespace paddle::platform;
  using namespace paddle::memory;
  int buf[4] = {1, 2, 3, 4};
  Transform<paddle::platform::CPUPlace> trans;
  CPUDeviceContext ctx;
  trans(ctx, buf, buf + 4, buf, buf, Multiply<int>());
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ((i + 1) * (i + 1), buf[i]);
  }
}

TEST(Transform, GPUBinary) {
  using namespace paddle::platform;
  using namespace paddle::memory;
  int buf[4] = {1, 2, 3, 4};
  GPUPlace gpu0(0);
  CUDADeviceContext ctx(gpu0);
  int* gpu_buf = static_cast<int*>(Alloc(gpu0, sizeof(buf)));
  Copy(gpu0, gpu_buf, CPUPlace(), buf, sizeof(buf));
  Transform<paddle::platform::GPUPlace> trans;
  trans(ctx, gpu_buf, gpu_buf + 4, gpu_buf, gpu_buf, Multiply<int>());
  ctx.Wait();
  Copy(CPUPlace(), buf, gpu0, gpu_buf, sizeof(buf));
  Free(gpu0, gpu_buf);
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ((i + 1) * (i + 1), buf[i]);
  }
}
