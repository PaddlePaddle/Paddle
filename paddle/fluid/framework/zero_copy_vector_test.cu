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

#include <cuda_runtime.h>
#include <stdlib.h>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/zero_copy_vector.h"
#include "paddle/fluid/platform/profiler.h"

static __global__ void AddKernel(float *z, const float *x, const float *y,
                                 const size_t *lod, int seq_width) {
  int block_dim_x = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int seq_start = lod[bid];
  int seq_end = lod[bid + 1];
  float y_elem = y[bid];
  for (int offset = seq_start; offset < seq_end; offset++) {
    for (int i = tid; i < seq_width; i += block_dim_x) {
      z[offset * seq_width + i] = x[offset * seq_width + i] + y_elem;
    }
  }
}

class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

template <typename DeviceContext, typename Place, typename VectorType>
void TestVector(VectorType *lod, const size_t seq_width, const size_t scale) {
  paddle::framework::Tensor cpu_x;
  paddle::framework::Tensor cpu_y;
  paddle::framework::Tensor cpu_z;
  paddle::framework::Tensor x;
  paddle::framework::Tensor y;
  paddle::framework::Tensor z;

  const size_t batch_size = lod->size() - 1;

  auto x_dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(lod->back()), static_cast<int64_t>(seq_width)});
  auto *cpu_x_data =
      cpu_x.mutable_data<float>(x_dims, paddle::platform::CPUPlace());
  for (int64_t i = 0; i < cpu_x.numel(); ++i) {
    cpu_x_data[i] = static_cast<float>(i) * 0.01;
  }

  auto y_dims =
      paddle::framework::make_ddim({static_cast<int64_t>(batch_size), 1});
  auto cpu_y_data =
      cpu_y.mutable_data<float>(y_dims, paddle::platform::CPUPlace());
  for (int64_t i = 0; i < cpu_y.numel(); ++i) {
    cpu_y_data[i] = static_cast<float>(i) * 10000;
  }

  auto *place = new Place();
  DeviceContext *context = new DeviceContext(*place);
  if (paddle::platform::is_cpu_place(*place)) {
    x = cpu_x;
    y = cpu_y;
  } else {
    TensorCopySync(cpu_x, *place, &x);
    TensorCopySync(cpu_y, *place, &y);
  }

  auto *x_data = x.data<float>();
  auto *y_data = y.data<float>();
  auto *z_data = z.mutable_data<float>(x_dims, *place);

  paddle::platform::ProfilerState state = paddle::platform::ProfilerState::kAll;
  paddle::platform::SetDeviceId(0);
  paddle::platform::EnableProfiler(state);

  double latency = 0.0f;
  {
    paddle::platform::RecordEvent record_event("add_kernel", context);

    Timer timer;
    timer.tic();

    for (int r = 0; r < 1000; ++r) {
      AddKernel<<<batch_size, 1024, 0, context->stream()>>>(
          z_data, x_data, y_data, lod->CUDAData(*place), seq_width);

      // This code ensure lod will be transfered from CPU to GPU next iteration
      // when VectorType is Vector.
      for (size_t i = 0; i < lod->size(); ++i) {
        (*lod)[i] = (*lod)[i] * scale;  // scale is 1
      }
    }
    latency = timer.toc();
  }
  context->Wait();

  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kDefault,
                                    "test_profiler");
  LOG(INFO) << "Runtime: " << latency << " ms";

  if (paddle::platform::is_cpu_place(*place)) {
    cpu_z = z;
  } else {
    TensorCopySync(z, paddle::platform::CPUPlace(), &cpu_z);
  }

  auto *cpu_z_data = cpu_z.data<float>();
  for (size_t i = 0; i < batch_size; ++i) {
    size_t seq_start = (*lod)[i];
    size_t seq_end = (*lod)[i + 1];
    for (size_t offset = seq_start; offset < seq_end; ++offset) {
      for (int64_t j = 0; j < seq_width; ++j) {
        ASSERT_EQ(cpu_z_data[offset * seq_width + j],
                  cpu_x_data[offset * seq_width + j] + cpu_y_data[i]);
      }
    }
  }

  delete place;
  delete context;
}

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &vec) {
  os << "{";
  bool is_first = true;
  for (auto &i : vec) {
    if (is_first) {
      os << i;
      is_first = false;
    } else {
      os << ", " << i;
    }
  }
  os << "}";

  return os;
}

std::vector<size_t> GenerateRadomLoD(int batch_size) {
  static unsigned int seed = 100;
  srand(seed);

  std::vector<size_t> lod;
  lod.resize(batch_size + 1);
  lod[0] = 0;
  for (int i = 0; i < batch_size; ++i) {
    lod[i + 1] = lod[i] + rand_r(&seed) % 9 + 1;
  }

  return lod;
}

TEST(ZeroCopyVector, profile) {
  for (auto batch_size : {2, 32, 256}) {
    std::vector<size_t> vec = GenerateRadomLoD(batch_size);

    for (auto seq_width : {32}) {
      LOG(INFO) << "======== Use ZeroCopyVector, seq_width: " << seq_width
                << ", batch_size: " << batch_size << " ========";
      paddle::framework::ZeroCopyVector<size_t> lod1(vec);
      TestVector<paddle::platform::CUDADeviceContext,
                 paddle::platform::CUDAPlace,
                 paddle::framework::ZeroCopyVector<size_t>>(&lod1, seq_width,
                                                            1U);

      LOG(INFO) << "======== Use MixedVector, seq_width: " << seq_width
                << ", batch_size: " << batch_size << " ========";
      paddle::framework::Vector<size_t> lod2(vec);
      TestVector<paddle::platform::CUDADeviceContext,
                 paddle::platform::CUDAPlace,
                 paddle::framework::Vector<size_t>>(&lod2, seq_width, 1U);
    }
  }
}
