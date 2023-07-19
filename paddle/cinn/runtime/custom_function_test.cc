// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <functional>
#include <sstream>
#include <vector>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>

#include "paddle/cinn/runtime/cuda/cuda_util.h"
#endif

#ifdef CINN_WITH_MKL_CBLAS
#include "paddle/cinn/runtime/cpu/cblas.h"
#endif

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/custom_function.h"

namespace cinn {
namespace runtime {

class CinnBufferAllocHelper {
 public:
  CinnBufferAllocHelper(cinn_device_kind_t device,
                        cinn_type_t type,
                        const std::vector<int>& shape,
                        int align = 0) {
    buffer_ = cinn_buffer_t::new_(device, type, shape, align);
  }

  template <typename T>
  T* mutable_data(const Target& target) {
    if (target_ != common::UnkTarget()) {
      CHECK_EQ(target, target_)
          << "Cannot alloc twice, the memory had alloced at " << target_
          << "! Please check.";
      return reinterpret_cast<T*>(buffer_->memory);
    }

    target_ = target;
    if (target == common::DefaultHostTarget()) {
      cinn_buffer_malloc(nullptr, buffer_);
    } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
      cudaMalloc(&buffer_->memory, buffer_->num_elements() * sizeof(T));
#else
      LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! "
                    "Please check.";
#endif
    } else {
      LOG(FATAL) << "Only support nvgpu and cpu, but here " << target
                 << "! Please check.";
    }

    return reinterpret_cast<T*>(buffer_->memory);
  }

  template <typename T>
  const T* data() {
    if (target_ == common::UnkTarget()) {
      LOG(FATAL) << "No memory had alloced! Please check.";
    }
    return reinterpret_cast<const T*>(buffer_->memory);
  }

  ~CinnBufferAllocHelper() {
    if (buffer_) {
      if (target_ == common::UnkTarget()) {
        // pass
      } else if (target_ == common::DefaultHostTarget()) {
        cinn_buffer_free(nullptr, buffer_);
      } else if (target_ == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
        cudaFree(buffer_->memory);
#else
        LOG(FATAL) << "NVGPU Target only support on flag CINN_WITH_CUDA ON! "
                      "Please check.";
#endif
      } else {
        LOG(FATAL) << "Only support nvgpu and cpu, but here " << target_
                   << "! Please check.";
      }
      delete buffer_;
    }
  }

  cinn_buffer_t& operator*() const noexcept { return *buffer_; }
  cinn_buffer_t* operator->() const noexcept { return buffer_; }
  cinn_buffer_t* get() const noexcept { return buffer_; }

 private:
  cinn_buffer_t* buffer_{nullptr};
  Target target_{common::UnkTarget()};
};

template <typename T>
void SetInputValue(T* input,
                   const T* input_h,
                   size_t num,
                   const Target& target) {
  if (target == common::DefaultHostTarget()) {
    for (int i = 0; i < num; ++i) {
      input[i] = input_h[i];
    }
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    cudaMemcpy(input, input_h, num * sizeof(T), cudaMemcpyHostToDevice);
#else
    LOG(FATAL)
        << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
  }
}

TEST(CinnAssertTrue, test_true) {
  Target target = common::DefaultTarget();

  CinnBufferAllocHelper x(cinn_x86_device, cinn_bool_t(), {1});

  // set inpute value true
  bool input_h = true;
  auto* input = x.mutable_data<bool>(target);

  SetInputValue(input, &input_h, 1, target);

  CinnBufferAllocHelper y(cinn_x86_device, cinn_bool_t(), {1});
  auto* output = y.mutable_data<bool>(target);

  cinn_pod_value_t v_args[2] = {cinn_pod_value_t(x.get()),
                                cinn_pod_value_t(y.get())};

  std::stringstream ss;
  ss << "Test AssertTrue(true) on " << target;
  const auto& msg = ss.str();
  int msg_key = static_cast<int>(std::hash<std::string>()(msg));
  cinn::runtime::utils::AssertTrueMsgTool::GetInstance()->SetMsg(msg_key, msg);
  cinn_assert_true(v_args, 2, msg_key, true, nullptr, target);

  if (target == common::DefaultHostTarget()) {
    ASSERT_EQ(input[0], output[0])
        << "The output of AssertTrue should be the same as input";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    bool output_h = false;
    cudaMemcpy(&output_h, output, sizeof(bool), cudaMemcpyDeviceToHost);

    ASSERT_EQ(input_h, output_h)
        << "The output of AssertTrue should be the same as input";
#endif
  }
}

TEST(CinnAssertTrue, test_false_only_warning) {
  Target target = common::DefaultTarget();

  CinnBufferAllocHelper x(cinn_x86_device, cinn_bool_t(), {1});

  // set inpute value false
  bool input_h = false;
  auto* input = x.mutable_data<bool>(target);

  SetInputValue(input, &input_h, 1, target);

  CinnBufferAllocHelper y(cinn_x86_device, cinn_bool_t(), {1});
  auto* output = y.mutable_data<bool>(target);

  cinn_pod_value_t v_args[2] = {cinn_pod_value_t(x.get()),
                                cinn_pod_value_t(y.get())};

  std::stringstream ss;
  ss << "Test AssertTrue(false, only_warning=true) on " << target;
  const auto& msg = ss.str();
  int msg_key = static_cast<int>(std::hash<std::string>()(msg));
  cinn::runtime::utils::AssertTrueMsgTool::GetInstance()->SetMsg(msg_key, msg);
  cinn_assert_true(v_args, 2, msg_key, true, nullptr, target);

  if (target == common::DefaultHostTarget()) {
    ASSERT_EQ(input[0], output[0])
        << "The output of AssertTrue should be the same as input";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    bool output_h = false;
    cudaMemcpy(&output_h, output, sizeof(bool), cudaMemcpyDeviceToHost);

    ASSERT_EQ(input_h, output_h)
        << "The output of AssertTrue should be the same as input";
#endif
  }
}

TEST(CustomCallGaussianRandom, test_target_nvgpu) {
  Target target = common::DefaultTarget();

  // Arg mean
  float mean = 0.0f;
  // Arg std
  float std = 1.0f;
  // Arg seed
  int seed = 10;

  // Output matrix out
  CinnBufferAllocHelper out(cinn_x86_device, cinn_float32_t(), {2, 3});
  auto* output = out.mutable_data<float>(target);

  int num_args = 1;
  cinn_pod_value_t v_args[1] = {cinn_pod_value_t(out.get())};

  if (target == common::DefaultHostTarget()) {
    LOG(INFO) << "Op gaussian random only support on NVGPU";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    cinn::runtime::cuda::cinn_call_gaussian_random(
        v_args, num_args, mean, std, seed, nullptr);

    float output_data[6] = {0.0};
    cudaMemcpy(output_data, output, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 6; i++) {
      VLOG(6) << output_data[i];
    }
#else
    LOG(FATAL)
        << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
  }
}

TEST(CustomCallUniformRandom, test_target_nvgpu) {
  Target target = common::DefaultTarget();

  // Arg min
  float min = -1.0f;
  // Arg max
  float max = 1.0f;
  // Arg seed
  int seed = 10;

  // Output matrix out
  CinnBufferAllocHelper out(cinn_x86_device, cinn_float32_t(), {2, 3});
  auto* output = out.mutable_data<float>(target);

  int num_args = 1;
  cinn_pod_value_t v_args[1] = {cinn_pod_value_t(out.get())};

  if (target == common::DefaultHostTarget()) {
    LOG(INFO) << "Op uniform random only support on NVGPU";
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    cinn::runtime::cuda::cinn_call_uniform_random(
        v_args, num_args, min, max, seed, nullptr);

    float output_data[6] = {0.0f};
    cudaMemcpy(output_data, output, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 6; i++) {
      VLOG(6) << output_data[i];
    }
#else
    LOG(FATAL)
        << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
  }
}

TEST(CustomCallCholesky, test) {
  Target target = common::DefaultTarget();

  // Batch size
  int batch_size = 1;
  // Dim
  int m = 3;
  // Upper
  bool upper = false;

  // Input matrix x
  CinnBufferAllocHelper x(cinn_x86_device, cinn_float32_t(), {m, m});
  float input_h[9] = {0.96329159,
                      0.88160539,
                      0.40593964,
                      0.88160539,
                      1.39001071,
                      0.48823422,
                      0.40593964,
                      0.48823422,
                      0.19755946};
  auto* input = x.mutable_data<float>(target);
  SetInputValue(input, input_h, m * m, target);

  // Output matrix out
  CinnBufferAllocHelper out(cinn_x86_device, cinn_float32_t(), {m, m});
  auto* output = out.mutable_data<float>(target);

  // Result matrix
  // In the calculation result of MKL, the matrix !upper part is the same as the
  // original input
  float host_result[9] = {0.98147416,
                          0.88160539,
                          0.40593964,
                          0.89824611,
                          0.76365221,
                          0.48823422,
                          0.41360193,
                          0.15284170,
                          0.055967092};
  // In the calculation results of cuSOLVER, the upper and lower triangles of
  // the matrix are the same
  float gpu_result[9] = {0.98147416,
                         0.89824611,
                         0.41360193,
                         0.89824611,
                         0.76365221,
                         0.15284170,
                         0.41360193,
                         0.15284170,
                         0.055967092};

  int num_args = 2;
  cinn_pod_value_t v_args[2] = {cinn_pod_value_t(x.get()),
                                cinn_pod_value_t(out.get())};

  if (target == common::DefaultHostTarget()) {
#ifdef CINN_WITH_MKL_CBLAS
    cinn_call_cholesky_host(v_args, num_args, batch_size, m, upper);
    for (int i = 0; i < batch_size * m * m; i++) {
      ASSERT_NEAR(output[i], host_result[i], 1e-5)
          << "The output of Cholesky should be the same as result";
    }
#else
    LOG(INFO) << "Host Target only support on flag CINN_WITH_MKL_CBLAS ON! "
                 "Please check.";
#endif
  } else if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    cinn::runtime::cuda::cinn_call_cholesky_nvgpu(
        v_args, num_args, batch_size, m, upper);
    std::vector<float> host_output(batch_size * m * m, 0.0f);
    cudaMemcpy(host_output.data(),
               output,
               batch_size * m * m * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size * m * m; i++) {
      ASSERT_NEAR(host_output[i], gpu_result[i], 1e-5)
          << "The output of Cholesky should be the same as result";
    }
#else
    LOG(INFO)
        << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
#endif
  }
}

#ifdef CINN_WITH_CUDA
TEST(CustomCallTriangularSolve, test) {
  Target target = common::DefaultNVGPUTarget();

  int batch_size = 1;
  int m = 3;
  int k = 1;
  bool left_side = true;
  bool upper = true;
  bool transpose_a = false;
  bool unit_diagonal = false;

  double input_a_host[9] = {1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, -1.0};
  double input_b_host[3] = {0.0, -9.0, 5.0};
  CinnBufferAllocHelper a(cinn_x86_device, cinn_float64_t(), {m, m});
  CinnBufferAllocHelper b(cinn_x86_device, cinn_float64_t(), {m, k});
  auto* input_a = a.mutable_data<double>(target);
  auto* input_b = b.mutable_data<double>(target);
  SetInputValue(input_a, input_a_host, m * m, target);
  SetInputValue(input_b, input_b_host, m * k, target);

  // Output matrix out
  CinnBufferAllocHelper out(cinn_x86_device, cinn_float64_t(), {m, k});
  auto* output = out.mutable_data<double>(target);

  // Result matrix res
  double result[3] = {7.0, -2.0, -5.0};

  constexpr int num_args = 3;
  cinn_pod_value_t v_args[num_args] = {cinn_pod_value_t(a.get()),
                                       cinn_pod_value_t(b.get()),
                                       cinn_pod_value_t(out.get())};
  cinn::runtime::cuda::cinn_call_triangular_solve_nvgpu(v_args,
                                                        num_args,
                                                        batch_size,
                                                        m,
                                                        k,
                                                        left_side,
                                                        upper,
                                                        transpose_a,
                                                        unit_diagonal);
  std::vector<double> device_output(batch_size * m * k, 0.0f);
  cudaMemcpy(device_output.data(),
             output,
             batch_size * m * k * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < batch_size * m * k; i++) {
    ASSERT_NEAR(device_output[i], result[i], 1e-5)
        << "The output of triangular solve should be the same as result";
  }
}
#endif

}  // namespace runtime
}  // namespace cinn
