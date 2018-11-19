// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <random>
#include <string>
#include "paddle/fluid/framework/tvm/tvm_runtime.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace tvm {
namespace runtime {

class Timer {
  using TimeType = decltype(std::chrono::high_resolution_clock::now());

 public:
  Timer() { start_ = std::chrono::high_resolution_clock::now(); }

  size_t Cost() {
    end_ = std::chrono::high_resolution_clock::now();
    auto ret = static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_)
            .count());
    return ret;
  }

 private:
  TimeType start_;
  TimeType end_;
};

template <typename T>
void PrepareOutput(const Tensor &x, const Tensor &y, Tensor *z, bool trans_x,
                   bool trans_y,
                   operators::math::MatDescriptor *x_desc = nullptr,
                   operators::math::MatDescriptor *y_desc = nullptr) {
  PADDLE_ENFORCE(x.place() == y.place());
  auto dim_x = x.dims();
  auto dim_y = y.dims();
  auto mat_dim_x = operators::math::CreateMatrixDescriptor(dim_x, 0, trans_x);
  auto mat_dim_y = operators::math::CreateMatrixDescriptor(dim_y, 0, trans_y);
  PADDLE_ENFORCE_EQ(mat_dim_x.width_, mat_dim_y.height_);
  PADDLE_ENFORCE(mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
                 mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0);
  std::vector<int64_t> dim_out;
  if (mat_dim_x.batch_size_ != 0) {
    dim_out = vectorize(dim_x);
    dim_out[dim_out.size() - 2] = mat_dim_x.height_;
    dim_out[dim_out.size() - 1] = mat_dim_y.width_;
  } else if (mat_dim_y.batch_size_ != 0) {
    dim_out = vectorize(dim_y);
    dim_out[dim_out.size() - 2] = mat_dim_x.height_;
    dim_out[dim_out.size() - 1] = mat_dim_y.width_;
  } else {
    dim_out = {mat_dim_x.height_, mat_dim_y.width_};
  }

  if (dim_x.size() == 1 && dim_out[dim_out.size() - 2] == 1) {
    std::swap(dim_out[dim_out.size() - 2], dim_out[dim_out.size() - 1]);
    dim_out.resize(dim_out.size() - 1);
  }

  if (dim_y.size() == 1 && dim_out[dim_out.size() - 1] == 1) {
    dim_out.resize(dim_out.size() - 1);
  }

  if (dim_out.empty()) {
    dim_out = {1};
  }

  z->Resize(make_ddim(dim_out));
  z->mutable_data<T>(x.place());

  if (x_desc != nullptr) {
    *x_desc = mat_dim_x;
  }

  if (y_desc != nullptr) {
    *y_desc = mat_dim_y;
  }
}

template <typename T>
struct CuBlasBatchMatMul {
  CuBlasBatchMatMul(const platform::CUDADeviceContext &ctx, const Tensor &x,
                    const Tensor &y, Tensor *z, bool trans_x, bool trans_y)
      : x_(x),
        y_(y),
        z_(z),
        blas_(operators::math::GetBlas<platform::CUDADeviceContext, T>(ctx)) {
    PrepareOutput<T>(x_, y_, z_, trans_x, trans_y, &x_desc_, &y_desc_);
  }

  void Run() {
    blas_.MatMul(x_, x_desc_, y_, y_desc_, static_cast<T>(1), z_,
                 static_cast<T>(0));
  }

 private:
  const Tensor &x_;
  const Tensor &y_;
  Tensor *z_;
  operators::math::MatDescriptor x_desc_;
  operators::math::MatDescriptor y_desc_;
  operators::math::BlasT<platform::CUDADeviceContext, T> blas_;
};

template <typename T>
struct TVMBatchMatMul {
  TVMBatchMatMul(const std::string &lib_name, const std::string &func_name,
                 const platform::CUDADeviceContext &ctx, const Tensor &x,
                 const Tensor &y, Tensor *z, bool trans_x, bool trans_y)
      : x_(x), y_(y) {
    PrepareOutput<T>(x, y, z, trans_x, trans_y, nullptr, nullptr);
    func_ = GetFuncFromLib(lib_name, func_name);
    z_.reset(new DLPackTensor(*z));
    SetStream(boost::get<platform::CUDAPlace>(ctx.GetPlace()).device,
              ctx.stream());
  }

  void Run() { func_(x_, y_, *z_); }

 private:
  PackedFunc func_;
  DLPackTensor x_, y_;
  std::unique_ptr<DLPackTensor> z_;
};

template <typename Runnable>
size_t Run(const platform::CUDADeviceContext &ctx, Runnable &&runnable, int n) {
  runnable.Run();
  ctx.Wait();
  Timer timer;
  for (int i = 0; i < n; ++i) runnable.Run();
  ctx.Wait();
  return timer.Cost();
}

template <typename T>
void FillRandom(T *x, int64_t n, const platform::Place &place) {
  std::vector<T> vec;
  auto *cpu_x = x;
  if (platform::is_gpu_place(place)) {
    vec.resize(n);
    cpu_x = vec.data();
  }

  std::random_device rnd;
  std::uniform_real_distribution<T> dist(2.0, 10.0);
  auto gen = [&] { return dist(rnd); };
  std::generate(cpu_x, cpu_x + n, gen);
  if (cpu_x != x) {
    cudaMemcpy(x, cpu_x, n * sizeof(T), cudaMemcpyHostToDevice);
  }
}

template <typename T>
Tensor ConvertToCpuTensor(const Tensor &tensor) {
  if (platform::is_cpu_place(tensor.place())) return tensor;
  Tensor cpu_tensor;
  cpu_tensor.mutable_data<T>(tensor.dims(), platform::CPUPlace());
  cudaMemcpy(cpu_tensor.data<T>(), tensor.data<T>(), tensor.numel() * sizeof(T),
             cudaMemcpyDeviceToHost);
  return cpu_tensor;
}

template <typename T>
Tensor GetActualMatMulValue(const Tensor &x_tensor_in,
                            const Tensor &y_tensor_in) {
  Tensor x_tensor = ConvertToCpuTensor<T>(x_tensor_in);
  Tensor y_tensor = ConvertToCpuTensor<T>(y_tensor_in);
  Tensor z_tensor;
  PrepareOutput<T>(x_tensor, y_tensor, &z_tensor, false, false);

  const T *x = x_tensor.data<T>();
  const T *y = y_tensor.data<T>();
  T *z = z_tensor.data<T>();

  int M = x_tensor.dims()[x_tensor.dims().size() - 2];
  int K = x_tensor.dims()[x_tensor.dims().size() - 1];
  int B = x_tensor.numel() / M / K;
  int N = y_tensor.dims()[y_tensor.dims().size() - 1];

  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        T tmp = 0;
        for (int k = 0; k < K; ++k) {
          auto idx_x = b * (M * K) + m * K + k;
          auto idx_y = b * (K * N) + k * N + n;
          tmp += x[idx_x] * y[idx_y];
        }
        auto idx_z = b * (M * N) + m * N + n;
        z[idx_z] = tmp;
      }
    }
  }
  return z_tensor;
}

template <typename T>
T MaxDiff(const Tensor &tensor1_in, const Tensor &tensor2_in) {
  Tensor tensor1 = ConvertToCpuTensor<T>(tensor1_in);
  Tensor tensor2 = ConvertToCpuTensor<T>(tensor2_in);
  if (tensor1.dims() == tensor2.dims()) {
    auto *x = tensor1.data<T>();
    auto *y = tensor2.data<T>();
    int64_t n = tensor1.numel();
    T max_diff = 0;
    for (int64_t i = 0; i < n; ++i) {
      max_diff = std::max(std::abs(x[i] - y[i]), max_diff);
    }
    return max_diff;
  } else {
    return static_cast<T>(-1);
  }
}

template <typename T>
bool TestMain(int batch_size, int feature_size, int M, int K, int N,
              bool trans_x = false, bool trans_y = false,
              int iterations = 100) {
  platform::CUDAPlace place(0);
  platform::CUDADeviceContext ctx(place);

  Tensor x, y;
  if (trans_x) {
    x.Resize({batch_size, feature_size, K, M});
  } else {
    x.Resize({batch_size, feature_size, M, K});
  }
  x.mutable_data<T>(place);
  FillRandom(x.data<T>(), x.numel(), place);

  if (trans_y) {
    y.Resize({batch_size, feature_size, N, K});
  } else {
    y.Resize({batch_size, feature_size, K, N});
  }
  y.mutable_data<T>(place);
  FillRandom(y.data<T>(), y.numel(), place);

  Tensor z1, z2;
  Tensor z_actual = GetActualMatMulValue<T>(x, y);
  {
    CuBlasBatchMatMul<T> cublas_matmul(ctx, x, y, &z1, trans_x, trans_y);
    std::cout << "CuBlas Prepared" << std::endl;
    auto time = Run(ctx, cublas_matmul, iterations);
    std::cout << "CuBlas Time: " << time / static_cast<double>(iterations)
              << "us" << std::endl;
  }

  {
    std::string lib_name = "tvm_batch_matmul_" + std::to_string(N) + "_" +
                           std::to_string(K) + ".so";
    TVMBatchMatMul<T> tvm_matmul(lib_name, "FP32_matmul", ctx, x, y, &z2,
                                 trans_x, trans_y);
    std::cout << "TVM Prepared" << std::endl;
    auto time = Run(ctx, tvm_matmul, iterations);
    std::cout << "TVM Time: " << time / static_cast<double>(iterations) << "us"
              << std::endl;
  }

  std::cout << "Diff1: " << MaxDiff<T>(z1, z_actual) << std::endl;
  std::cout << "Diff2: " << MaxDiff<T>(z2, z_actual) << std::endl;
  return true;
}

TEST(compare_cublas_tvm, matmul_K_17_N_128) {
  ASSERT_TRUE(TestMain<float>(64, 8, 1, 17, 128));
}

TEST(compare_cublas_tvm, matmul_K_128_N_17) {
  ASSERT_TRUE(TestMain<float>(64, 8, 1, 128, 17));
}

}  // namespace runtime
}  // namespace tvm
}  // namespace framework
}  // namespace paddle
