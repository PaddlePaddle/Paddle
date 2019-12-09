/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/variant.h"  // for UNUSED

DEFINE_int32(burning, 10, "Burning times.");
DEFINE_int32(repeat, 3000, "Repeat times.");
DEFINE_int32(max_size, 1000, "The Max size would be tested.");
DEFINE_string(filter, "", "The Benchmark name would be run.");

class BenchJITKernel {
 public:
  BenchJITKernel() = default;
  virtual ~BenchJITKernel() = default;
  virtual void Run() = 0;
  virtual const char* Name() = 0;
  virtual const char* Dtype() = 0;
  virtual const char* Place() = 0;
};

static std::vector<BenchJITKernel*> g_all_benchmarks;

BenchJITKernel* InsertBenchmark(BenchJITKernel* b) {
  g_all_benchmarks.push_back(b);
  return b;
}

#define BENCH_JITKERNEL(name, dtype, place)                                    \
  class BenchJITKernel_##name##_##dtype##_##place##_ : public BenchJITKernel { \
   public:                                                                     \
    const char* Name() override { return #name; }                              \
    const char* Dtype() override { return #dtype; }                            \
    const char* Place() override { return #place; }                            \
    void Run() override;                                                       \
  };                                                                           \
  static auto inserted_##name##_##dtype##_##place##_ UNUSED =                  \
      InsertBenchmark(new BenchJITKernel_##name##_##dtype##_##place##_());     \
  void BenchJITKernel_##name##_##dtype##_##place##_::Run()

void RUN_ALL_BENCHMARK() {
  for (auto p : g_all_benchmarks) {
    if (!FLAGS_filter.empty() && FLAGS_filter != p->Name()) {
      continue;
    }
    LOG(INFO) << "Benchmark " << p->Name() << "." << p->Dtype() << "."
              << p->Place();
    p->Run();
  }
}

template <typename T>
void RandomVec(const int n, T* a, const T lower = static_cast<T>(-20.f),
               const T upper = static_cast<T>(20.f), unsigned int seed = 100) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

std::vector<int> TestSizes() {
  std::vector<int> s;
  for (int i = 1; i <= FLAGS_max_size; ++i) {
    s.push_back(i);
  }
  return s;
}

std::vector<int> TestMatmul() {
  std::vector<int> s;
  s.push_back(100);
  s.push_back(128);
  s.push_back(500);
  s.push_back(768);
  s.push_back(1000);
  s.push_back(1536);
  s.push_back(3072);
  return s;
}

template <typename KernelTuple, typename... Args>
struct BenchFunc {
  // return this function avg time
  // TODO(TJ): clear cache every time
  double operator()(const typename KernelTuple::func_type tgt, Args... args) {
    for (int i = 0; i < FLAGS_burning; ++i) {
      tgt(args...);
    }
    auto start = paddle::platform::PosixInNsec() * 1e-3;
    for (int i = 0; i < FLAGS_repeat; ++i) {
      tgt(args...);
    }
    auto end = paddle::platform::PosixInNsec() * 1e-3;
    return static_cast<double>(end - start) / FLAGS_repeat;
  }
};

namespace jit = paddle::operators::jit;

template <typename KernelTuple, typename PlaceType, typename... Args>
void BenchAllImpls(const typename KernelTuple::attr_type& attr, Args... args) {
  BenchFunc<KernelTuple, Args...> benchmark;
  std::vector<std::pair<std::string, double>> infos;
  auto funcs = jit::GetAllCandidateFuncsWithTypes<KernelTuple, PlaceType>(attr);
  for (auto f : funcs) {
    infos.push_back(std::make_pair(f.first, benchmark(f.second, args...)));
  }

  // Test result from Get function
  auto tgt = jit::KernelFuncs<KernelTuple, PlaceType>::Cache().At(attr);
  if (!tgt) {
    LOG(FATAL) << "Target can not be empty!";
  }
  infos.push_back(std::make_pair("Target", benchmark(tgt, args...)));

  // print
  std::ostringstream loginfos;
  loginfos << "Kernel Type " << jit::to_string(KernelTuple::kernel_type) << ": "
           << attr << ": ";
  for (auto pair : infos) {
    loginfos << pair.first << " takes " << pair.second << " us; ";
  }
  LOG(INFO) << loginfos.str();
}

using Tensor = paddle::framework::Tensor;

template <typename KernelTuple, typename PlaceType>
void BenchKernelMatMul() {
  using T = typename KernelTuple::data_type;
  for (int m : {1, 2, 30, 128}) {
    for (int n : TestMatmul()) {
      for (int k : TestMatmul()) {
        Tensor a, b, c;
        a.Resize({m * k});
        b.Resize({k * n});
        c.Resize({m * n});
        RandomVec<T>(m * k, a.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(k * n, b.mutable_data<T>(PlaceType()), -2.f, 2.f);
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();
        T* c_data = c.mutable_data<T>(PlaceType());
        paddle::platform::CPUDeviceContext ctx;
        auto blas =
            paddle::operators::math::GetBlas<paddle::platform::CPUDeviceContext,
                                             float>(ctx);
        if (n % 128 == 0 && k % 128 == 0) {
          Tensor X1, W1, Y1;

          X1.Resize({m * (k + 4)});
          T* X1_data = X1.mutable_data<T>(PlaceType());
          W1.Resize({(k + 4) * (n + 4)});
          T* W1_data = W1.mutable_data<T>(PlaceType());
          Y1.Resize({m * (n + 4)});
          T* Y1_data = Y1.mutable_data<T>(PlaceType());
          const int NN = n + 4;
          const int KK = k + 4;
          auto start_X = paddle::platform::PosixInNsec() * 1e-3;
#pragma omp parallel for
          for (int i = 0; i < m; i++) {
            memcpy(X1_data + i * KK, a_data + i * k, k * sizeof(a_data[0]));
          }
          auto end_X = paddle::platform::PosixInNsec() * 1e-3;
          auto sum_X = end_X - start_X;
          LOG(INFO) << "mem copy X " << sum_X;
          auto start_W = paddle::platform::PosixInNsec() * 1e-3;
#pragma omp parallel for
          for (int i = 0; i < k; i++) {
            memcpy(W1_data + i * NN, b_data + i * n, n * sizeof(b_data[0]));
          }
          auto end_W = paddle::platform::PosixInNsec() * 1e-3;
          auto sum_W = end_W - start_W;
          LOG(INFO) << "mem copy W " << sum_W;
          for (int i = 0; i < FLAGS_burning; ++i) {
            blas.GEMM(false, false, m, m, k, static_cast<T>(1.0), X1_data, KK,
                      W1_data, NN, static_cast<T>(0.0), Y1_data, NN);
          }
          double sum = 0;
          for (int i = 0; i < FLAGS_repeat; ++i) {
            auto start = paddle::platform::PosixInNsec() * 1e-3;
            blas.GEMM(false, false, m, m, k, static_cast<T>(1.0), X1_data, KK,
                      W1_data, NN, static_cast<T>(0.0), Y1_data, NN);
            auto end = paddle::platform::PosixInNsec() * 1e-3;
            sum += end - start;
          }
          LOG(INFO) << "m: " << m << " k:" << k << " n: " << n
                    << " Matmul time: " << sum / FLAGS_repeat;
        } else {
          for (int i = 0; i < FLAGS_burning; ++i) {
            blas.MatMul(m, n, k, a_data, b_data, c_data);
          }
          double sum = 0;
          for (int i = 0; i < FLAGS_repeat; ++i) {
            auto start = paddle::platform::PosixInNsec() * 1e-3;
            blas.MatMul(m, n, k, a_data, b_data, c_data);
            auto end = paddle::platform::PosixInNsec() * 1e-3;
            sum += end - start;
          }
          LOG(INFO) << "m: " << m << " k:" << k << " n: " << n
                    << " Matmul time: " << sum / FLAGS_repeat;
        }
      }
    }
  }
}

#define BenchKernelVMul BenchKernelXYZN
#define BenchKernelVAdd BenchKernelXYZN
#define BenchKernelVAddRelu BenchKernelXYZN
#define BenchKernelVSub BenchKernelXYZN

#define BenchKernelVScal BenchKernelAXYN
#define BenchKernelVAddBias BenchKernelAXYN

#define BenchKernelVRelu BenchKernelXYN
#define BenchKernelVIdentity BenchKernelXYN
#define BenchKernelVSquare BenchKernelXYN
#define BenchKernelVExp BenchKernelXYN
#define BenchKernelVSigmoid BenchKernelXYN
#define BenchKernelVTanh BenchKernelXYN
#define BenchKernelVCopy BenchKernelXYN

#define BenchKernelHMax BenchKernelXRN
#define BenchKernelHSum BenchKernelXRN

#define BenchKernelLSTMCtHt BenchKernelLSTM
#define BenchKernelLSTMC1H1 BenchKernelLSTM

#define BenchKernelGRUH1 BenchKernelGRU
#define BenchKernelGRUHtPart1 BenchKernelGRU
#define BenchKernelGRUHtPart2 BenchKernelGRU

using CPUPlace = paddle::platform::CPUPlace;

#define BENCH_FP32_CPU(name)                                \
  BENCH_JITKERNEL(name, FP32, CPU) {                        \
    BenchKernel##name<jit::name##Tuple<float>, CPUPlace>(); \
  }

BENCH_FP32_CPU(MatMul);
// Benchmark all jit kernels including jitcode, mkl and refer.
// To use this tool, run command: ./benchmark [options...]
// Options:
//     --burning: the burning time before count
//     --repeat: the repeat times
//     --max_size: the max size would be tested
//     --filter: the bench name would be run
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Burning " << FLAGS_burning << " times, Repeat " << FLAGS_repeat
            << " times.";

  RUN_ALL_BENCHMARK();
}
