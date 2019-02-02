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
#include "paddle/fluid/operators/jit/kernels.h"
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

#define BENCH_FP32_CPU(name) BENCH_JITKERNEL(name, FP32, CPU)

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

template <typename KernelTuples, typename... Args>
struct BenchFunc {
  // return this function avg time
  double operator()(const typename KernelTuples::func_type tgt, Args... args) {
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

template <jit::KernelType KT, typename KernelTuples, typename PlaceType,
          typename... Args>
void BenchAllImpls(const typename KernelTuples::attr_type& attr, Args... args) {
  BenchFunc<KernelTuples, Args...> benchmark;
  std::vector<std::pair<std::string, double>> infos;
  // test refer
  auto refer = jit::GetRefer<KT, KernelTuples>();
  if (!refer) {
    LOG(FATAL) << "Refer can not be empty!";
  }
  infos.push_back(std::make_pair("Refer", benchmark(refer, args...)));

  // test jitcode
  auto jitcode = jit::GetJitCode<KT, KernelTuples, PlaceType>(attr);
  if (jitcode) {
    infos.push_back(std::make_pair("JitCode", benchmark(jitcode, args...)));
  }
  // test all impls in more
  jit::KernelKey kkey(KT, PlaceType());
  auto& pool = jit::KernelPool().Instance().AllKernels();
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto& impls = iter->second;
    for (auto& impl : impls) {
      auto i = dynamic_cast<const jit::KernelMore<KernelTuples>*>(impl.get());
      if (i && i->UseMe(attr)) {
        auto more = i->GetFunc();
        infos.push_back(
            std::make_pair(i->ImplType(), benchmark(more, args...)));
      }
    }
  }
  // Test result from Get function
  auto tgt = jit::Get<KT, KernelTuples, PlaceType>(attr);
  if (!tgt) {
    LOG(FATAL) << "Target can not be empty!";
  }
  infos.push_back(std::make_pair("Target", benchmark(tgt, args...)));

  // print
  std::ostringstream loginfos;
  loginfos << "Kernel Type " << jit::to_string(KT) << ": " << attr << ": ";
  for (auto pair : infos) {
    loginfos << pair.first << " takes " << pair.second << " us; ";
  }
  LOG(INFO) << loginfos.str();
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchXYZNKernel() {
  for (int d : TestSizes()) {
    std::vector<T> x(d), y(d), z(d);
    RandomVec<T>(d, x.data());
    RandomVec<T>(d, y.data());
    BenchAllImpls<KT, jit::XYZNTuples<T>, PlaceType>(d, x.data(), y.data(),
                                                     z.data(), d);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchAXYNKernel() {
  for (int d : TestSizes()) {
    const T a = static_cast<T>(3);
    std::vector<T> x(d), y(d);
    RandomVec<T>(d, x.data());
    BenchAllImpls<KT, jit::AXYNTuples<T>, PlaceType>(d, &a, x.data(), y.data(),
                                                     d);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchXYNKernel() {
  for (int d : TestSizes()) {
    std::vector<T> x(d), y(d);
    RandomVec<T>(d, x.data());
    BenchAllImpls<KT, jit::XYNTuples<T>, PlaceType>(d, x.data(), y.data(), d);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchLSTMKernel() {
  for (bool use_peephole : {true, false}) {
    for (int d : TestSizes()) {
      const jit::lstm_attr_t attr(d, jit::kVSigmoid, jit::kVTanh, jit::kVTanh,
                                  use_peephole);
      std::vector<T> x(4 * d), ct_1(d), ct(d), ht(d), wp(3 * d), checked(2 * d);
      RandomVec<T>(4 * d, x.data(), -2.f, 2.f);
      RandomVec<T>(3 * d, wp.data(), -2.f, 2.f);
      RandomVec<T>(d, ct_1.data(), -2.f, 2.f);
      const T* ct_1_data = ct_1.data();
      const T* wp_data = wp.data();
      T* x_data = x.data();
      T* checked_data = checked.data();
      T* ct_data = ct.data();
      T* ht_data = ht.data();
      jit::lstm_t step;
      step.gates = x_data;
      step.ct_1 = ct_1_data;
      step.ct = ct_data;
      step.ht = ht_data;
      if (use_peephole) {
        step.wp = wp_data;
        step.checked = checked_data;
      }
      BenchAllImpls<KT, jit::LSTMTuples<T>, PlaceType>(attr, &step, &attr);
    }
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchGRUKernel() {
  for (int d : TestSizes()) {
    const jit::gru_attr_t attr(d, jit::kVSigmoid, jit::kVTanh);
    std::vector<T> x(3 * d), ht_1(d), ht(d);
    RandomVec<T>(3 * d, x.data(), -2.f, 2.f);
    RandomVec<T>(d, ht_1.data(), -2.f, 2.f);
    const T* ht_1_data = ht_1.data();
    T* x_data = x.data();
    T* ht_data = ht.data();
    jit::gru_t step;
    step.gates = x_data;
    step.ht_1 = ht_1_data;
    step.ht = ht_data;
    BenchAllImpls<KT, jit::GRUTuples<T>, PlaceType>(attr, &step, &attr);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchSeqPoolKernel() {
  std::vector<jit::SeqPoolType> pool_types = {
      jit::SeqPoolType::kSum, jit::SeqPoolType::kAvg, jit::SeqPoolType::kSqrt};
  for (auto type : pool_types) {
    for (int w : TestSizes()) {
      jit::seq_pool_attr_t attr(w, type);
      for (int h : TestSizes()) {
        attr.h = h;
        std::vector<T> x(h * w), y(w);
        RandomVec<T>(h * w, x.data(), -2.f, 2.f);
        const T* x_data = x.data();
        T* y_data = y.data();
        BenchAllImpls<KT, jit::SeqPoolTuples<T>, PlaceType>(attr, x_data,
                                                            y_data, &attr);
      }
    }
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void BenchMatMulKernel() {
  for (int m : {1, 2, 3, 4}) {
    for (int n : TestSizes()) {
      for (int k : TestSizes()) {
        std::vector<T> a(m * k), b(k * n), c(m * n);
        RandomVec<T>(m * k, a.data(), -2.f, 2.f);
        RandomVec<T>(k * n, b.data(), -2.f, 2.f);
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* c_data = c.data();
        BenchAllImpls<KT, jit::MatMulTuples<T>, PlaceType>(k, a_data, b_data,
                                                           c_data, m, n, k);
      }
    }
  }
}

using T = float;
using PlaceType = paddle::platform::CPUPlace;

// xyzn
BENCH_FP32_CPU(kVMul) { BenchXYZNKernel<jit::kVMul, T, PlaceType>(); }

BENCH_FP32_CPU(kVAdd) { BenchXYZNKernel<jit::kVAdd, T, PlaceType>(); }

BENCH_FP32_CPU(kVAddRelu) { BenchXYZNKernel<jit::kVAddRelu, T, PlaceType>(); }

BENCH_FP32_CPU(kVSub) { BenchXYZNKernel<jit::kVSub, T, PlaceType>(); }

// axyn
BENCH_FP32_CPU(kVScal) { BenchAXYNKernel<jit::kVScal, T, PlaceType>(); }

BENCH_FP32_CPU(kVAddBias) { BenchAXYNKernel<jit::kVAddBias, T, PlaceType>(); }

// xyn
BENCH_FP32_CPU(kVRelu) { BenchXYNKernel<jit::kVRelu, T, PlaceType>(); }

BENCH_FP32_CPU(kVIdentity) { BenchXYNKernel<jit::kVIdentity, T, PlaceType>(); }

BENCH_FP32_CPU(kVSquare) { BenchXYNKernel<jit::kVSquare, T, PlaceType>(); }

BENCH_FP32_CPU(kVExp) { BenchXYNKernel<jit::kVExp, T, PlaceType>(); }

BENCH_FP32_CPU(kVSigmoid) { BenchXYNKernel<jit::kVSigmoid, T, PlaceType>(); }

BENCH_FP32_CPU(kVTanh) { BenchXYNKernel<jit::kVTanh, T, PlaceType>(); }

// lstm and peephole
BENCH_FP32_CPU(kLSTMCtHt) { BenchLSTMKernel<jit::kLSTMCtHt, T, PlaceType>(); }

BENCH_FP32_CPU(kLSTMC1H1) { BenchLSTMKernel<jit::kLSTMC1H1, T, PlaceType>(); }

// gru functions
BENCH_FP32_CPU(kGRUH1) { BenchGRUKernel<jit::kGRUH1, T, PlaceType>(); }

BENCH_FP32_CPU(kGRUHtPart1) {
  BenchGRUKernel<jit::kGRUHtPart1, T, PlaceType>();
}

BENCH_FP32_CPU(kGRUHtPart2) {
  BenchGRUKernel<jit::kGRUHtPart2, T, PlaceType>();
}

// seq pool function
BENCH_FP32_CPU(kSeqPool) { BenchSeqPoolKernel<jit::kSeqPool, T, PlaceType>(); }

// matmul
BENCH_FP32_CPU(kMatMul) { BenchMatMulKernel<jit::kMatMul, T, PlaceType>(); }

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
