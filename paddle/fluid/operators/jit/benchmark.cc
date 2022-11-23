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

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

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
void RandomVec(const int n,
               T* a,
               const T lower = static_cast<T>(-20.f),
               const T upper = static_cast<T>(20.f),
               unsigned int seed = 100) {
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
    PADDLE_THROW(
        paddle::platform::errors::Fatal("Benchmark target can not be empty."));
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

using Tensor = phi::DenseTensor;
template <typename KernelTuple, typename PlaceType>
void BenchKernelXYZN() {
  using T = typename KernelTuple::data_type;
  for (int d : TestSizes()) {
    Tensor x, y, z;
    x.Resize({d});
    y.Resize({d});
    z.Resize({d});
    T* x_data = x.mutable_data<T>(PlaceType());
    T* y_data = y.mutable_data<T>(PlaceType());
    T* z_data = z.mutable_data<T>(PlaceType());
    RandomVec<T>(d, x_data);
    RandomVec<T>(d, y_data);
    BenchAllImpls<KernelTuple, PlaceType>(
        d, x.data<T>(), y.data<T>(), z_data, d);
    // test inplace
    BenchAllImpls<KernelTuple, PlaceType>(d, x.data<T>(), z_data, z_data, d);
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelAXYN() {
  using T = typename KernelTuple::data_type;
  for (int d : TestSizes()) {
    const T a = static_cast<T>(3);
    Tensor x, y;
    x.Resize({d});
    y.Resize({d});
    T* x_data = x.mutable_data<T>(PlaceType());
    T* y_data = y.mutable_data<T>(PlaceType());
    RandomVec<T>(d, x_data);
    BenchAllImpls<KernelTuple, PlaceType>(d, &a, x.data<T>(), y_data, d);
    // test inplace
    BenchAllImpls<KernelTuple, PlaceType>(d, &a, x.data<T>(), x_data, d);
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelXRN() {
  using T = typename KernelTuple::data_type;
  for (int d : TestSizes()) {
    Tensor x;
    RandomVec<T>(d, x.mutable_data<T>({d}, PlaceType()));
    T res;
    BenchAllImpls<KernelTuple, PlaceType>(d, x.data<T>(), &res, d);
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelXYN() {
  using T = typename KernelTuple::data_type;
  for (int d : TestSizes()) {
    Tensor x, y;
    x.Resize({d});
    y.Resize({d});
    T* x_data = x.mutable_data<T>(PlaceType());
    T* y_data = y.mutable_data<T>(PlaceType());
    RandomVec<T>(d, x_data);
    BenchAllImpls<KernelTuple, PlaceType>(d, x.data<T>(), y_data, d);
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelLSTM() {
  using T = typename KernelTuple::data_type;
  for (bool use_peephole : {true, false}) {
    for (int d : TestSizes()) {
      const jit::lstm_attr_t attr(
          d, jit::kVSigmoid, jit::kVTanh, jit::kVTanh, use_peephole);
      Tensor x, ct_1, ct, ht, wp, checked;
      x.Resize({4 * d});
      ct_1.Resize({d});
      ct.Resize({d});
      ht.Resize({d});
      wp.Resize({3 * d});
      checked.Resize({2 * d});
      auto place = PlaceType();
      RandomVec<T>(x.numel(), x.mutable_data<T>(place), -2.f, 2.f);
      RandomVec<T>(wp.numel(), wp.mutable_data<T>(place), -2.f, 2.f);
      RandomVec<T>(ct_1.numel(), ct_1.mutable_data<T>(place), -2.f, 2.f);
      const T* ct_1_data = ct_1.data<T>();
      const T* wp_data = wp.data<T>();
      T* x_data = x.mutable_data<T>(place);
      T* checked_data = checked.mutable_data<T>(place);
      T* ct_data = ct.mutable_data<T>(place);
      T* ht_data = ht.mutable_data<T>(place);
      jit::lstm_t step;
      step.gates = x_data;
      step.ct_1 = ct_1_data;
      step.ct = ct_data;
      step.ht = ht_data;
      if (use_peephole) {
        step.wp = wp_data;
        step.checked = checked_data;
      }
      BenchAllImpls<KernelTuple, PlaceType>(attr, &step, &attr);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelGRU() {
  using T = typename KernelTuple::data_type;
  for (int d : TestSizes()) {
    const jit::gru_attr_t attr(d, jit::kVSigmoid, jit::kVTanh);
    auto place = PlaceType();
    Tensor x, ht_1, ht;
    x.Resize({3 * d});
    ht_1.Resize({d});
    ht.Resize({d});
    RandomVec<T>(3 * d, x.mutable_data<T>(place), -2.f, 2.f);
    RandomVec<T>(d, ht_1.mutable_data<T>(place), -2.f, 2.f);
    const T* ht_1_data = ht_1.data<T>();
    T* x_data = x.mutable_data<T>(place);
    T* ht_data = ht.mutable_data<T>(place);
    jit::gru_t step;
    step.gates = x_data;
    step.ht_1 = ht_1_data;
    step.ht = ht_data;
    BenchAllImpls<KernelTuple, PlaceType>(attr, &step, &attr);
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelSeqPool() {
  using T = typename KernelTuple::data_type;
  std::vector<jit::SeqPoolType> pool_types = {
      jit::SeqPoolType::kSum, jit::SeqPoolType::kAvg, jit::SeqPoolType::kSqrt};
  for (auto type : pool_types) {
    for (int w : TestSizes()) {
      jit::seq_pool_attr_t attr(w, type);
      for (int h : TestSizes()) {
        attr.h = h;
        Tensor x, y;
        x.Resize({h * w});
        y.Resize({w});
        RandomVec<T>(h * w, x.mutable_data<T>(PlaceType()), -2.f, 2.f);
        const T* x_data = x.data<T>();
        T* y_data = y.mutable_data<T>(PlaceType());
        BenchAllImpls<KernelTuple, PlaceType>(attr, x_data, y_data, &attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelEmbSeqPool() {
  using T = typename KernelTuple::data_type;
  std::vector<jit::SeqPoolType> pool_types = {jit::SeqPoolType::kSum};
  int64_t tbl_h = 1e4;
  for (int tbl_w : {10, 16, 256}) {
    Tensor table;
    table.Resize({tbl_h, tbl_w});
    RandomVec<T>(tbl_h * tbl_w, table.mutable_data<T>(PlaceType()), -2.f, 2.f);
    const T* table_data = table.data<T>();
    for (auto type : pool_types) {
      for (int idx_w : {1, 2, 10, 16}) {
        for (int idx_h : {1, 2, 9, 13, 16}) {
          int64_t out_w = tbl_w * idx_w;
          jit::emb_seq_pool_attr_t attr(
              tbl_h, tbl_w, idx_h, idx_w, out_w, type);
          Tensor idx, out;
          idx.Resize({idx_h, idx_w});
          out.Resize({out_w});
          RandomVec<int64_t>(idx_h * idx_w,
                             idx.mutable_data<int64_t>(PlaceType()),
                             0,
                             tbl_h - 1);
          const int64_t* idx_data = idx.data<int64_t>();
          T* o_data = out.mutable_data<T>(PlaceType());
          BenchAllImpls<KernelTuple, PlaceType>(
              attr, table_data, idx_data, o_data, &attr);
        }
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelSgd() {
  using T = typename KernelTuple::data_type;
  const T lr = 0.1;
  auto UnDuplicatedRandomVec = [](int n,
                                  const int64_t lower,
                                  const int64_t upper) -> std::vector<int64_t> {
    PADDLE_ENFORCE_LE(
        static_cast<size_t>(upper - lower),
        n - 1,
        paddle::platform::errors::InvalidArgument(
            "The range of Sgd (upper - lower) should be equal to or lower "
            "than n-1 (Sgd size -1). But upper - lower is %d and n-1 is %d.",
            static_cast<size_t>(upper - lower),
            (n - 1)));
    PADDLE_ENFORCE_GT(
        n,
        0,
        paddle::platform::errors::InvalidArgument(
            "The Sgd size should be larger than 0. But the n is %d.", n));
    std::vector<int64_t> all, out;
    for (int i = 0; i < n; ++i) {
      all.push_back(i);
    }
    std::random_device rnd;
    int64_t seed_tmp = rnd();
    std::default_random_engine rng(seed_tmp);
    std::shuffle(all.begin(), all.end(), rng);
    out.insert(out.begin(), all.begin(), all.begin() + n);
    return out;
  };
  for (int param_h : {1, 1000}) {
    for (int grad_w : {1, 2, 8, 16, 30, 256}) {
      // only benchmark inplace
      Tensor param;
      param.Resize({param_h, grad_w});
      T* param_data = param.mutable_data<T>(PlaceType());
      RandomVec<T>(param_h * grad_w, param_data, -2.f, 2.f);
      for (int rows_size = 1; rows_size <= std::min(param_h, 10); ++rows_size) {
        Tensor grad;
        grad.Resize({rows_size, grad_w});
        std::vector<int64_t> rows =
            UnDuplicatedRandomVec(rows_size, 0, rows_size - 1);
        RandomVec<T>(
            rows_size * grad_w, grad.mutable_data<T>(PlaceType()), -2.f, 2.f);
        const T* grad_data = grad.data<T>();
        const int64_t* rows_data = rows.data();
        jit::sgd_attr_t attr(param_h, grad_w, rows_size, grad_w, rows_size);
        BenchAllImpls<KernelTuple, PlaceType>(
            attr, &lr, param_data, grad_data, rows_data, param_data, &attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelMatMul() {
  using T = typename KernelTuple::data_type;
  for (int m : {1, 2, 3, 4}) {
    for (int n : TestSizes()) {
      for (int k : TestSizes()) {
        Tensor a, b, c;
        a.Resize({m * k});
        b.Resize({k * n});
        c.Resize({m * n});
        RandomVec<T>(m * k, a.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(k * n, b.mutable_data<T>(PlaceType()), -2.f, 2.f);
        const T* a_data = a.data<T>();
        const T* b_data = b.data<T>();
        T* c_data = c.mutable_data<T>(PlaceType());
        const jit::matmul_attr_t attr{m, n, k};
        BenchAllImpls<KernelTuple, PlaceType>(
            attr, a_data, b_data, c_data, &attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelSoftmax() {
  using T = typename KernelTuple::data_type;
  for (int bs : {1, 2, 10}) {
    for (int n : TestSizes()) {
      Tensor x, y;
      x.Resize({bs, n});
      y.Resize({bs, n});
      RandomVec<T>(bs * n, x.mutable_data<T>(PlaceType()), -2.f, 2.f);
      const T* x_data = x.data<T>();
      T* y_data = y.mutable_data<T>(PlaceType());
      BenchAllImpls<KernelTuple, PlaceType>(n, x_data, y_data, n, bs, 1);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelLayerNorm() {
  using T = typename KernelTuple::data_type;
  const T epsilon = 9.99999975e-06;
  for (int n : {1, 2, 10}) {
    for (int x_dim_0 : {1, 9, 17, 50}) {
      int left = n * x_dim_0;
      for (int x_dim_1 : TestSizes()) {
        int right = x_dim_1;
        int sz = left * right;
        Tensor x, mean, var, scale, bias, out;
        x.Resize({n, x_dim_0, x_dim_1});
        out.Resize({n, x_dim_0, x_dim_1});
        mean.Resize({n, x_dim_0});
        var.Resize({n, x_dim_0});
        scale.Resize({x_dim_1});
        bias.Resize({x_dim_1});

        RandomVec<T>(sz, x.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(left, mean.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(left, var.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(right, scale.mutable_data<T>(PlaceType()), -2.f, 2.f);
        RandomVec<T>(right, bias.mutable_data<T>(PlaceType()), -2.f, 2.f);

        const T* scale_data = scale.data<T>();
        const T* bias_data = bias.data<T>();
        T* x_data = x.data<T>();
        T* mean_data = mean.data<T>();
        T* var_data = var.data<T>();
        T* out_data = out.mutable_data<T>(PlaceType());

        BenchAllImpls<KernelTuple, PlaceType>(right,
                                              x_data,
                                              out_data,
                                              mean_data,
                                              var_data,
                                              scale_data,
                                              bias_data,
                                              left,
                                              epsilon,
                                              right);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelCRFDecoding() {
  using T = typename KernelTuple::data_type;
  constexpr int state_trans_base_idx = 2;
  for (int seq_len : {1, 11, 17, 50}) {
    for (int tag_num : TestSizes()) {
      int x_sz = seq_len * tag_num;
      int w_sz = (tag_num + state_trans_base_idx) * tag_num;
      Tensor x, w, alpha, track;
      x.Resize({seq_len, tag_num});
      w.Resize({tag_num + state_trans_base_idx, tag_num});
      alpha.Resize({seq_len, tag_num});
      track.Resize({seq_len, tag_num});

      RandomVec<T>(x_sz, x.mutable_data<T>(PlaceType()), -2.f, 2.f);
      RandomVec<T>(w_sz, w.mutable_data<T>(PlaceType()), -2.f, 2.f);

      const T* x_data = x.data<T>();
      const T* w_data = w.data<T>();
      T* alpha_data = alpha.mutable_data<T>(PlaceType());
      int* track_data = track.mutable_data<int>(PlaceType());

      BenchAllImpls<KernelTuple, PlaceType>(
          tag_num, seq_len, x_data, w_data, alpha_data, track_data, tag_num);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void BenchKernelVBroadcast() {
  using T = typename KernelTuple::data_type;
  for (int64_t w : {1, 16, 64, 100, 256}) {
    Tensor x;
    x.Resize({w});
    RandomVec<T>(w, x.mutable_data<T>(PlaceType()));
    const T* x_data = x.data<T>();
    for (int h : TestSizes()) {
      Tensor y;
      y.Resize({h * w});
      T* y_data = y.mutable_data<T>(PlaceType());
      BenchAllImpls<KernelTuple, PlaceType>(
          w, x_data, y_data, static_cast<int64_t>(h), w);
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

// xyzn
BENCH_FP32_CPU(VMul);
BENCH_FP32_CPU(VAdd);
BENCH_FP32_CPU(VAddRelu);
BENCH_FP32_CPU(VSub);

// axyn
BENCH_FP32_CPU(VScal);
BENCH_FP32_CPU(VAddBias);

// xyn
BENCH_FP32_CPU(VRelu);
BENCH_FP32_CPU(VIdentity);
BENCH_FP32_CPU(VSquare);
BENCH_FP32_CPU(VExp);
BENCH_FP32_CPU(VSigmoid);
BENCH_FP32_CPU(VTanh);
BENCH_FP32_CPU(VCopy);

// xrn
BENCH_FP32_CPU(HMax);
BENCH_FP32_CPU(HSum);

// LSTM
BENCH_FP32_CPU(LSTMCtHt);
BENCH_FP32_CPU(LSTMC1H1);

// GRU
BENCH_FP32_CPU(GRUH1);
BENCH_FP32_CPU(GRUHtPart1);
BENCH_FP32_CPU(GRUHtPart2);

BENCH_FP32_CPU(LayerNorm);
BENCH_FP32_CPU(CRFDecoding);

BENCH_FP32_CPU(SeqPool);
BENCH_FP32_CPU(EmbSeqPool);
BENCH_FP32_CPU(MatMul);
BENCH_FP32_CPU(Softmax);
BENCH_FP32_CPU(Sgd);
BENCH_FP32_CPU(VBroadcast);

// Benchmark all jit kernels including jitcode, mkl and refer.
// To use this tool, run command: ./benchmark [options...]
// Options:
//     --burning: the burning time before count
//     --repeat: the repeat times
//     --max_size: the max size would be tested
//     --filter: the bench name would be run
int main(int argc, char* argv[]) {
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Burning " << FLAGS_burning << " times, Repeat " << FLAGS_repeat
            << " times.";

  RUN_ALL_BENCHMARK();
}
