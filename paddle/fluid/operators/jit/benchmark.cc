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

// #include <cstring>  // for memcpy
// #include <random>
#include <iostream>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"

DEFINE_int32(burning, 10, "Burning times.");
DEFINE_int32(repeat, 3000, "Repeat times.");
DEFINE_int32(max_size, 1000, "The Max size would be tested.");

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
void RandomVec(const int n, T* a, const T lower = static_cast<T>(-20.f),
               const T upper = static_cast<T>(20.f)) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
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

// return this function avg time
template <typename T, typename Func>
double BenchTartgetFunc(const Func tgt, const std::vector<T>& x,
                        const std::vector<T>& y, std::vector<T>& z) {  // NOLINT
  const T* x_data = x.data();
  const T* y_data = y.data();
  const int d = z.size();
  T* z_data = z.data();

  for (int i = 0; i < FLAGS_burning; ++i) {
    tgt(x_data, y_data, z_data, d);
  }
  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeat; ++i) {
    tgt(x_data, y_data, z_data, d);
  }
  auto end = GetCurrentUS();
  return (end - start) / FLAGS_repeat;
}

// Benchmark all jit kernels including jitcode, mkl and refer.
// To use this tool, run command: ./benchmark [options...]
// Options:
//     --burning: the burning time before count
//     --repeat: the repeat times
//     --max_size: the max size would be tested
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  using T = float;
  using PlaceType = paddle::platform::CPUPlace;
  namespace jit = paddle::operators::jit;
  const auto KT = jit::vmul;
  LOG(INFO) << "Burning " << FLAGS_burning << " times, Repeat " << FLAGS_repeat
            << " times.";
  for (int d : TestSizes()) {
    // for (kernels type) {  // TODO(TJ): more jit::KernelType
    std::vector<std::pair<std::string, double>> infos;
    std::vector<T> x(d), y(d), z(d);
    RandomVec<T>(d, x.data());
    RandomVec<T>(d, y.data());
    // refer
    auto refer = jit::GetRefer<KT, T, jit::VMulTuples<T>::func_type,
                               jit::VMulTuples<T>::attr_type>();
    if (refer) {
      auto res =
          BenchTartgetFunc<T, jit::VMulTuples<T>::func_type>(refer, x, y, z);
      infos.push_back(std::make_pair("Refer", res));
    }

    // test jitcode
    auto jitcode = jit::GetJitCode<KT, T, jit::VMulTuples<T>::func_type,
                                   jit::VMulTuples<T>::attr_type, PlaceType>(d);
    if (jitcode) {
      auto res =
          BenchTartgetFunc<T, jit::VMulTuples<T>::func_type>(jitcode, x, y, z);
      infos.push_back(std::make_pair("JitCode", res));
    }

    // test all impls in more
    jit::KernelKey kkey(KT, PlaceType());
    auto& pool = jit::KernelPool().Instance().AllKernels();
    auto iter = pool.find(kkey);
    if (iter != pool.end()) {
      auto& impls = iter->second;
      for (auto& impl : impls) {
        auto i =
            dynamic_cast<const jit::KernelImpl<T, jit::VMulTuples<T>::func_type,
                                               jit::VMulTuples<T>::attr_type>*>(
                impl.get());
        if (i && i->UseMe(d)) {
          auto more = i->GetFunc();
          auto res =
              BenchTartgetFunc<T, jit::VMulTuples<T>::func_type>(more, x, y, z);
          infos.push_back(std::make_pair("More", res));
        }
      }
    }

    // Test result from Get function
    auto tgt = jit::Get<KT, T, jit::VMulTuples<T>::func_type,
                        jit::VMulTuples<T>::attr_type, PlaceType>(d);
    if (!tgt) {
      LOG(ERROR) << "Target can not be empty!";
    }
    auto res = BenchTartgetFunc<T, jit::VMulTuples<T>::func_type>(tgt, x, y, z);
    infos.push_back(std::make_pair("Target", res));

    // print
    std::ostringstream loginfos;
    loginfos << "Kernel Type: " << KT << ", size " << d << ": ";
    for (auto pair : infos) {
      loginfos << pair.first << " takes " << pair.second << " us; ";
    }
    LOG(INFO) << loginfos.str();
    // }
  }
}
