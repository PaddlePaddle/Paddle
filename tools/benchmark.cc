//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <sys/time.h>
#include <time.h>

#include "tools/benchmark.h"

namespace paddle {
namespace framework {

inline uint64_t NanoTime() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

template <typename DeviceContext>
void Benchmark<DeviceContext>::Register(const char* op) {
  auto& op_info = OpInfoMap::Instance().Get(op);
}

template <typename DeviceContext>
void Benchmark<DeviceContext>::Run(int iters) const {}

}  // namespace framework
}  // namespace paddle
