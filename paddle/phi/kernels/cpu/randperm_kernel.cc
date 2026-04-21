// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/randperm_kernel.h"

#include <array>
#include <cstdint>
#include <limits>

#include "paddle/common/flags.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

// ---------------------------------------------------------------------------
// This is NOT the same as std::mt19937 or std::mt19937_64.
// Using this engine ensures bit-for-bit identical output with torch.randperm.
// ---------------------------------------------------------------------------

constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

class TorchMT19937Engine {
 public:
  inline explicit TorchMT19937Engine(uint64_t seed = 5489) {
    init_with_uint32(seed);
  }

  inline uint32_t operator()() {
    if (--(left_) == 0) {
      next_state();
    }
    uint32_t y = *(state_.data() + next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
  }

  inline uint64_t random64() {
    uint32_t r1 = (*this)();
    uint32_t r2 = (*this)();
    return (static_cast<uint64_t>(r1) << 32) | static_cast<uint64_t>(r2);
  }

 private:
  std::array<uint32_t, MERSENNE_STATE_N> state_;
  int left_ = 1;
  uint32_t next_ = 0;

  inline void init_with_uint32(uint64_t seed) {
    state_[0] = seed & 0xffffffff;
    for (int j = 1; j < MERSENNE_STATE_N; j++) {
      state_[j] = (1812433253 * (state_[j - 1] ^ (state_[j - 1] >> 30)) + j);
    }
    left_ = 1;
    next_ = 0;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  inline void next_state() {
    uint32_t* p = state_.data();
    left_ = MERSENNE_STATE_N;
    next_ = 0;

    for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }
    for (int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }
    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], state_[0]);
  }
};

template <typename T, typename Context>
void RandpermKernel(const Context& dev_ctx,
                    int n,
                    DataType dtype UNUSED,
                    DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);

  if (FLAGS_use_accuracy_compatible_kernel) {
    // MT19937 engine with that seed so the random sequence is identical.
    uint64_t seed = dev_ctx.GetGenerator()->GetCurrentSeed();
    TorchMT19937Engine engine(seed);

    if (n < static_cast<int>(std::numeric_limits<uint32_t>::max() / 20)) {
      // For small n: classic Fisher-Yates shuffle using 32-bit random values
      for (int i = 0; i < n; ++i) {
        out_data[i] = static_cast<T>(i);
      }
      for (int i = 0; i < n - 1; i++) {
        int64_t z = engine() % (n - i);
        T save = out_data[i];
        out_data[i] = out_data[z + i];
        out_data[z + i] = save;
      }
    } else {
      // For large n: inside-out Fisher-Yates using 64-bit random values
      for (int i = 0; i < n; i++) {
        int64_t z = static_cast<int64_t>(engine.random64() % (i + 1));
        out_data[i] = out_data[z];
        out_data[z] = static_cast<T>(i);
      }
    }

    // Advance the generator state so that successive randperm calls within the
    // same run produce different results
    dev_ctx.GetGenerator()->SetCurrentSeed(engine());
  } else {
    int seed = 0;
    std::shared_ptr<std::mt19937_64> engine;
    if (seed) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed);
    } else {
      engine = dev_ctx.GetGenerator()->GetCPUEngine();
    }
    for (int i = 0; i < n; ++i) {
      out_data[i] = static_cast<T>(i);
    }
    std::shuffle(out_data, out_data + n, *engine);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(randperm,
                   CPU,
                   ALL_LAYOUT,
                   phi::RandpermKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
