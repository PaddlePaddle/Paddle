/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime_api.h>  // NOLINT

class RandomNumGen {
 public:
  __host__ __device__ __forceinline__ RandomNumGen(int gid, uint64_t seed) {
    next_random = seed + gid;
    next_random ^= next_random >> 33U;
    next_random *= 0xff51afd7ed558ccdUL;
    next_random ^= next_random >> 33U;
    next_random *= 0xc4ceb9fe1a85ec53UL;
    next_random ^= next_random >> 33U;
  }
  __host__ __device__ __forceinline__ ~RandomNumGen() = default;
  __host__ __device__ __forceinline__ void SetSeed(int seed) {
    next_random = seed;
    NextValue();
  }
  __host__ __device__ __forceinline__ unsigned long long SaveState() const {
    return next_random;
  }
  __host__ __device__ __forceinline__ void LoadState(unsigned long long state) {
    next_random = state;
  }
  __host__ __device__ __forceinline__ int Random() {
    int ret_value = (int) (next_random & 0x7fffffffULL);
    NextValue();
    return ret_value;
  }
  __host__ __device__ __forceinline__ int RandomMod(int mod) {
    return Random() % mod;
  }
  __host__ __device__ __forceinline__ int64_t Random64() {
    int64_t ret_value = (next_random & 0x7FFFFFFFFFFFFFFFLL);
    NextValue();
    return ret_value;
  }
  __host__ __device__ __forceinline__ int64_t RandomMod64(int64_t mod) {
    return Random64() % mod;
  }
  __host__ __device__ __forceinline__ float RandomUniformFloat(float max = 1.0f, float min = 0.0f) {
    int value = (int) (next_random & 0xffffff);
    auto ret_value = (float) value;
    ret_value /= 0xffffffL;
    ret_value *= (max - min);
    ret_value += min;
    NextValue();
    return ret_value;
  }
  __host__ __device__ __forceinline__ bool RandomBool(float true_prob) {
    float value = RandomUniformFloat();
    return value <= true_prob;
  }
  __host__ __device__ __forceinline__ void NextValue() {
    //next_random = next_random * (unsigned long long)0xc4ceb9fe1a85ec53UL + generator_id;
    //next_random = next_random * (unsigned long long)25214903917ULL + 11;
    next_random = next_random * (unsigned long long) 13173779397737131ULL + 1023456798976543201ULL;
  }

 private:
  unsigned long long next_random = 1;
};
