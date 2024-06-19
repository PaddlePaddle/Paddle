// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#define BLOOMFILTER_MAGIC_NUM_NEW 17070416

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif  // _WIN32

#include <cinttypes>

namespace phi {
namespace math {

#pragma pack(push, 4)
struct bloomfilter {
  uint64_t magic_num;
  uint64_t m;
  uint64_t k;
  uint64_t count;
  unsigned char bit_vector[1];
};
#pragma pack(pop)

inline int bloomfilter_get(const struct bloomfilter *bloomfilter,
                           const void *key,
                           size_t len);
inline int bloomfilter_check(struct bloomfilter *filter);

#define bit_get(v, n) ((v)[(n) >> 3] & (0x1 << (0x7 - ((n)&0x7))))
#define ROTL64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))
#define BIG_CONSTANT(x) (x##LLU)

uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

void murmurhash3_x64_128(const void *key,
                         const int len,
                         const uint32_t seed,
                         void *out) {
  const uint8_t *data = (const uint8_t *)key;
  const int nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;
  int i = 0;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  //----------
  // body

  const uint64_t *blocks = (const uint64_t *)(data);

  uint64_t k1;
  uint64_t k2;

  for (i = 0; i < nblocks; i++) {
    k1 = blocks[i * 2 + 0];
    k2 = blocks[i * 2 + 1];

    k1 *= c1;
    k1 = ROTL64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = ROTL64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = ROTL64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = ROTL64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  //----------
  // tail

  const uint8_t *tail = (const uint8_t *)(data + nblocks * 16);
  uint64_t nk1 = 0;
  uint64_t nk2 = 0;

  uint64_t tail0_64 = *(uint64_t *)(tail);     // NOLINT
  uint64_t tail_64 = *(uint64_t *)(tail + 8);  // NOLINT
  uint64_t mask0 = 0xffffffffffffffff;
  uint64_t mask = 0x00ffffffffffffff;

  int flag = len & 15;
  if (flag && flag <= 8) {
    tail0_64 &= (mask0 >> ((8 - flag) << 3));
  } else if (flag > 8) {
    tail_64 &= (mask >> ((15 - flag) << 3));
    nk2 ^= tail_64;
    nk2 *= c2;
    nk2 = ROTL64(nk2, 33);
    nk2 *= c1;
    h2 ^= nk2;
  }

  if (flag) {
    nk1 ^= tail0_64;
    nk1 *= c1;
    nk1 = ROTL64(nk1, 31);
    nk1 *= c2;
    h1 ^= nk1;
  }

  //----------
  // finalization

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  reinterpret_cast<uint64_t *>(out)[0] = h1;
  reinterpret_cast<uint64_t *>(out)[1] = h2;
}

inline int bloomfilter_check(struct bloomfilter *filter) {
  if (filter->magic_num == BLOOMFILTER_MAGIC_NUM_NEW) {
    return 1;
  } else {
    fprintf(stderr, "error magic_num, %" PRIu64 "\n", filter->magic_num);
    return 0;
  }
}

inline int bloomfilter_get(const struct bloomfilter *bloomfilter,
                           const void *key,
                           size_t len) {
  uint32_t i;
  uint64_t result[2];

  for (i = 0; i < bloomfilter->k; i++) {
    murmurhash3_x64_128(key, len, i, &result);
    result[0] %= bloomfilter->m;
    result[1] %= bloomfilter->m;
    if (!bit_get(bloomfilter->bit_vector, result[0])) {
      return 0;
    }
    if (!bit_get(bloomfilter->bit_vector, result[1])) {
      return 0;
    }
  }
  return 1;
}

}  // namespace math
}  // namespace phi
