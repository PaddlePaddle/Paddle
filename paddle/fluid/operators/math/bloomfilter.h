/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#define BLOOMFILTER_MAGIC_NUM_NEW 17070416

#include <inttypes.h>
#include <stdlib.h>

#include <stdio.h>
#include <string.h>

#include <unistd.h>

namespace paddle {
namespace operators {
namespace math {

#pragma pack(4)
struct bloomfilter {
  uint64_t magic_num;
  uint64_t m;
  uint64_t k;
  uint64_t count;
  unsigned char bit_vector[1];
};
int bloomfilter_get(const struct bloomfilter *bloomfilter, const void *key,
                    size_t len);
int bloomfilter_check(struct bloomfilter *filter);

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

void murmurhash3_x64_128(const void *key, const int len, const uint32_t seed,
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
  // no break here!!!
  switch (len & 15) {
    case 15:
      nk2 ^= ((uint64_t)tail[14]) << 48;
    case 14:
      nk2 ^= ((uint64_t)tail[13]) << 40;
    case 13:
      nk2 ^= ((uint64_t)tail[12]) << 32;
    case 12:
      nk2 ^= ((uint64_t)tail[11]) << 24;
    case 11:
      nk2 ^= ((uint64_t)tail[10]) << 16;
    case 10:
      nk2 ^= ((uint64_t)tail[9]) << 8;
    case 9:
      nk2 ^= ((uint64_t)tail[8]) << 0;
      nk2 *= c2;
      nk2 = ROTL64(nk2, 33);
      nk2 *= c1;
      h2 ^= nk2;
    case 8:
      nk1 ^= ((uint64_t)tail[7]) << 56;
    case 7:
      nk1 ^= ((uint64_t)tail[6]) << 48;
    case 6:
      nk1 ^= ((uint64_t)tail[5]) << 40;
    case 5:
      nk1 ^= ((uint64_t)tail[4]) << 32;
    case 4:
      nk1 ^= ((uint64_t)tail[3]) << 24;
    case 3:
      nk1 ^= ((uint64_t)tail[2]) << 16;
    case 2:
      nk1 ^= ((uint64_t)tail[1]) << 8;
    case 1:
      nk1 ^= ((uint64_t)tail[0]) << 0;
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

  //  ((uint64_t *)out)[0] = h1;
  reinterpret_cast<uint64_t *>(out)[0] = h1;
  //  ((uint64_t *)out)[1] = h2;
  reinterpret_cast<uint64_t *>(out)[1] = h2;
}

int bloomfilter_check(struct bloomfilter *filter) {
  if (filter->magic_num == BLOOMFILTER_MAGIC_NUM_NEW) {
    return 1;
  } else {
    fprintf(stderr, "error magic_num %ld\n", filter->magic_num);
    return 0;
  }
}

int bloomfilter_get(const struct bloomfilter *bloomfilter, const void *key,
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
}  // namespace operators
}  // namespace paddle
