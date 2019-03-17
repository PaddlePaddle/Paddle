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
#define BLOOMFILTER_HEADER_SIZE 32
#define BLOOMFILTER_MAGIC_NUM_OLD 17062621
#define BLOOMFILTER_MAGIC_NUM_NEW 17070416

#include <stdlib.h>
#include <inttypes.h>

namespace paddle {
namespace operators {
namespace math {

struct bloomfilter {
    uint64_t  magic_num;
    uint64_t  m;
    uint64_t  k;
    uint64_t  count;
    unsigned char bit_vector[1];
};

int bloomfilter_check(struct bloomfilter* filter);

void
bloomfilter_init(struct bloomfilter *bloomfilter, uint64_t m, uint64_t k);

int
bloomfilter_set(struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_set_nocheck(struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_get(const struct bloomfilter *bloomfilter, const void *key, size_t len);

int
bloomfilter_dump(struct bloomfilter *bloomfilter, const void *path);

int
bloomfilter_load(struct bloomfilter **bloomfilter, const void *path);

int
bloomfilter_get_hash(struct bloomfilter *bloomfilter, const void *key, size_t len, char *dst);

uint64_t
char_to_little_endian_64bits(unsigned char *bytes);

uint32_t
char_to_little_endian_32bits(unsigned char *bytes);

}  // namespace math
}  // namespace operators
}  // namespace paddle
