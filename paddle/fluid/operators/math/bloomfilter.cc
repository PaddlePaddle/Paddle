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

#include "paddle/fluid/operators/math/bloomfilter.h"

#include <string.h>
#include <unistd.h>
#include <stdio.h>

namespace paddle {
namespace operators {
namespace math {

#define bit_set(v, n)    ((v)[(n) >> 3] |= (0x1 << (0x7 - ((n) & 0x7))))
#define bit_get(v, n)    ((v)[(n) >> 3] &  (0x1 << (0x7 - ((n) & 0x7))))
#define bit_clr(v, n)    ((v)[(n) >> 3] &=~(0x1 << (0x7 - ((n) & 0x7))))

#define ROTL32(x, r)	(((x) << (r)) | ((x) >> (32 - (r))))
#define ROTL64(x, r)	(((x) << (r)) | ((x) >> (64 - (r))))
#define BIG_CONSTANT(x) (x##LLU)

uint32_t fmix32(uint32_t h) {
    return h;
}

//uint64_t getblock64(const uint64_t * p, int i) {
//	return p[i];
//}

uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
}

void murmur3_hash32(const void *key, size_t len, uint32_t seed, void *out) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    int i = 0;
    uint32_t k1 = 0;
    uint32_t h1 = seed;

    const uint8_t *data = (const uint8_t *) key;
    const int nblocks = len >> 2;

    const uint32_t *blocks = (const uint32_t *) (data + nblocks * 4);
    const uint8_t *tail = (const uint8_t *) (data + nblocks * 4);

    for (i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = ROTL32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    switch (len & 3) {
    case 3:
        k1 ^= tail[2] << 16;
        break;
    case 2:
        k1 ^= tail[1] << 8;
        break;
    case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        break;
    };

    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    *(uint32_t*) out = h1;
}

void murmurhash3_x64_128(const void * key, const int len, const uint32_t seed, void * out) {
    const uint8_t * data = (const uint8_t*) key;
    const int nblocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;
    int i = 0;

    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

    //----------
    // body

    const uint64_t * blocks = (const uint64_t *) (data);

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

    const uint8_t * tail = (const uint8_t*) (data + nblocks * 16);
    uint64_t nk1 = 0;
    uint64_t nk2 = 0;
    //no break here!!!
    switch (len & 15) {
        case 15:
            nk2 ^= ((uint64_t) tail[14]) << 48;
        case 14:
            nk2 ^= ((uint64_t) tail[13]) << 40;
        case 13:
            nk2 ^= ((uint64_t) tail[12]) << 32;
        case 12:
            nk2 ^= ((uint64_t) tail[11]) << 24;
        case 11:
            nk2 ^= ((uint64_t) tail[10]) << 16;
        case 10:
            nk2 ^= ((uint64_t) tail[9]) << 8;
        case 9:
            nk2 ^= ((uint64_t) tail[8]) << 0;
            nk2 *= c2;
            nk2 = ROTL64(nk2, 33);
            nk2 *= c1;
            h2 ^= nk2;
        case 8:
            nk1 ^= ((uint64_t) tail[7]) << 56;
        case 7:
            nk1 ^= ((uint64_t) tail[6]) << 48;
        case 6:
            nk1 ^= ((uint64_t) tail[5]) << 40;
        case 5:
            nk1 ^= ((uint64_t) tail[4]) << 32;
        case 4:
            nk1 ^= ((uint64_t) tail[3]) << 24;
        case 3:
            nk1 ^= ((uint64_t) tail[2]) << 16;
        case 2:
            nk1 ^= ((uint64_t) tail[1]) << 8;
        case 1:
            nk1 ^= ((uint64_t) tail[0]) << 0;
            nk1 *= c1;
            nk1 = ROTL64(nk1, 31);
            nk1 *= c2;
            h1 ^= nk1;
    };

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

    ((uint64_t*) out)[0] = h1;
    ((uint64_t*) out)[1] = h2;
}

void bloomfilter_init(struct bloomfilter *bloomfilter, uint64_t m, uint64_t k)
{
    memset(bloomfilter, 0, sizeof(*bloomfilter));
    bloomfilter->m = m;
    bloomfilter->k = k;
    bloomfilter->magic_num = BLOOMFILTER_MAGIC_NUM_NEW;
    bloomfilter->count = 0;
    memset(bloomfilter->bit_vector, 0, bloomfilter->m >> 3);
}

int bloomfilter_check(const struct bloomfilter* filter){
    if( filter->magic_num == BLOOMFILTER_MAGIC_NUM_NEW){
        return 1;
    }else{
        fprintf(stderr, "error magic_num %ld\n", filter->magic_num);
        return 0;
    }
}

int
bloomfilter_load_32bits(struct bloomfilter **bloomfilter, FILE *fp) {
    if(fp == NULL) {
        return 0;
    }
    unsigned char bytes[4];
    struct bloomfilter* t;
    fread(bytes, 4, 1, fp);
    uint32_t magic_num = char_to_little_endian_32bits(bytes);
    if(magic_num != BLOOMFILTER_MAGIC_NUM_OLD) {
        return 0;
    }
    fread(bytes, 4, 1, fp);
    uint32_t m = char_to_little_endian_32bits(bytes);
    if(m % 8 != 0) {
        return 0;
    }
    fread(bytes, 4, 1, fp);
    uint32_t k = char_to_little_endian_32bits(bytes);

    fread(bytes, 4, 1, fp);
    uint32_t count = char_to_little_endian_32bits(bytes);
    t = (struct bloomfilter*)malloc(sizeof(struct bloomfilter)+(m>>3));
    memset(t, 0, sizeof(struct bloomfilter) + (m >> 3));
    t->m = m;
    t->k = k;
    t->magic_num = magic_num;
    t->count = count;
    fseek(fp, BLOOMFILTER_HEADER_SIZE - 16, SEEK_CUR);
    fread(t->bit_vector, m >> 3, 1, fp);
    fseek(fp, 0, SEEK_END); // seek to end of file
    unsigned int filesize = ftell(fp);
    if (filesize != m / 8 + BLOOMFILTER_HEADER_SIZE) {
        free(t);
        return 0;
    }
    *bloomfilter = t;
    return 1;
}

int
bloomfilter_load(struct bloomfilter **bloomfilter, const void *path)
{
    struct bloomfilter* t;
    unsigned char bytes[8];
    FILE * file = fopen(reinterpret_cast<const char*>(path), "rb");
    if (file != NULL) {
        if(bloomfilter_load_32bits(bloomfilter, file) > 0) {
            fclose(file);
            return 1;
        }
        //back to beginning of file
        fseek(file, 0, SEEK_SET);
        fread(bytes, 8, 1, file);
        uint64_t magic_num = char_to_little_endian_64bits(bytes);
        if(magic_num  != BLOOMFILTER_MAGIC_NUM_NEW) {
            fclose(file);
            return 0;
        }
        fread(bytes, 8, 1, file);
        uint64_t m = char_to_little_endian_64bits(bytes);
        if(m % 8 != 0) {
            fclose(file);
            return 0;
        }
        fread(bytes, 8, 1, file);
        uint64_t k = char_to_little_endian_64bits(bytes);

        fread(bytes, 8, 1, file);
        uint64_t count = char_to_little_endian_64bits(bytes);

        t = (struct bloomfilter*)malloc(sizeof(struct bloomfilter)+(m>>3));
        memset(t, 0, sizeof(struct bloomfilter) + (m >> 3));
        t->m = m;
        t->k = k;
        t->magic_num = magic_num;
        t->count = count;
        fread(t->bit_vector, m >> 3, 1, file);
        fseek(file, 0, SEEK_END); // seek to end of file
        unsigned int filesize = ftell(file);
        fclose(file);
        if(filesize != m / 8 + BLOOMFILTER_HEADER_SIZE) {
            free(t);
            return 0;
        }
        *bloomfilter = t;
        return 1;
    }
    fprintf(stderr, "file %s not exist\n", reinterpret_cast<const char*>(path));
    return 0;
}

int
bloomfilter_set(struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    if(bloomfilter_get(bloomfilter, key, len) > 0) {
        return 0;
    }
    uint32_t i;
    uint64_t  result[2];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        bit_set(bloomfilter->bit_vector, result[0]);
        bit_set(bloomfilter->bit_vector, result[1]);
    }
    bloomfilter->count++;
    return 1;
}

int
bloomfilter_set_nocheck(struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    uint32_t i;
    uint64_t  result[2];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        bit_set(bloomfilter->bit_vector, result[0]);
        bit_set(bloomfilter->bit_vector, result[1]);
    }
    bloomfilter->count++;
    return 1;
}

int
bloomfilter_get(const struct bloomfilter *bloomfilter, const void *key, size_t len)
{
    uint32_t i;
    uint64_t  result[2];

    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        result[0] %= bloomfilter->m;
        result[1] %= bloomfilter->m;
        if (!bit_get(bloomfilter->bit_vector, result[0])){
            return 0;
        }
        if (!bit_get(bloomfilter->bit_vector, result[1])){
            return 0;
        }
    }
    return 1;
}

int
bloomfilter_get_hash(struct bloomfilter *bloomfilter, const void *key, size_t len, char *dst)
{
    uint32_t i;
    uint64_t  result[2];
    char hash[255] = "";
    char valstr[32];
    for (i = 0; i < bloomfilter->k; i++) {
        murmurhash3_x64_128(key, len, i, &result);
        sprintf(valstr, "%ld,", result[0]);
        strcat(hash, valstr);
        sprintf(valstr, "%ld,", result[1]);
        strcat(hash, valstr);
    }
    strcpy(dst, hash);
    return 1;
}

int
bloomfilter_dump(struct bloomfilter *bloomfilter, const void *path)
{
    FILE * file = fopen(reinterpret_cast<const char*>(path), "wb");
    if (file != NULL) {
        fwrite(&bloomfilter->magic_num, sizeof(bloomfilter->magic_num), 1, file);
        fwrite(&bloomfilter->m, sizeof(bloomfilter->m), 1, file);
        fwrite(&bloomfilter->k, sizeof(bloomfilter->k), 1, file);
        fwrite(&bloomfilter->count, sizeof(bloomfilter->count), 1, file);
        fwrite(bloomfilter->bit_vector, (bloomfilter->m >> 3), 1, file);
        fclose(file);
        return 1;
    }
    return 0;
}

/**
 * works either big-endian or little-endian architectures
 */
uint32_t
char_to_little_endian_32bits(unsigned char *bytes) {
    return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}

/**
 * works either big-endian or little-endian architectures
 */
uint64_t
char_to_little_endian_64bits(unsigned char *bytes) {
    uint64_t bytes_ull[8];
    int i;
    for(i = 0; i < 8; i++) {
        bytes_ull[i] = bytes[i];
    }
    return bytes_ull[0] | (bytes_ull[1] << 8) | (bytes_ull[2] << 16) | (bytes_ull[3] << 24) | 
            (bytes_ull[4] << 32) | (bytes_ull[5] << 40) | (bytes_ull[6] << 48) | (bytes_ull[7] << 56);
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
