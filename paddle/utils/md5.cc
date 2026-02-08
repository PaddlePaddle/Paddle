// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from ulwanski md5 project
// Copyright (c) 2021 Marek Ulwański
// Licensed under the MIT License -
// https://github.com/ulwanski/md5/blob/master/LICENSE

#include "paddle/utils/md5.h"
#include <cstdint>
namespace paddle {
#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))
#define STEP(f, a, b, c, d, x, t, s)                       \
  (a) += f((b), (c), (d)) + (x) + (t);                     \
  (a) = (((a) << (s)) | (((a)&0xffffffff) >> (32 - (s)))); \
  (a) += (b);

#if defined(__i386__) || defined(__x86_64__) || defined(__vax__)
#define SET(n) (*reinterpret_cast<const MD5_u32 *>(&ptr[(n)*4]))
#define GET(n) SET(n)
#else
#define SET(n)                                                              \
  (ctx->block[(n)] = (MD5_u32)ptr[(n)*4] | ((MD5_u32)ptr[(n)*4 + 1] << 8) | \
                     ((MD5_u32)ptr[(n)*4 + 2] << 16) |                      \
                     ((MD5_u32)ptr[(n)*4 + 3] << 24))
#define GET(n) (ctx->block[(n)])
#endif
typedef uint32_t MD5_u32;

typedef struct {
  MD5_u32 lo, hi;
  MD5_u32 a, b, c, d;
  unsigned char buffer[64];
  MD5_u32 block[16];
} MD5_CTX;

static void MD5_Init(MD5_CTX *ctx);
static void MD5_Update(MD5_CTX *ctx, const void *data, size_t size);
static void MD5_Final(unsigned char *result, MD5_CTX *ctx);

static const void *body(MD5_CTX *ctx, const void *data, size_t size) {
  const unsigned char *ptr;
  MD5_u32 a, b, c, d;
  MD5_u32 saved_a, saved_b, saved_c, saved_d;

  ptr = (const unsigned char *)data;

  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  do {
    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;

    ptr += 64;
  } while (size -= 64);

  ctx->a = a;
  ctx->b = b;
  ctx->c = c;
  ctx->d = d;

  return ptr;
}

void MD5_Init(MD5_CTX *ctx) {
  ctx->a = 0x67452301;
  ctx->b = 0xefcdab89;
  ctx->c = 0x98badcfe;
  ctx->d = 0x10325476;

  ctx->lo = 0;
  ctx->hi = 0;
}

void MD5_Update(MD5_CTX *ctx, const void *data, size_t size) {
  MD5_u32 saved_lo;
  size_t used, free;

  saved_lo = ctx->lo;
  if ((ctx->lo = (saved_lo + size) & 0x1fffffff) < saved_lo) ctx->hi++;
  ctx->hi += size >> 29;
  used = saved_lo & 0x3f;

  if (used) {
    free = 64 - used;
    if (size < free) {
      memcpy(&ctx->buffer[used], data, size);
      return;
    }

    memcpy(&ctx->buffer[used], data, free);
    data = (unsigned char *)data + free;
    size -= free;
    body(ctx, ctx->buffer, 64);
  }

  if (size >= 64) {
    data = body(ctx, data, size & ~static_cast<size_t>(0x3f));
    size &= 0x3f;
  }

  memcpy(ctx->buffer, data, size);
}

void MD5_Final(unsigned char *result, MD5_CTX *ctx) {
  size_t used, free;
  used = ctx->lo & 0x3f;
  ctx->buffer[used++] = 0x80;
  free = 64 - used;

  if (free < 8) {
    memset(&ctx->buffer[used], 0, free);
    body(ctx, ctx->buffer, 64);
    used = 0;
    free = 64;
  }

  memset(&ctx->buffer[used], 0, free - 8);

  ctx->lo <<= 3;
  ctx->buffer[56] = ctx->lo;
  ctx->buffer[57] = ctx->lo >> 8;
  ctx->buffer[58] = ctx->lo >> 16;
  ctx->buffer[59] = ctx->lo >> 24;
  ctx->buffer[60] = ctx->hi;
  ctx->buffer[61] = ctx->hi >> 8;
  ctx->buffer[62] = ctx->hi >> 16;
  ctx->buffer[63] = ctx->hi >> 24;
  body(ctx, ctx->buffer, 64);
  result[0] = ctx->a;
  result[1] = ctx->a >> 8;
  result[2] = ctx->a >> 16;
  result[3] = ctx->a >> 24;
  result[4] = ctx->b;
  result[5] = ctx->b >> 8;
  result[6] = ctx->b >> 16;
  result[7] = ctx->b >> 24;
  result[8] = ctx->c;
  result[9] = ctx->c >> 8;
  result[10] = ctx->c >> 16;
  result[11] = ctx->c >> 24;
  result[12] = ctx->d;
  result[13] = ctx->d >> 8;
  result[14] = ctx->d >> 16;
  result[15] = ctx->d >> 24;
  memset(ctx, 0, sizeof(*ctx));
}

/* Return Calculated raw result(always little-endian), the size is always 16 */
static void md5bin(const void *data, size_t len, unsigned char out[16]) {
  MD5_CTX c;
  MD5_Init(&c);
  MD5_Update(&c, data, len);
  MD5_Final(out, &c);
}

static char hb2hex(unsigned char hb) {
  hb = hb & 0xF;
  return hb < 10 ? '0' + hb : hb - 10 + 'a';
}

std::string md5(const void *data, size_t len) {
  std::string res;
  unsigned char out[16];
  md5bin(data, len, out);
  for (size_t i = 0; i < 16; ++i) {
    res.push_back(hb2hex(out[i] >> 4));
    res.push_back(hb2hex(out[i]));
  }
  return res;
}
std::string md5(std::string data) { return md5(data.c_str(), data.length()); }
}  // namespace paddle
