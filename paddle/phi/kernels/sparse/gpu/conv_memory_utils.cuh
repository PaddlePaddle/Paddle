#pragma once

#ifdef PADDLE_WITH_CUDA

template <int bytes>
struct global_load;

template <>
struct global_load<16>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint4 &data = *reinterpret_cast<uint4 *>(&D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)(pred_guard & 1)), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  }
};

template <>
struct global_load<8>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint2 const *ptr_ldg = reinterpret_cast<uint2 const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 2; ldg_idx++)
    {
      uint2 &data = *(reinterpret_cast<uint2 *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %3, 0;\n"
          "  mov.b32 %0, %4;\n"
          "  mov.b32 %1, %5;\n"
          "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
          "}\n"
          : "=r"(data.x), "=r"(data.y)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data.x), "r"(data.y));
    }
  }
};

template <>
struct global_load<4>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    unsigned const *ptr_ldg = reinterpret_cast<unsigned const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 4; ldg_idx++)
    {
      unsigned &data = *(reinterpret_cast<unsigned *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %2, 0;\n"
          "  mov.b32 %0, %3;\n"
          "  @p ld.global.u32 %0, [%1];\n"
          "}\n"
          : "=r"(data)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "r"(data));
    }
  }
};

template <>
struct global_load<2>
{
  __device__ __inline__ global_load(uint4 &D, void const *ptr, int pred_guard)
  {
    uint16_t const *ptr_ldg = reinterpret_cast<uint16_t const *>(ptr);
#pragma unroll
    for (int ldg_idx = 0; ldg_idx < 8; ldg_idx++)
    {
      uint16_t &data = *(reinterpret_cast<uint16_t *>(&D) + ldg_idx);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %2, 0;\n"
          "  mov.b16 %0, %3;\n"
          "  @p ld.global.u16 %0, [%1];\n"
          "}\n"
          : "=h"(data)
          : "l"(ptr_ldg + ldg_idx), "r"((int)(pred_guard & (1 << ldg_idx))), "h"(data));
    }
  }
};

#endif // PADDLE_WITH_CUDA
