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

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static constexpr char predefined_cuda_functions_fp32[] = R"(
__device__ inline float Max(float x, float y) { return fmaxf(x, y); }
__device__ inline float Exp(float x) { return expf(x); }
__device__ inline float Log(float x) { return logf(x); }
__device__ inline float Sqrt(float x) { return sqrtf(x); }

)";

static constexpr char predefined_cuda_functions_fp64[] = R"(
__device__ inline double Max(double x, double y) { return fmax(x, y); }
__device__ inline double Exp(double x) { return exp(x); }
__device__ inline double Log(double x) { return log(x); }
__device__ inline double Sqrt(double x) { return sqrt(x); }

)";
#ifdef PADDLE_WITH_HIP
static constexpr char predefined_cuda_functions_fp16[] = R"(
__device__ inline __half Exp(const __half x) { return hexp(x); }
__device__ inline __half Log(const __half x) { return hlog(x); }
__device__ inline __half Sqrt(const __half x) { return hsqrt(x); }
)";
#else
// List some built-in functions of __half implemented in cuda_fp16.hpp
static constexpr char predefined_cuda_functions_fp16[] = R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))

struct __align__(2) __half {
  __device__ __half() { }

 protected:
  unsigned short __x;
};

__device__ __half __float2half(const float f) {
  __half val;
  asm("{ cvt.rn.f16.f32 %0, %1; }\n" : "=h"(__HALF_TO_US(val)

) : "f"(f));
  return val;
}

__device__ float __half2float(const __half h) {
  float val;
  asm("{ cvt.f32.f16 %0, %1; }\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}

#define __CUDA_FP16_DECL__ __host__ __device__
/******************************************************************************
*                             __half comparison                              *
******************************************************************************/
#define __COMPARISON_OP_HALF_MACRO(name) do {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp."#name".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(__HALF_TO_CUS(a)), "h"(__HALF_TO_CUS(b))); \
   return val ? true : false; \
} while(0);
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(eq);
}
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ne);
}
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(le);
}
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ge);
}
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(lt);
}
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(gt);
}
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(equ);
}
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(neu);
}
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(leu);
}
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(geu);
}
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(ltu);
}
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b)
{
    __COMPARISON_OP_HALF_MACRO(gtu);
}
#undef __COMPARISON_OP_HALF_MACRO

/******************************************************************************
*                             __half arithmetic                             *
******************************************************************************/
#define __BINARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(add);
}
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(sub);
}
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(mul);
}
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(add.sat);
}
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
{
    __BINARY_OP_HALF_MACRO(mul.sat);
}
#undef __BINARY_OP_HALF_MACRO
#define __TERNARY_OP_HALF_MACRO(name) do {\
   __half val; \
   asm( "{"#name".f16 %0,%1,%2,%3;\n}" \
        :"=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)),"h"(__HALF_TO_CUS(b)),"h"(__HALF_TO_CUS(c))); \
   return val; \
} while(0);
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn.sat);
}
#undef __TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ __half __hdiv(__half a, __half b) {
    __half v, abs, den;
    __HALF_TO_US(den) = 0x008F;
    float fa, fb, fv, rcp;

    fa = __half2float(a);
    fb = __half2float(b);

    asm("{rcp.approx.f32 %0, %1;\n}" :"=f"(rcp) : "f"(fb));

    fv = rcp * fa;

    v = __float2half(fv);
    __HALF_TO_US(abs) = (unsigned short)(((unsigned int)__HALF_TO_CUS(v)) & 0x00007FFF);
    if (__hlt(abs, den) && (!(__HALF_TO_CUS(abs) == 0x0000))) {
        float err = __fmaf_rn(-fb, fv, fa);
        fv = __fmaf_rn(rcp, err, fv);
        v = __float2half(fv);
    }
    return v;
}

__CUDA_FP16_DECL__ __half __hneg(const __half a)
{
    __half zero;
    zero = __float2half(0.0);
    return __hsub(zero, a);
}

/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __half operator+(const __half &lh, const __half &rh) { return __hadd(lh, rh); }
__device__ __forceinline__ __half operator-(const __half &lh, const __half &rh) { return __hsub(lh, rh); }
__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
__device__ __forceinline__ __half operator/(const __half &lh, const __half &rh) { return __hdiv(lh, rh); }

/* Unary plus and inverse operators */
__device__ __forceinline__ __half operator+(const __half &h) { return h; }
__device__ __forceinline__ __half operator-(const __half &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __half &lh, const __half &rh) { return __heq(lh, rh); }
__device__ __forceinline__ bool operator!=(const __half &lh, const __half &rh) { return __hne(lh, rh); }
__device__ __forceinline__ bool operator> (const __half &lh, const __half &rh) { return __hgt(lh, rh); }
__device__ __forceinline__ bool operator< (const __half &lh, const __half &rh) { return __hlt(lh, rh); }
__device__ __forceinline__ bool operator>=(const __half &lh, const __half &rh) { return __hge(lh, rh); }
__device__ __forceinline__ bool operator<=(const __half &lh, const __half &rh) { return __hle(lh, rh); }

#define __SPEC_CASE(i,r, spc, ulp) \
   "{.reg.b16 spc, ulp, p;\n"\
   "  mov.b16 spc,"#spc";\n"\
   "  mov.b16 ulp,"#ulp";\n"\
   "  set.eq.f16.f16 p,"#i", spc;\n"\
   "  fma.rn.f16 "#r",p,ulp,"#r";\n}\n"

__CUDA_FP16_DECL__ __half hexp(const __half a) {
    __half val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         h,r;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  mov.b32         C, 0x3fb8aa3b;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f,f;        \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X1F79, 0x9400)
        __SPEC_CASE(h, r, 0X25CF, 0x9400)
        __SPEC_CASE(h, r, 0XC13B, 0x0400)
        __SPEC_CASE(h, r, 0XC1EF, 0x0200)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}

__CUDA_FP16_DECL__ __half hlog(const __half a) {
    __half val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         r,h;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  lg2.approx.f32      f,f;        \n"
        "  mov.b32         C, 0x3f317218;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X160D, 0x9C00)
        __SPEC_CASE(h, r, 0X3BFE, 0x8010)
        __SPEC_CASE(h, r, 0X3C0B, 0x8080)
        __SPEC_CASE(h, r, 0X6051, 0x1C00)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));
    return val;
}

#define __APPROX_FCAST(fun) do {\
   __half val;\
   asm("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  cvt.f32.f16     f,r;      \n"\
                "  "#fun".approx.f32   f,f;  \n"\
                "  cvt.rn.f16.f32      r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(__HALF_TO_US(val)) : "h"(__HALF_TO_CUS(a)));\
   return val;\
} while(0);
__CUDA_FP16_DECL__ __half hsqrt(const __half a) {
    __APPROX_FCAST(sqrt);
}

#if defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_FP16_DECL__ __half __ldg(const __half *ptr)
{
    __half ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(__HALF_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}

#undef __LDG_PTR
#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/

__device__ inline __half Exp(const __half x) { return hexp(x); }
__device__ inline __half Log(const __half x) { return hlog(x); }
__device__ inline __half Sqrt(const __half x) { return hsqrt(x); }

#undef __HALF_TO_US
#undef __HALF_TO_CUS

typedef __half float16;

)";
#endif
static constexpr char cuda_kernel_template_1d[] = R"(
extern "C" __global__ void $func_name($parameters) {
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < N;
      idx += gridDim.x * blockDim.x) {
    $compute_body
  }
}
)";
}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
