// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/cpu/mkl_math.h"

#include <glog/logging.h>
#include <mkl.h>
#include <mkl_vml_functions.h>

#include <cmath>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"
#include "paddle/common/enforce.h"

#define CINN_MKL_VECTOR_MATH_FP(fn__, name__)                             \
  void cinn_mkl_##name__##_v_fp32(cinn_buffer_t *x, cinn_buffer_t *out) { \
    PADDLE_ENFORCE_EQ(x->num_elements(),                                  \
                      out->num_elements(),                                \
                      ::common::errors::InvalidArgument(                  \
                          "X's number of elements (%d) should "           \
                          "be equal to output's (%d).",                   \
                          x->num_elements(),                              \
                          out->num_elements()));                          \
    vs##fn__(x->num_elements(),                                           \
             reinterpret_cast<float *>(x->memory),                        \
             reinterpret_cast<float *>(out->memory));                     \
  }                                                                       \
  void cinn_mkl_##name__##_v_fp64(cinn_buffer_t *x, cinn_buffer_t *out) { \
    PADDLE_ENFORCE_EQ(x->num_elements(),                                  \
                      out->num_elements(),                                \
                      ::common::errors::InvalidArgument(                  \
                          "X's number of elements (%d) should "           \
                          "be equal to output's (%d).",                   \
                          x->num_elements(),                              \
                          out->num_elements()));                          \
    vd##fn__(x->num_elements(),                                           \
             reinterpret_cast<double *>(x->memory),                       \
             reinterpret_cast<double *>(out->memory));                    \
  }

CINN_MKL_VECTOR_MATH_FP(Exp, exp);
CINN_MKL_VECTOR_MATH_FP(Erf, erf);
CINN_MKL_VECTOR_MATH_FP(Sqrt, sqrt);
CINN_MKL_VECTOR_MATH_FP(Ln, log);
CINN_MKL_VECTOR_MATH_FP(Floor, floor);
CINN_MKL_VECTOR_MATH_FP(Ceil, ceil);
CINN_MKL_VECTOR_MATH_FP(Round, round);
CINN_MKL_VECTOR_MATH_FP(Tanh, tanh);
//! Todo: current mklml.so not support
// CINN_MKL_VECTOR_MATH_FP(Log2, log2);
// CINN_MKL_VECTOR_MATH_FP(Log10, log10);
// CINN_MKL_VECTOR_MATH_FP(Trunc, trunc);
// CINN_MKL_VECTOR_MATH_FP(Cos, cos);
// CINN_MKL_VECTOR_MATH_FP(Sin, sin);
// CINN_MKL_VECTOR_MATH_FP(Cosh, cosh);
// CINN_MKL_VECTOR_MATH_FP(Tan, tan);
// CINN_MKL_VECTOR_MATH_FP(Sinh, sinh);
// CINN_MKL_VECTOR_MATH_FP(Acos, acos);
// CINN_MKL_VECTOR_MATH_FP(Acosh, acosh);
// CINN_MKL_VECTOR_MATH_FP(Asin, asin);
// CINN_MKL_VECTOR_MATH_FP(Asinh, asinh);
// CINN_MKL_VECTOR_MATH_FP(Atan, atan);
// CINN_MKL_VECTOR_MATH_FP(Atanh, atanh);

CINN_REGISTER_HELPER(mkl_math) {
  using cinn::backends::FunctionProto;

  auto host_target = cinn::common::DefaultHostTarget();

#define REGISTER_MKL_FUNCS(fn__)                                     \
  REGISTER_EXTERN_FUNC_HELPER(cinn_mkl_##fn__##_v_fp32, host_target) \
      .SetRetType<void>()                                            \
      .AddInputType<cinn_buffer_t *>()                               \
      .AddOutputType<cinn_buffer_t *>()                              \
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))   \
      .End();                                                        \
  REGISTER_EXTERN_FUNC_HELPER(cinn_mkl_##fn__##_v_fp64, host_target) \
      .SetRetType<void>()                                            \
      .AddInputType<cinn_buffer_t *>()                               \
      .AddOutputType<cinn_buffer_t *>()                              \
      .SetShapeInference(FunctionProto::ShapeFollowNthArgument(0))   \
      .End();

  REGISTER_MKL_FUNCS(exp);
  REGISTER_MKL_FUNCS(erf);
  REGISTER_MKL_FUNCS(sqrt);
  REGISTER_MKL_FUNCS(log);
  REGISTER_MKL_FUNCS(floor);
  REGISTER_MKL_FUNCS(ceil);
  REGISTER_MKL_FUNCS(round);
  REGISTER_MKL_FUNCS(tanh);
  //! Todo: current mklml.so not support
  // REGISTER_MKL_FUNCS(log2);
  // REGISTER_MKL_FUNCS(log10);
  // REGISTER_MKL_FUNCS(trunc);
  // REGISTER_MKL_FUNCS(cos);
  // REGISTER_MKL_FUNCS(sin);
  // REGISTER_MKL_FUNCS(cosh);
  // REGISTER_MKL_FUNCS(tan);
  // REGISTER_MKL_FUNCS(sinh);
  // REGISTER_MKL_FUNCS(acos);
  // REGISTER_MKL_FUNCS(acosh);
  // REGISTER_MKL_FUNCS(asin);
  // REGISTER_MKL_FUNCS(asinh);
  // REGISTER_MKL_FUNCS(atan);
  // REGISTER_MKL_FUNCS(atanh);

  return true;
}
