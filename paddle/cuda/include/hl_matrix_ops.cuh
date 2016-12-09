/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef HL_MATRIX_OPS_CUH_
#define HL_MATRIX_OPS_CUH_

#include "hl_base.h"

#ifdef __NVCC__
#define HL_DEVICE   __device__
#else
#define HL_DEVICE
#endif

/**
 * @brief   parameter macro.
 */
#define ONE_PARAMETER(name)     \
        private: \
          const T p;\
        public: \
          name(const T s) : p(s) {}

#define TWO_PARAMETER(name)     \
        private: \
          const T p1;\
          const T p2;\
        public: \
          name(const T s1, T s2) : p1(s1), p2(s2) {}

#define THREE_PARAMETER(name)     \
        private: \
          const T p1;\
          const T p2;\
          const T p3;\
        public: \
          name(const T s1, T s2, T s3) : p1(s1), p2(s2), p3(s3) {}

#define FOUR_PARAMETER(name)     \
        private: \
          const T p1;\
          const T p2;\
          const T p3;\
          const T p4;\
        public: \
          name(const T s1, T s2, T s3, T s4) : p1(s1), p2(s2), p3(s3), p4(s4) {}

/**
 * @brief   unary operator macro.
 *
 * @param   name    operator name.
 * @param   op      operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b
 *
 * @see    hl_gpu_apply_unary_op
 * @see    hl_cpu_apply_unary_op
 */
#define DEFINE_MATRIX_UNARY_OP(name, op) \
    namespace unary {\
    template<class T>\
    class name {\
    public:\
        HL_DEVICE inline void gpuOperator(T &a) {op;}\
        inline void cpuOperator(T &a) {op;}\
    };\
    }


/**
 * @brief   unary operator macro.
 *
 * @param   name        operator name.
 * @param   PARA_MACRO  parameter macro.
 * @param   op          operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b
 *
 * @see    hl_gpu_apply_unary_op
 * @see    hl_cpu_apply_unary_op
 */
#define DEFINE_MATRIX_UNARY_PARAMETER_OP(name, PARA_MACRO, op) \
    namespace unary {\
    template<class T>\
    class name {\
    PARA_MACRO(name)\
    public:\
        HL_DEVICE inline void gpuOperator(T &a) {op;}\
        inline void cpuOperator(T &a) {op;}\
    };\
    }


/**
 * @brief   binary operator macro.
 *
 * @param   name    operator name.
 * @param   op      operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b
 *
 * @see    hl_gpu_apply_unary_op
 * @see    hl_cpu_apply_unary_op
 */
#define DEFINE_MATRIX_BINARY_OP(name, op) \
    namespace binary {\
    template<class T>\
    class name {\
    public:\
        HL_DEVICE inline void gpuOperator(T &a, T &b) {op;}\
        inline void cpuOperator(T &a, T &b) {op;}\
    };\
    }


/**
 * @brief   binary operator macro.
 *
 * @param   name        operator name.
 * @param   PARA_MACRO  parameter macro.
 * @param   op          operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b
 *
 * @see    hl_gpu_apply_binary_op
 * @see    hl_cpu_apply_binary_op
 */
#define DEFINE_MATRIX_BINARY_PARAMETER_OP(name, PARA_MACRO, op) \
    namespace binary {\
    template<class T>\
    class name {\
    PARA_MACRO(name)\
    public:\
        HL_DEVICE inline void gpuOperator(T &a, T &b) {op;}\
        inline void cpuOperator(T &a, T &b) {op;}\
    };\
    }


/**
 * @brief   ternary operator macro.
 *
 * @param   name    operator name.
 * @param   op      operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b, c
 *
 * @see    hl_gpu_apply_ternary_op
 * @see    hl_cpu_apply_ternary_op
 */
#define DEFINE_MATRIX_TERNARY_OP(name, op) \
    namespace ternary {\
    template<class T>\
    class name {\
    public:\
        HL_DEVICE inline void gpuOperator(T &a, T &b, T &c) {op;}\
        inline void cpuOperator(T &a, T &b, T &c) {op;}\
    };\
    }


/**
 * @brief   ternary operator macro.
 *
 * @param   name        operator name.
 * @param   PARA_MACRO  parameter macro.
 * @param   op          operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b, c
 *
 * @see    hl_gpu_apply_ternary_op
 * @see    hl_cpu_apply_ternary_op
 */
#define DEFINE_MATRIX_TERNARY_PARAMETER_OP(name, PARA_MACRO, op) \
    namespace ternary {\
    template<class T>\
    class name {\
    private:\
    PARA_MACRO(name)\
    public:\
        HL_DEVICE inline void gpuOperator(T &a, T &b, T &c) {op;}\
        inline void cpuOperator(T &a, T &b, T &c) {op;}\
    };\
    }


/**
 * @brief   quaternary operator macro.
 *
 * @param   name        operator name.
 * @param   op          operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b, c, d
 *
 * @see    hl_gpu_apply_quaternary_op
 * @see    hl_cpu_apply_quaternary_op
 */
#define DEFINE_MATRIX_QUATERNARY_OP(name, op)     \
  namespace quaternary {\
  template<class T>\
  class name {\
   public:\
   HL_DEVICE inline void gpuOperator(T &a, T &b, T &c, T &d) {op;}\
   inline void cpuOperator(T&a, T &b, T &c, T &d) {op;}\
  };\
  }


/**
 * @brief   quaternary operator macro.
 *
 * @param   name        operator name.
 * @param   PARA_MACRO  parameter macro.
 * @param   op          operator expression.
 *
 * @note   op format: op supports multiple expressions that are separated
 *         by a comma. e.g. a, b, c, d
 *
 * @see    hl_gpu_apply_quaternary_op
 * @see    hl_cpu_apply_quaternary_op
 */
#define DEFINE_MATRIX_QUATERNARY_PARAMETER_OP(name, PARA_MACRO, op)     \
  namespace quaternary {\
  template<class T>\
  class name {\
   private:\
   PARA_MACRO(name)\
   public:\
   HL_DEVICE inline void gpuOperator(T &a, T &b, T &c, T &d) {op;}\
   inline void cpuOperator(T &a, T &b, T &c, T &d) {op;}\
  };\
  }

#endif /* HL_MATRIX_OPS_CUH_ */
