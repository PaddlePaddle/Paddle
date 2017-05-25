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

#ifndef HL_CPU_SCALAR_CUH_
#define HL_CPU_SCALAR_CUH_

#ifndef PADDLE_TYPE_DOUBLE
/* size of float */
#define VECTOR_SIZE     4
#else
/* size of double */
#define VECTOR_SIZE     8
#endif

typedef real vecType;

inline void set_zero(vecType &mm) { mm = (vecType) 0.0f; }

/* Consider a real as a vector */
#define VECTOR_LEN      1
#define VECTOR_SET      set_zero

template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  return mm;
}

#endif  // HL_CPU_SCALAR_CUH_
