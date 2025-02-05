/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/visit_type.h"

namespace paddle {

// Note: Keep this file only for compatibility with custom operators

///////// Floating Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  PD_VISIT_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

#define PD_DISPATCH_FLOATING_AND_HALF_TYPES(TYPE, NAME, ...) \
  PD_VISIT_FLOATING_AND_HALF_TYPES(TYPE, NAME, __VA_ARGS__)

///////// Integral Dispatch Marco ///////////

#define PD_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  PD_VISIT_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

///////// Complex Dispatch Marco ///////////

#define PD_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  PD_VISIT_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

///////// Floating and Integral Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(TYPE, NAME, ...) \
  PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

///////// Floating and Complex Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  PD_VISIT_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

///////// Floating and Complex and other type Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_COMPLEX_AND_1_TYPE( \
    SPECIFIED_TYPE, TYPE, NAME, ...)                 \
  PD_VISIT_FLOATING_AND_COMPLEX_AND_1_TYPE(          \
      SPECIFIED_TYPE, TYPE, NAME, __VA_ARGS__)

///////// Floating and Complex and 2 other type Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_COMPLEX_AND_2_TYPES(  \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, ...) \
  PD_VISIT_FLOATING_AND_COMPLEX_AND_2_TYPES(           \
      SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, __VA_ARGS__)

///////// Floating, Integral and Complex Dispatch Marco ///////////

#define PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  PD_VISIT_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(TYPE, NAME, __VA_ARGS__)

}  // namespace paddle
