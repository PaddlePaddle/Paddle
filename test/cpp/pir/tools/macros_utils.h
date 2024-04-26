// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/pir/include/core/type_id.h"

#define IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TYPE_CLASS) \
  namespace pir {                                    \
  namespace detail {                                 \
  template <>                                        \
  class IR_API TypeIdResolver<TYPE_CLASS> {          \
   public:                                           \
    static TypeId Resolve() { return id_; }          \
    static UniqueingId id_;                          \
  };                                                 \
  }                                                  \
  }  // namespace pir
