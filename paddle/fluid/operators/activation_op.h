/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <type_traits>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/float16.h"

#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace paddle {
namespace operators {

using phi::To32BitIndex;

using ActBwdOpFwdDeps = phi::funcs::ActBwdOpFwdDeps;

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

#define USE_PHI_FUNCTOR(name)                         \
  template <typename T>                               \
  using name##Functor = phi::funcs::name##Functor<T>; \
  template <typename T>                               \
  using name##GradFunctor = phi::funcs::name##GradFunctor<T>;

#define USE_PHI_DOUBLE_GRAD_FUNCTOR(name) \
  template <typename T>                   \
  using name##GradGradFunctor = phi::funcs::name##GradGradFunctor<T>;

USE_PHI_FUNCTOR(Mish)

}  // namespace operators
}  // namespace paddle
