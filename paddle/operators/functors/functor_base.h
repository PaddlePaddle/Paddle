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

#pragma once

#include "paddle/framework/eigen.h"

namespace paddle {
namespace operators {
namespace functors {

template <typename AttrReaderType, typename T>
struct FunctorBase {
  using ElemType = T;
  using AttrReader = AttrReaderType;
};

template <typename Derived>
struct UnaryFunctor : public Derived {
  static constexpr size_t IN_NUM = 1;
  static constexpr size_t OUT_NUM = 1;

  UnaryFunctor(const typename Derived::AttrReader& reader) : Derived(reader) {}

  template <typename Device>
  void operator()(Device& dev, const framework::Tensor& in,
                  framework::Tensor* out) const {
    auto i = framework::EigenVector<typename Derived::ElemType>::Flatten(in);
    auto o = framework::EigenVector<typename Derived::ElemType>::Flatten(*out);
    this->apply(o.device(dev), i);
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
