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

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_util.h"

namespace cinn::adt::equation::config {

class AnchorSdEquationContext final {
 public:
  AnchorSdEquationContext(const AnchorSdEquationContext&) = default;
  AnchorSdEquationContext(AnchorSdEquationContext&&) = default;
  AnchorSdEquationContext& operator=(const AnchorSdEquationContext&) = default;
  AnchorSdEquationContext& operator=(AnchorSdEquationContext&&) = default;

  AnchorSdEquationContext(std::size_t num_strides)
      : strides_(util::MakeStrides(num_strides)),
        sd_iterators_(util::MakeIterators(num_strides)) {}

  void GenerateSdEquation(const Index& tensor_index) {
    const auto& sd_index = util::MakeDot(sd_iterators_, strides_, &equations_);
    util::Equal(sd_index, tensor_index, &equations_);
  }

  const List<Stride>& strides() const { return strides_; }

  const List<Iterator>& sd_iterators() const { return sd_iterators_; }

  const Equations& equations() const { return equations_; }

 private:
  List<Stride> strides_;
  List<Iterator> sd_iterators_;
  Equations equations_;
};

}  // namespace cinn::adt::equation::config
