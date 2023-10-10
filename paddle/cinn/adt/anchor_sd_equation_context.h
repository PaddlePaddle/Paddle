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
#include "paddle/cinn/adt/schedule_mesh.h"

namespace cinn::adt::config {

using AnchorIndex = Index;

class AnchorSdEquationContext final {
 public:
  AnchorSdEquationContext(const AnchorSdEquationContext&) = default;
  AnchorSdEquationContext(AnchorSdEquationContext&&) = default;
  AnchorSdEquationContext& operator=(const AnchorSdEquationContext&) = default;
  AnchorSdEquationContext& operator=(AnchorSdEquationContext&&) = default;

  AnchorSdEquationContext(const ScheduleMesh& sched_mesh,
                          const AnchorIndex& anchor_index)
      : sd_strides_(MakeStrides(GetOutputRank(sched_mesh))),
        sd_iterators_(MakeIterators(GetOutputRank(sched_mesh))),
        anchor_strides_(MakeStrides(GetInputRank(sched_mesh))) {
    InitStride2Constant(sched_mesh);
    GenerateSdEquation(sched_mesh, anchor_index);
  }

  const List<Stride>& sd_strides() const { return sd_strides_; }

  const List<Stride>& anchor_strides() const { return anchor_strides_; }

  const List<Iterator>& sd_iterators() const { return sd_iterators_; }

  const Equations& equations() const { return equations_; }

  const std::unordered_map<Stride, const Constant>& stride2constant() const {
    return stride2constant_;
  }

 private:
  void InitStride2Constant(const ScheduleMesh& sched_mesh);

  void GenerateSdEquation(const ScheduleMesh& sched_mesh,
                          const Index& tensor_index);

  List<Stride> sd_strides_;
  List<Iterator> sd_iterators_;
  List<Stride> anchor_strides_;
  Equations equations_;
  std::unordered_map<Stride, const Constant> stride2constant_;
};

}  // namespace cinn::adt::config
