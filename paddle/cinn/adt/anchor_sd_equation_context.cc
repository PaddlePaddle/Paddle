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

#include "paddle/cinn/adt/anchor_sd_equation_context.h"
#include "paddle/common/enforce.h"

namespace cinn::adt::config {

void GenerateScheduleMeshEquations(const ScheduleMesh& sched_mesh,
                                   const List<Iterator>& tmp_anchor_iterators,
                                   const List<Iterator>& sd_iterators,
                                   Equations* equations);

namespace {

void GenerateScheduleMeshEquationsImpl(const List<ScheduleDim>& sched_dims,
                                       const List<Iterator>& input_iterators,
                                       const List<Iterator>& output_iterators,
                                       Equations* equations) {
  PADDLE_ENFORCE_EQ(
      input_iterators->size() == output_iterators->size(),
      true,
      ::common::errors::InvalidArgument(
          "The size of input iterators and output iterators should be equal, "
          "but got input iterators size = %d, output iterators size = %d.",
          input_iterators->size(),
          output_iterators->size()));
  for (std::size_t i = 0; i < output_iterators->size(); ++i) {
    Equal(input_iterators->at(i), output_iterators->at(i), equations);
  }
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations) {
  const auto& [middle_sched_mesh, shape] = sched_reshape.tuple();
  List<Iterator> middle_iterators =
      MakeIterators(GetOutputRank(middle_sched_mesh));
  List<DimExpr> middle_dims = GetOutputDimValues(middle_sched_mesh);
  PADDLE_ENFORCE_EQ(
      shape.value()->size() == output_iterators->size(),
      true,
      ::common::errors::InvalidArgument(
          "The size of shape and output iterators should be equal, but got "
          "shape size = %d, output iterators size = %d.",
          shape.value()->size(),
          output_iterators->size()));
  List<DimExpr> output_dims = GetOutputDimValues(ScheduleMesh{sched_reshape});
  const auto& middle_index = MakeDot(middle_iterators, middle_dims, equations);
  const auto& output_index = MakeDot(output_iterators, output_dims, equations);
  Equal(middle_index, output_index, equations);

  GenerateScheduleMeshEquations(
      middle_sched_mesh, input_iterators, middle_iterators, equations);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  PADDLE_ENFORCE_EQ(GetOutputRank(sched_mesh) == output_iterators->size(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The size of output iterators should be equal to the "
                        "rank of the schedule mesh, but got output iterators "
                        "size = %d, rank of the schedule mesh = %d.",
                        output_iterators->size(),
                        GetOutputRank(sched_mesh)));
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < perm.value()->size(); ++i) {
    Equal(middle_iterators->at(perm.value()->at(i)),
          output_iterators->at(i),
          equations);
  }
  GenerateScheduleMeshEquations(
      sched_mesh, input_iterators, middle_iterators, equations);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  PADDLE_ENFORCE_EQ(GetOutputRank(sched_mesh) == output_iterators->size(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The size of output iterators should be equal to the "
                        "rank of the schedule mesh, but got output iterators "
                        "size = %d, rank of the schedule mesh = %d.",
                        output_iterators->size(),
                        GetOutputRank(sched_mesh)));
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < output_iterators->size(); ++i) {
    Equal(middle_iterators->at(i), output_iterators->at(i), equations);
  }
  GenerateScheduleMeshEquations(
      sched_mesh, input_iterators, middle_iterators, equations);
}

}  // namespace

void GenerateScheduleMeshEquations(const ScheduleMesh& sched_mesh,
                                   const List<Iterator>& tmp_anchor_iterators,
                                   const List<Iterator>& sd_iterators,
                                   Equations* equations) {
  return std::visit(
      [&](const auto& impl) {
        return GenerateScheduleMeshEquationsImpl(
            impl, tmp_anchor_iterators, sd_iterators, equations);
      },
      sched_mesh.variant());
}

void AnchorSdEquationContext::GenerateSdEquation(const ScheduleMesh& sched_mesh,
                                                 const Index& anchor_index) {
  const auto& tmp_anchor_iterators = MakeIterators(GetInputRank(sched_mesh));

  {
    const auto& anchor_dim_values =
        GetOutputDimValues(GetInputScheduleMesh(sched_mesh));
    const auto& tmp_anchor_index =
        MakeDot(tmp_anchor_iterators, anchor_dim_values, &equations_);
    Equal(tmp_anchor_index, anchor_index, &equations_);
  }

  GenerateScheduleMeshEquations(
      sched_mesh, tmp_anchor_iterators, sd_iterators_, &equations_);
}

}  // namespace cinn::adt::config
