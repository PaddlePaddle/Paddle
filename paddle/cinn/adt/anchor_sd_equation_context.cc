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

namespace cinn::adt::config {

void GenerateScheduleMeshEquations(
    const ScheduleMesh& sched_mesh,
    const List<Iterator>& tmp_anchor_iterators,
    const List<Iterator>& sd_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant);

namespace {

void GenerateScheduleMeshEquationsImpl(
    const List<ScheduleDim>& sched_dims,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant) {
  CHECK_EQ(input_iterators->size(), output_iterators->size());
  for (std::size_t i = 0; i < output_iterators->size(); ++i) {
    Equal(input_iterators->at(i), output_iterators->at(i), equations);
  }
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant) {
  const auto& [middle_sched_mesh, shape] = sched_reshape.tuple();
  List<Iterator> middle_iterators =
      MakeIterators(GetOutputRank(middle_sched_mesh));
  List<Dim> middle_dims = MakeDims(GetOutputRank(middle_sched_mesh));
  CHECK_EQ(shape.value()->size(), output_iterators->size());
  List<Dim> output_dims = MakeDims(output_iterators->size());
  {
    List<Constant> middle_dim_values = GetOutputDimValues(middle_sched_mesh);
    for (std::size_t i = 0; i < middle_dim_values->size(); ++i) {
      CHECK(dim2constant->emplace(middle_dims->at(i), middle_dim_values->at(i))
                .second);
    }

    List<Constant> output_dim_values =
        GetOutputDimValues(ScheduleMesh{sched_reshape});
    for (std::size_t i = 0; i < output_dims->size(); ++i) {
      CHECK(dim2constant->emplace(output_dims->at(i), output_dim_values->at(i))
                .second);
    }
  }
  const auto& middle_index = MakeDot(middle_iterators, middle_dims, equations);
  const auto& output_index = MakeDot(output_iterators, output_dims, equations);
  Equal(middle_index, output_index, equations);

  GenerateScheduleMeshEquations(middle_sched_mesh,
                                input_iterators,
                                middle_iterators,
                                equations,
                                dim2constant);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  CHECK_EQ(GetOutputRank(sched_mesh), output_iterators->size());
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < perm.value()->size(); ++i) {
    Equal(middle_iterators->at(perm.value()->at(i)),
          output_iterators->at(i),
          equations);
  }
  GenerateScheduleMeshEquations(
      sched_mesh, input_iterators, middle_iterators, equations, dim2constant);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  CHECK_EQ(GetOutputRank(sched_mesh), output_iterators->size());
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < output_iterators->size(); ++i) {
    Equal(middle_iterators->at(i), output_iterators->at(i), equations);
  }
  GenerateScheduleMeshEquations(
      sched_mesh, input_iterators, middle_iterators, equations, dim2constant);
}

}  // namespace

void GenerateScheduleMeshEquations(
    const ScheduleMesh& sched_mesh,
    const List<Iterator>& tmp_anchor_iterators,
    const List<Iterator>& sd_iterators,
    Equations* equations,
    std::unordered_map<Dim, const Constant>* dim2constant) {
  return std::visit(
      [&](const auto& impl) {
        return GenerateScheduleMeshEquationsImpl(
            impl, tmp_anchor_iterators, sd_iterators, equations, dim2constant);
      },
      sched_mesh.variant());
}

void AnchorSdEquationContext::InitDim2Constant(const ScheduleMesh& sched_mesh) {
  const auto& AddDimValue = [&](const List<Dim>& dims,
                                const List<Constant>& dim_values) {
    CHECK_EQ(dims->size(), dim_values->size());
    for (std::size_t i = 0; i < dims->size(); ++i) {
      CHECK(dim2constant_.emplace(dims->at(i), dim_values->at(i)).second);
    }
  };

  const auto& anchor_dim_values =
      GetOutputDimValues(GetInputScheduleMesh(sched_mesh));
  AddDimValue(anchor_dims_, anchor_dim_values);

  const auto& sd_dim_values = GetOutputDimValues(sched_mesh);
  AddDimValue(sd_dims_, sd_dim_values);
}

void AnchorSdEquationContext::GenerateSdEquation(const ScheduleMesh& sched_mesh,
                                                 const Index& anchor_index) {
  const auto& tmp_anchor_iterators = MakeIterators(GetInputRank(sched_mesh));

  {
    const auto& tmp_anchor_index =
        MakeDot(tmp_anchor_iterators, anchor_dims_, &equations_);
    Equal(tmp_anchor_index, anchor_index, &equations_);
  }

  GenerateScheduleMeshEquations(sched_mesh,
                                tmp_anchor_iterators,
                                sd_iterators_,
                                &equations_,
                                &dim2constant_);
}

}  // namespace cinn::adt::config
