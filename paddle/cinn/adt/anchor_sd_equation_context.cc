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
    std::unordered_map<Stride, const Constant>* stride2constant);

namespace {

void GenerateScheduleMeshEquationsImpl(
    const List<ScheduleDim>& sched_dims,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Stride, const Constant>* stride2constant) {
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
    std::unordered_map<Stride, const Constant>* stride2constant) {
  const auto& [middle_sched_mesh, shape] = sched_reshape.tuple();
  List<Iterator> middle_iterators =
      MakeIterators(GetOutputRank(middle_sched_mesh));
  List<Stride> middle_strides = MakeStrides(GetOutputRank(middle_sched_mesh));
  CHECK_EQ(shape.value()->size(), output_iterators->size());
  List<Stride> output_strides = MakeStrides(output_iterators->size());
  {
    List<Constant> middle_stride_values =
        GetOutputStrideValues(middle_sched_mesh);
    for (std::size_t i = 0; i < middle_stride_values->size(); ++i) {
      CHECK(stride2constant
                ->emplace(middle_strides->at(i), middle_stride_values->at(i))
                .second);
    }

    List<Constant> output_stride_values =
        GetOutputStrideValues(ScheduleMesh{sched_reshape});
    for (std::size_t i = 0; i < output_strides->size(); ++i) {
      CHECK(stride2constant
                ->emplace(output_strides->at(i), output_stride_values->at(i))
                .second);
    }
  }
  const auto& middle_index =
      MakeDot(middle_iterators, middle_strides, equations);
  const auto& output_index =
      MakeDot(output_iterators, output_strides, equations);
  Equal(middle_index, output_index, equations);

  GenerateScheduleMeshEquations(middle_sched_mesh,
                                input_iterators,
                                middle_iterators,
                                equations,
                                stride2constant);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Stride, const Constant>* stride2constant) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  CHECK_EQ(GetOutputRank(sched_mesh), output_iterators->size());
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < perm.value()->size(); ++i) {
    Equal(middle_iterators->at(perm.value()->at(i)),
          output_iterators->at(i),
          equations);
  }
  GenerateScheduleMeshEquations(sched_mesh,
                                input_iterators,
                                middle_iterators,
                                equations,
                                stride2constant);
}

void GenerateScheduleMeshEquationsImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding,
    const List<Iterator>& input_iterators,
    const List<Iterator>& output_iterators,
    Equations* equations,
    std::unordered_map<Stride, const Constant>* stride2constant) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  CHECK_EQ(GetOutputRank(sched_mesh), output_iterators->size());
  List<Iterator> middle_iterators = MakeIterators(output_iterators->size());
  for (std::size_t i = 0; i < output_iterators->size(); ++i) {
    Equal(middle_iterators->at(i), output_iterators->at(i), equations);
  }
  GenerateScheduleMeshEquations(sched_mesh,
                                input_iterators,
                                middle_iterators,
                                equations,
                                stride2constant);
}

}  // namespace

void GenerateScheduleMeshEquations(
    const ScheduleMesh& sched_mesh,
    const List<Iterator>& tmp_anchor_iterators,
    const List<Iterator>& sd_iterators,
    Equations* equations,
    std::unordered_map<Stride, const Constant>* stride2constant) {
  return std::visit(
      [&](const auto& impl) {
        return GenerateScheduleMeshEquationsImpl(impl,
                                                 tmp_anchor_iterators,
                                                 sd_iterators,
                                                 equations,
                                                 stride2constant);
      },
      sched_mesh.variant());
}

void AnchorSdEquationContext::InitStride2Constant(
    const ScheduleMesh& sched_mesh) {
  const auto& AddStrideValue = [&](const List<Stride>& strides,
                                   const List<Constant>& stride_values) {
    CHECK_EQ(strides->size(), stride_values->size());
    for (std::size_t i = 0; i < strides->size(); ++i) {
      CHECK(stride2constant_.emplace(strides->at(i), stride_values->at(i))
                .second);
    }
  };

  const auto& anchor_stride_values =
      GetOutputStrideValues(GetInputScheduleMesh(sched_mesh));
  AddStrideValue(anchor_strides_, anchor_stride_values);

  const auto& sd_stride_values = GetOutputStrideValues(sched_mesh);
  AddStrideValue(sd_strides_, sd_stride_values);
}

void AnchorSdEquationContext::GenerateSdEquation(const ScheduleMesh& sched_mesh,
                                                 const Index& anchor_index) {
  const auto& tmp_anchor_iterators = MakeIterators(GetInputRank(sched_mesh));

  {
    const auto& tmp_anchor_index =
        MakeDot(tmp_anchor_iterators, anchor_strides_, &equations_);
    Equal(tmp_anchor_index, anchor_index, &equations_);
  }

  GenerateScheduleMeshEquations(sched_mesh,
                                tmp_anchor_iterators,
                                sd_iterators_,
                                &equations_,
                                &stride2constant_);
}

}  // namespace cinn::adt::config
