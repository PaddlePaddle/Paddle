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

#include "paddle/cinn/adt/print_utils/print_schedule_mesh.h"
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"
#include "paddle/cinn/adt/print_utils/print_schedule_dim.h"
#include "paddle/cinn/adt/schedule_mesh.h"

namespace cinn::adt {

std::string ToTxtString(const List<int>& ints) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < ints->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += std::to_string(ints.Get(idx));
  }
  ret += "]";
  return ret;
}

namespace {

std::string ToTxtString(const tMeshDim<List<LoopSize>>& mesh_dim_loop_sizes) {
  std::string ret;
  ret += "dims=" + ToTxtString(mesh_dim_loop_sizes.value());
  return ret;
}

std::string ToTxtString(const tMeshPerm<List<int>>& mesh_perm_ints) {
  std::string ret;
  ret += "perm=" + ToTxtString(mesh_perm_ints.value());
  return ret;
}

std::string ToTxtString(
    const tMeshPaddingTo<List<LoopSize>>& mesh_padding_loop_sizes) {
  std::string ret;
  ret += "padding_to=" + ToTxtString(mesh_padding_loop_sizes.value());
  return ret;
}

std::string ToTxtStringMeshImpl(const List<ScheduleDim>& schedule_dims) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < schedule_dims->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(schedule_dims.Get(idx));
  }
  ret += "]";

  return ret;
}

std::string ToTxtStringMeshImpl(
    const ScheduleMeshReshape<ScheduleMesh>& schedule_mesh_reshape) {
  std::string ret;
  const auto& [schedule_mesh, loop_sizes] = schedule_mesh_reshape.tuple();
  ret += ToTxtString(schedule_mesh);
  ret += ".reshape(";
  ret += ToTxtString(loop_sizes);
  ret += ")";
  return ret;
}

std::string ToTxtStringMeshImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& schedule_mesh_transpose) {
  std::string ret;
  const auto& [schedule_mesh, loop_sizes] = schedule_mesh_transpose.tuple();
  ret += ToTxtString(schedule_mesh);
  ret += ".transpose(";
  ret += ToTxtString(loop_sizes);
  ret += ")";
  return ret;
}

std::string ToTxtStringMeshImpl(
    const ScheduleMeshPadding<ScheduleMesh>& schedule_mesh_padding) {
  std::string ret;
  const auto& [schedule_mesh, loop_sizes] = schedule_mesh_padding.tuple();
  ret += ToTxtString(schedule_mesh);
  ret += ".padding(";
  ret += ToTxtString(loop_sizes);
  ret += ")";
  return ret;
}

}  // namespace

std::string ToTxtString(const ScheduleMesh& schedule_mesh) {
  return std::visit([&](const auto& impl) { return ToTxtStringMeshImpl(impl); },
                    schedule_mesh.variant());
}

}  // namespace cinn::adt
