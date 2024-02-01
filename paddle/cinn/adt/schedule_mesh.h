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

#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/schedule_dim.h"

namespace cinn::adt {

DEFINE_ADT_TAG(tMeshDim);
DEFINE_ADT_TAG(tMeshPerm);
DEFINE_ADT_TAG(tMeshPaddingTo);

template <typename T>
class ScheduleMeshReshape final : public Tuple<T, tMeshDim<List<LoopSize>>> {
 public:
  using Tuple<T, tMeshDim<List<LoopSize>>>::Tuple;
};

template <typename T>
class ScheduleMeshTranspose final : public Tuple<T, tMeshPerm<List<int>>> {
 public:
  using Tuple<T, tMeshPerm<List<int>>>::Tuple;
};

template <typename T>
class ScheduleMeshPadding final
    : public Tuple<T, tMeshPaddingTo<List<LoopSize>>> {
 public:
  using Tuple<T, tMeshPaddingTo<List<LoopSize>>>::Tuple;
};

DEFINE_ADT_UNION(ScheduleMesh,
                 List<ScheduleDim>,
                 ScheduleMeshReshape<ScheduleMesh>,
                 ScheduleMeshTranspose<ScheduleMesh>,
                 ScheduleMeshPadding<ScheduleMesh>);

ScheduleMesh MeshReshape(const ScheduleMesh& sched_mesh,
                         const std::vector<std::int64_t>& shape);

ScheduleMesh MeshTranspose(const ScheduleMesh& sched_mesh,
                           const List<int>& perm);

ScheduleMesh MeshPadding(const ScheduleMesh& sched_mesh,
                         const List<LoopSize>& padding_to);

ScheduleMesh MeshPaddingRoundUp(
    const ScheduleMesh& sched_mesh,
    const std::vector<std::optional<std::int64_t>>& align_size);

std::size_t GetInputRank(const ScheduleMesh& sched_mesh);

std::size_t GetOutputRank(const ScheduleMesh& sched_mesh);

List<DimExpr> GetOutputDimValues(const ScheduleMesh& sched_mesh);

ScheduleMesh GetInputScheduleMesh(const ScheduleMesh& sched_mesh);

std::tuple<ScheduleMesh, List<LoopType>> CreateOptimizedScheduleMesh(
    const List<ScheduleDim>& loop_sizes);

}  // namespace cinn::adt
