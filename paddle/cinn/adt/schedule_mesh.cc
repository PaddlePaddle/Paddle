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

#include "paddle/cinn/adt/schedule_mesh.h"
#include "paddle/cinn/adt/print.h"

namespace cinn::adt {

namespace {

std::size_t GetInputRankImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims->size();
}

std::size_t GetInputRankImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [sched_mesh, _] = sched_reshape.tuple();
  return GetInputRank(sched_mesh);
}

std::size_t GetInputRankImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, _] = sched_transpose.tuple();
  return GetInputRank(sched_mesh);
}

std::size_t GetInputRankImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  return GetInputRank(sched_mesh);
}

}  // namespace

std::size_t GetInputRank(const ScheduleMesh& sched_mesh) {
  return std::visit([&](const auto& impl) { return GetInputRankImpl(impl); },
                    sched_mesh.variant());
}

namespace {

std::size_t GetOutputRankImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [_, shapes] = sched_reshape.tuple();
  return shapes.value()->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  CHECK_EQ(GetOutputRank(sched_mesh), perm.value()->size());
  return perm.value()->size();
}

std::size_t GetOutputRankImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [_, padding_to] = sched_padding.tuple();
  return padding_to.value()->size();
}

}  // namespace

std::size_t GetOutputRank(const ScheduleMesh& sched_mesh) {
  return std::visit([&](const auto& impl) { return GetOutputRankImpl(impl); },
                    sched_mesh.variant());
}

namespace {

List<DimExpr> GetOutputDimValuesImpl(const List<ScheduleDim>& sched_dims) {
  List<DimExpr> ret{};
  for (const auto& sched_dim : *sched_dims) {
    const auto& loop_size = GetLoopSize(sched_dim);
    ret->emplace_back(loop_size);
  }
  return ret;
}

List<DimExpr> GetOutputDimValuesImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [_, shape] = sched_reshape.tuple();
  List<DimExpr> ret{};
  for (const auto& loop_size : *shape.value()) {
    ret->emplace_back(loop_size);
  }
  return ret;
}

List<DimExpr> GetOutputDimValuesImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, perm] = sched_transpose.tuple();
  const auto& input_dims = GetOutputDimValues(sched_mesh);
  List<DimExpr> ret{};
  for (const auto& idx : *perm.value()) {
    ret->emplace_back(input_dims->at(idx));
  }
  return ret;
}

List<DimExpr> GetOutputDimValuesImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [_, shape] = sched_padding.tuple();
  List<DimExpr> ret{};
  for (const auto& loop_size : *shape.value()) {
    ret->emplace_back(loop_size);
  }
  return ret;
}

}  // namespace

List<DimExpr> GetOutputDimValues(const ScheduleMesh& sched_mesh) {
  return std::visit(
      [&](const auto& impl) { return GetOutputDimValuesImpl(impl); },
      sched_mesh.variant());
}

namespace {

ScheduleMesh GetInputScheduleMeshImpl(const List<ScheduleDim>& sched_dims) {
  return sched_dims;
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshReshape<ScheduleMesh>& sched_reshape) {
  const auto& [sched_mesh, _] = sched_reshape.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshTranspose<ScheduleMesh>& sched_transpose) {
  const auto& [sched_mesh, _] = sched_transpose.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

ScheduleMesh GetInputScheduleMeshImpl(
    const ScheduleMeshPadding<ScheduleMesh>& sched_padding) {
  const auto& [sched_mesh, _] = sched_padding.tuple();
  return GetInputScheduleMesh(sched_mesh);
}

}  // namespace

ScheduleMesh GetInputScheduleMesh(const ScheduleMesh& sched_mesh) {
  return std::visit(
      [&](const auto& impl) { return GetInputScheduleMeshImpl(impl); },
      sched_mesh.variant());
}

namespace {

constexpr int kThreadSize = 1024;

class ScheduleMeshPolicy {
 public:
  ScheduleMeshPolicy(const ScheduleMeshPolicy&) = delete;
  ScheduleMeshPolicy(ScheduleMeshPolicy&&) = delete;
  virtual ~ScheduleMeshPolicy() = default;

  virtual bool Match(const ShardableDimAndPerm& dim_perm) const = 0;

  virtual std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const ShardableDimAndPerm& dim_perm) const = 0;

 protected:
  ScheduleMeshPolicy() = default;
};

List<int> ExpandToN(const List<int>& perm, int n) {
  CHECK_LE(perm->size(), n);
  List<int> ret{};
  ret->insert(ret->begin(), perm->begin(), perm->end());
  for (int i = 0; i < n; ++i) {
    if (std::find(perm->begin(), perm->end(), i) == perm->end()) {
      ret->push_back(i);
    } else {
      // Do nothing
    }
  }
  CHECK_EQ(ret->size(), n);
  return ret;
}

List<ScheduleDim> PermuteLoopSizes(const List<ScheduleDim>& loop_sizes,
                                   const List<int>& perm) {
  CHECK_GE(loop_sizes->size(), perm->size());
  List<ScheduleDim> ret{};
  for (std::size_t i = 0; i < perm->size(); ++i) {
    ret->push_back(loop_sizes->at(perm->at(i)));
  }
  for (std::size_t i = 0; i < loop_sizes->size(); ++i) {
    if (std::find(perm->begin(), perm->end(), i) == perm->end()) {
      ret->push_back(loop_sizes->at(i));
    } else {
      // Do nothing
    }
  }
  return ret;
}

class NaiveInjectiveScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  NaiveInjectiveScheduleMeshPolicy() = default;

  bool Match(const ShardableDimAndPerm& dim_perm) const override {
    const auto& [loop_sizes, _] = dim_perm;
    for (const auto& sched_dim : *loop_sizes) {
      if (!sched_dim.Has<tInjective<LoopSize>>()) {
        return false;
      }
    }
    return true;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const ShardableDimAndPerm& dim_perm) const override {
    VLOG(4) << "Match NaiveInjectiveScheduleMeshPolicy";
    auto [loop_sizes, perm] = dim_perm;
    ScheduleMesh sched_mesh{loop_sizes};
    if (!perm->empty()) {
      sched_mesh =
          MeshTranspose(sched_mesh, ExpandToN(perm, loop_sizes->size()));
      loop_sizes = PermuteLoopSizes(loop_sizes, perm);
    } else {
      // Do nothing
    }
    List<LoopType> loop_types{};
    loop_types->emplace_back(S0x{});
    for (std::size_t i = 1; i < loop_sizes->size(); ++i) {
      loop_types->emplace_back(Temporal{});
    }
    return std::make_tuple(sched_mesh, loop_types);
  }
};

class NaiveReduceScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  NaiveReduceScheduleMeshPolicy() = default;

  bool Match(const ShardableDimAndPerm& dim_perm) const override {
    const auto& [loop_sizes, _] = dim_perm;
    for (const auto& sched_dim : *loop_sizes) {
      if (!sched_dim.Has<tInjective<LoopSize>>()) {
        return true;
      }
    }
    return false;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const ShardableDimAndPerm& dim_perm) const override {
    VLOG(4) << "Match NaiveReduceScheduleMeshPolicy";
    auto [loop_sizes, perm] = dim_perm;
    ScheduleMesh sched_mesh{loop_sizes};
    if (!perm->empty()) {
      sched_mesh =
          MeshTranspose(sched_mesh, ExpandToN(perm, loop_sizes->size()));
      loop_sizes = PermuteLoopSizes(loop_sizes, perm);
    } else {
      // Do nothing
    }
    List<LoopType> loop_types{};
    loop_types->emplace_back(S0x{});
    for (std::size_t i = 1; i < loop_sizes->size(); ++i) {
      loop_types->emplace_back(Temporal{});
    }
    List<int> non_reduce_axes{};
    List<int> reduce_axes{};
    for (std::size_t i = 0; i < loop_sizes->size(); ++i) {
      const auto& loop_size = loop_sizes->at(i);
      if (loop_size.Has<tInjective<LoopSize>>()) {
        non_reduce_axes->emplace_back(i);
      } else {
        reduce_axes->emplace_back(i);
      }
    }
    List<int> reduce_perm{};
    reduce_perm->insert(
        reduce_perm->end(), non_reduce_axes->begin(), non_reduce_axes->end());
    reduce_perm->insert(
        reduce_perm->end(), reduce_axes->begin(), reduce_axes->end());
    sched_mesh = MeshTranspose(sched_mesh, reduce_perm);
    return std::make_tuple(sched_mesh, loop_types);
  }
};

const std::vector<std::unique_ptr<ScheduleMeshPolicy>>&
GetAllScheduleMeshPolicies() {
  static std::vector<std::unique_ptr<ScheduleMeshPolicy>> policies{};
  policies.emplace_back(std::make_unique<NaiveInjectiveScheduleMeshPolicy>());
  policies.emplace_back(std::make_unique<NaiveReduceScheduleMeshPolicy>());
  return policies;
}

}  // namespace

std::tuple<ScheduleMesh, List<LoopType>> CreateOptimizedScheduleMesh(
    const ShardableDimAndPerm& dim_perm) {
  for (const auto& policy : GetAllScheduleMeshPolicies()) {
    if (policy->Match(dim_perm)) {
      return policy->Optimize(dim_perm);
    }
  }
  LOG(FATAL) << "Dead code, no valid schedule mesh policy found";
}

std::tuple<List<ScheduleMesh>, List<List<LoopType>>>
CreateOptimizedScheduleMeshs(const List<ShardableDimAndPerm> dim_perms) {
  List<ScheduleMesh> ret_mesh{};
  List<List<LoopType>> ret_looptypes{};
  for (const auto& dim_perm : *dim_perms) {
    const auto& [mesh, loop_types] = CreateOptimizedScheduleMesh(dim_perm);
    VLOG(1) << "Finish CreateOptimizedScheduleMesh";
    ret_mesh->emplace_back(mesh);
    ret_looptypes->emplace_back(loop_types);
  }
  return std::make_tuple(ret_mesh, ret_looptypes);
}

ScheduleMesh MeshReshape(const ScheduleMesh& sched_mesh,
                         const std::vector<std::int64_t>& shape) {
  const auto& origin_shape = GetOutputDimValues(sched_mesh);
  std::int64_t origin_numel = 1;
  for (const auto& dim : *origin_shape) {
    CHECK(dim.Has<std::int64_t>());
    origin_numel *= dim.Get<std::int64_t>();
  }

  std::int64_t numel = 1;
  bool dynamic_shape = false;
  for (const auto& dim : shape) {
    if (dim < 0) {
      CHECK(dim == -1 && !dynamic_shape);
      dynamic_shape = true;
    } else {
      numel *= dim;
    }
  }

  CHECK(dynamic_shape || numel == origin_numel);
  List<LoopSize> reshape_to{};
  for (const auto& dim : shape) {
    if (dim < 0) {
      CHECK_EQ(origin_numel % numel, 0);
      reshape_to->emplace_back(origin_numel / numel);
    } else {
      reshape_to->emplace_back(dim);
    }
  }
  return ScheduleMeshReshape<ScheduleMesh>(sched_mesh, reshape_to);
}

ScheduleMesh MeshTranspose(const ScheduleMesh& sched_mesh,
                           const List<int>& perm) {
  return ScheduleMeshTranspose<ScheduleMesh>{sched_mesh, perm};
}

ScheduleMesh MeshPadding(const ScheduleMesh& sched_mesh,
                         const List<LoopSize>& padding_to) {
  const auto& ret = ScheduleMeshPadding<ScheduleMesh>(sched_mesh, padding_to);
  const auto& input_dims = GetOutputDimValues(sched_mesh);
  const auto& output_dims = GetOutputDimValues(ret);
  CHECK_EQ(input_dims->size(), output_dims->size());
  for (std::size_t i = 0; i < input_dims->size(); ++i) {
    if (input_dims->at(i).Has<std::int64_t>() &&
        output_dims->at(i).Has<std::int64_t>()) {
      CHECK_LE(input_dims->at(i).Get<std::int64_t>(),
               output_dims->at(i).Get<std::int64_t>());
    }
  }
  return ret;
}

ScheduleMesh MeshPaddingRoundUp(
    const ScheduleMesh& sched_mesh,
    const std::vector<std::optional<std::int64_t>>& align_sizes) {
  const auto& shape = GetOutputDimValues(sched_mesh);
  CHECK_EQ(shape->size(), align_sizes.size());
  List<LoopSize> padding_to{};
  bool create_new_sched_mesh = false;
  for (std::size_t i = 0; i < shape->size(); ++i) {
    if (!align_sizes.at(i).has_value()) {
      continue;
    }
    std::int64_t align_size = align_sizes.at(i).value();
    CHECK(shape->at(i).Has<std::int64_t>());
    std::int64_t dim = shape->at(i).Get<std::int64_t>();
    std::int64_t padding_size =
        (dim + align_size - 1) / align_size * align_size;

    if (padding_size != dim) {
      create_new_sched_mesh = true;
    }
    padding_to->emplace_back(padding_size);
  }
  if (!create_new_sched_mesh) {
    return sched_mesh;
  }
  return ScheduleMeshPadding<ScheduleMesh>(sched_mesh, padding_to);
}

}  // namespace cinn::adt
