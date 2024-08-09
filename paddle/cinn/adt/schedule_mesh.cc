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
#include "paddle/common/enforce.h"
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
  PADDLE_ENFORCE_EQ(GetOutputRank(sched_mesh) == perm.value()->size(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The size of perm should be equal to the output rank, "
                        "but got perm size = %d, output rank = %d.",
                        perm.value()->size(),
                        GetOutputRank(sched_mesh)));
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

  virtual bool Match(const List<ScheduleDim>& loop_sizes) const = 0;

  virtual std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const List<ScheduleDim>& loop_sizes) const = 0;

 protected:
  ScheduleMeshPolicy() = default;
};

class NaiveInjectiveScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  NaiveInjectiveScheduleMeshPolicy() = default;

  bool Match(const List<ScheduleDim>& loop_sizes) const override {
    for (const auto& sched_dim : *loop_sizes) {
      if (!sched_dim.Has<tInjective<LoopSize>>()) {
        return false;
      }
    }
    return true;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const List<ScheduleDim>& loop_sizes) const override {
    VLOG(4) << "Match NaiveInjectiveScheduleMeshPolicy";
    ScheduleMesh sched_mesh{loop_sizes};
    List<LoopType> loop_types{};
    for (const auto& _ : *loop_sizes) {
      loop_types->emplace_back(Temporal{});
    }
    return std::make_tuple(sched_mesh, loop_types);
  }
};

class NaiveReduceScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  NaiveReduceScheduleMeshPolicy() = default;

  bool Match(const List<ScheduleDim>& loop_sizes) const override {
    for (const auto& sched_dim : *loop_sizes) {
      if (!sched_dim.Has<tInjective<LoopSize>>()) {
        return true;
      }
    }
    return false;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const List<ScheduleDim>& loop_sizes) const override {
    VLOG(4) << "Match NaiveReduceScheduleMeshPolicy";
    ScheduleMesh sched_mesh{loop_sizes};
    List<LoopType> loop_types{};
    List<int> non_reduce_axes{};
    List<int> reduce_axes{};
    for (std::int64_t i = 0; i < loop_sizes->size(); ++i) {
      loop_types->emplace_back(Temporal{});
      const auto& loop_size = loop_sizes->at(i);
      if (loop_size.Has<tInjective<LoopSize>>()) {
        non_reduce_axes->emplace_back(i);
      } else {
        reduce_axes->emplace_back(i);
      }
    }
    List<int> perm{};
    perm->insert(perm->end(), non_reduce_axes->begin(), non_reduce_axes->end());
    perm->insert(perm->end(), reduce_axes->begin(), reduce_axes->end());
    sched_mesh = MeshTranspose(sched_mesh, perm);
    return std::make_tuple(sched_mesh, loop_types);
  }
};

class AllInjectiveScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  AllInjectiveScheduleMeshPolicy() = default;

  bool Match(const List<ScheduleDim>& loop_sizes) const override {
    for (const auto& sched_dim : *loop_sizes) {
      if (!sched_dim.Has<tInjective<LoopSize>>()) {
        return false;
      }
      if (!GetLoopSize(sched_dim).Has<std::int64_t>()) {
        return false;
      }
    }
    return true;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const List<ScheduleDim>& loop_sizes) const override {
    ScheduleMesh sched_mesh{loop_sizes};
    std::int64_t acc = 1;
    for (const auto& sched_dim : *loop_sizes) {
      acc *= GetLoopSize(sched_dim).Get<std::int64_t>();
    }
    sched_mesh = MeshReshape(sched_mesh, {acc});
    sched_mesh = MeshPaddingRoundUp(sched_mesh, {kThreadSize});
    sched_mesh = MeshReshape(sched_mesh, {-1, kThreadSize});

    return std::make_tuple(sched_mesh, List<LoopType>{S0x{}, S1x{}});
  }
};

List<int> ConcatIntLists(const List<int>& lhs, const List<int>& rhs) {
  List<int> ret{};
  for (int i : *lhs) {
    ret->emplace_back(i);
  }
  for (int i : *rhs) {
    ret->emplace_back(i);
  }
  return ret;
}

std::vector<std::int64_t> ConcatIntLists(const std::vector<std::int64_t>& lhs,
                                         const std::vector<std::int64_t>& rhs) {
  std::vector<std::int64_t> ret{};
  for (int i : lhs) {
    ret.emplace_back(i);
  }
  for (int i : rhs) {
    ret.emplace_back(i);
  }
  return ret;
}

std::vector<std::optional<std::int64_t>> ConcatIntListsToOptionalList(
    const std::vector<std::int64_t>& lhs,
    const std::vector<std::int64_t>& rhs) {
  std::vector<std::optional<std::int64_t>> ret{};
  for (int i : lhs) {
    ret.emplace_back(i);
  }
  for (int i : rhs) {
    ret.emplace_back(i);
  }
  return ret;
}

class GeneralScheduleMeshPolicy final : public ScheduleMeshPolicy {
 public:
  GeneralScheduleMeshPolicy() = default;

  bool Match(const List<ScheduleDim>& loop_sizes) const override {
    for (const auto& sched_dim : *loop_sizes) {
      if (!GetLoopSize(sched_dim).Has<std::int64_t>()) {
        return false;
      }
    }
    return true;
  }

  std::tuple<ScheduleMesh, List<LoopType>> Optimize(
      const List<ScheduleDim>& loop_sizes) const override {
    const auto& injective_axes = GetInjectiveAxis(loop_sizes);
    const auto& reduce_axes = GetReduceAxis(loop_sizes);

    std::vector<std::int64_t> reduce_shape{};
    for (int reduce_axis : *reduce_axes) {
      reduce_shape.emplace_back(
          GetLoopSize(loop_sizes->at(reduce_axis)).Get<std::int64_t>());
    }

    ScheduleMesh sched_mesh{loop_sizes};
    sched_mesh =
        MeshTranspose(sched_mesh, ConcatIntLists(injective_axes, reduce_axes));
    sched_mesh = MeshReshape(sched_mesh, ConcatIntLists({-1}, reduce_shape));
    sched_mesh = MeshPaddingRoundUp(
        sched_mesh, ConcatIntListsToOptionalList({kThreadSize}, reduce_shape));
    sched_mesh = MeshReshape(sched_mesh,
                             ConcatIntLists({-1, kThreadSize}, reduce_shape));

    List<LoopType> loop_types{S0x{}, S1x{}};
    for (std::size_t i = 0; i < reduce_axes->size(); ++i) {
      loop_types->emplace_back(Temporal{});
    }
    return std::make_tuple(sched_mesh, loop_types);
  }
};

const std::vector<std::unique_ptr<ScheduleMeshPolicy>>&
GetAllScheduleMeshPolicies() {
  static std::vector<std::unique_ptr<ScheduleMeshPolicy>> policies{};
  policies.emplace_back(std::make_unique<NaiveInjectiveScheduleMeshPolicy>());
  policies.emplace_back(std::make_unique<NaiveReduceScheduleMeshPolicy>());
  policies.emplace_back(std::make_unique<AllInjectiveScheduleMeshPolicy>());
  policies.emplace_back(std::make_unique<GeneralScheduleMeshPolicy>());
  return policies;
}

}  // namespace

std::tuple<ScheduleMesh, List<LoopType>> CreateOptimizedScheduleMesh(
    const List<ScheduleDim>& loop_sizes) {
  for (const auto& policy : GetAllScheduleMeshPolicies()) {
    if (policy->Match(loop_sizes)) {
      return policy->Optimize(loop_sizes);
    }
  }
  PADDLE_THROW(::common::errors::Fatal(
      "Dead code, no valid schedule mesh policy found"));
}

ScheduleMesh MeshReshape(const ScheduleMesh& sched_mesh,
                         const std::vector<std::int64_t>& shape) {
  const auto& origin_shape = GetOutputDimValues(sched_mesh);
  std::int64_t origin_numel = 1;
  for (const auto& dim : *origin_shape) {
    PADDLE_ENFORCE_EQ(
        dim.Has<std::int64_t>(),
        true,
        ::common::errors::InvalidArgument(
            "Each dimension in 'origin_shape' must have an int64_t value."));
    origin_numel *= dim.Get<std::int64_t>();
  }

  std::int64_t numel = 1;
  bool dynamic_shape = false;
  for (const auto& dim : shape) {
    if (dim < 0) {
      PADDLE_ENFORCE_EQ(dim == -1 && !dynamic_shape,
                        true,
                        ::common::errors::InvalidArgument(
                            "Negative dimension in 'shape' must be "
                            "-1 to represent dynamic shape. "
                            "But received: %d",
                            dim));
      dynamic_shape = true;
    } else {
      numel *= dim;
    }
  }
  PADDLE_ENFORCE_EQ(dynamic_shape || numel == origin_numel,
                    true,
                    ::common::errors::InvalidArgument(
                        "The total number of elements must match between "
                        "'shape' and 'origin_shape' "
                        "unless there is a dynamic shape. "
                        "But received: numel = %d, origin_numel = %d",
                        numel,
                        origin_numel));
  List<LoopSize> reshape_to{};
  for (const auto& dim : shape) {
    if (dim < 0) {
      PADDLE_ENFORCE_EQ(origin_numel % numel == 0UL,
                        true,
                        ::common::errors::InvalidArgument(
                            "The origin_numel should be divisible by numel"));
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
  PADDLE_ENFORCE_EQ(
      input_dims->size(),
      output_dims->size(),
      ::common::errors::InvalidArgument(
          "The size of input_dims and output_dims should be equal, "
          "but got input_dims size = %d, output_dims size = %d.",
          input_dims->size(),
          output_dims->size()));
  for (std::size_t i = 0; i < input_dims->size(); ++i) {
    if (input_dims->at(i).Has<std::int64_t>() &&
        output_dims->at(i).Has<std::int64_t>()) {
      PADDLE_ENFORCE_LE(input_dims->at(i).Get<std::int64_t>(),
                        output_dims->at(i).Get<std::int64_t>(),
                        ::common::errors::InvalidArgument(
                            "The input_dims should be equal to output_dims, "
                            "but got input_dims not equal to output_dims = "));
    }
  }
  return ret;
}

ScheduleMesh MeshPaddingRoundUp(
    const ScheduleMesh& sched_mesh,
    const std::vector<std::optional<std::int64_t>>& align_sizes) {
  const auto& shape = GetOutputDimValues(sched_mesh);
  PADDLE_ENFORCE_EQ(shape->size(),
                    align_sizes.size(),
                    ::common::errors::InvalidArgument(
                        "The size of shape and align_sizes should be equal, "
                        "but got shape size = %d, align_sizes size = %d.",
                        shape->size(),
                        align_sizes.size()));
  List<LoopSize> padding_to{};
  bool create_new_sched_mesh = false;
  for (std::size_t i = 0; i < shape->size(); ++i) {
    if (!align_sizes.at(i).has_value()) {
      continue;
    }
    std::int64_t align_size = align_sizes.at(i).value();
    PADDLE_ENFORCE_EQ(
        shape->at(i).Has<std::int64_t>(),
        true,
        ::common::errors::InvalidArgument(
            "Each dimension in 'shape' must have an int64_t value. "
            "But the dimension at index %d does not.",
            i));
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
