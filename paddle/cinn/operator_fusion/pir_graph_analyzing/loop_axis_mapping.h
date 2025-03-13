// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <variant>
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

#define DECLARE_TRANSFORM_PTR(T) \
  struct T;                      \
  using T##Ptr = std::shared_ptr<T>;

DECLARE_TRANSFORM_PTR(UnsupportedTransform);
DECLARE_TRANSFORM_PTR(IdentityTransform);
DECLARE_TRANSFORM_PTR(TransposeTransform);
DECLARE_TRANSFORM_PTR(AppendAxisTransform);
DECLARE_TRANSFORM_PTR(DeleteAxisTransform);
DECLARE_TRANSFORM_PTR(ReshapeTransform);
#undef DECLARE_TRANSFORM_PTR

using AxisTransform = std::variant<UnsupportedTransformPtr,
                                   IdentityTransformPtr,
                                   TransposeTransformPtr,
                                   AppendAxisTransformPtr,
                                   DeleteAxisTransformPtr,
                                   ReshapeTransformPtr>;
using AxisTransformRoute = std::vector<AxisTransform>;

struct UnsupportedTransform {
 public:
  static UnsupportedTransformPtr InstancePtr() {
    static UnsupportedTransformPtr instance(new UnsupportedTransform());
    return instance;
  }
  std::string DebugStr() const { return "Unsupported"; }
  AxisTransform reverse() { return InstancePtr(); }

 private:
  UnsupportedTransform() = default;
};

struct IdentityTransform {
 public:
  static IdentityTransformPtr InstancePtr() {
    static IdentityTransformPtr instance(new IdentityTransform());
    return instance;
  }
  std::string DebugStr() const { return "Identity"; }
  AxisTransform reverse() { return InstancePtr(); }

 private:
  IdentityTransform() = default;
};

struct TransposeTransform {
  explicit TransposeTransform(const std::vector<int32_t>& perm) : perm(perm) {}
  std::vector<int32_t> perm;
  std::string DebugStr() const {
    return "Transpose{perm=(" + cinn::utils::Join(perm, ",") + ")}";
  }
  AxisTransform reverse() {
    return std::make_shared<TransposeTransform>(GetReversePerm(perm));
  }
};

struct AppendAxisTransform {
  AppendAxisTransform(const std::vector<int64_t>& axis,
                      const std::vector<symbol::DimExpr>& shape)
      : axis(axis), shape(shape) {
    PADDLE_ENFORCE_EQ(axis.size(),
                      shape.size(),
                      ::common::errors::InvalidArgument(
                          "Axis size and shape size must be equal."));
  }
  explicit AppendAxisTransform(const std::vector<int64_t>& axis) : axis(axis) {
    shape = std::vector<symbol::DimExpr>(axis.size(), symbol::DimExpr{1});
  }
  std::vector<int64_t> axis;
  std::vector<symbol::DimExpr> shape;
  std::string DebugStr() const {
    return "AppendAxis{axis=(" + cinn::utils::Join(axis, ",") + "), shape=(" +
           cinn::utils::Join(shape, ",") + ")}";
  }
  AxisTransform reverse();
};

struct DeleteAxisTransform {
  explicit DeleteAxisTransform(const std::vector<int64_t>& axis,
                               const std::vector<symbol::DimExpr>& shape)
      : axis(axis), shape(shape) {
    PADDLE_ENFORCE_EQ(axis.size(),
                      shape.size(),
                      ::common::errors::InvalidArgument(
                          "Axis size and shape size must be equal."));
  }
  std::vector<int64_t> axis;
  std::vector<symbol::DimExpr> shape;
  std::string DebugStr() const {
    return "DeleteAxis{axis=(" + cinn::utils::Join(axis, ",") + "), shape=(" +
           cinn::utils::Join(shape, ",") + ")}";
  }
  AxisTransform reverse();
};

struct ReshapeTransform {
  explicit ReshapeTransform(const std::vector<symbol::DimExpr>& in_shape,
                            const std::vector<symbol::DimExpr>& out_shape)
      : in_shape(in_shape), out_shape(out_shape) {}
  std::vector<symbol::DimExpr> in_shape;
  std::vector<symbol::DimExpr> out_shape;
  std::string DebugStr() const {
    return "Reshape{in_shape=(" + cinn::utils::Join(in_shape, ",") +
           "), out_shape=(" + cinn::utils::Join(out_shape, ",") + ")}";
  }
  AxisTransform reverse() {
    return std::make_shared<ReshapeTransform>(out_shape, in_shape);
  }
};

std::ostream& operator<<(std::ostream& os, const AxisTransform& transform);
std::ostream& operator<<(std::ostream& os, const AxisTransformRoute& route);

AxisTransform ReverseTransform(const AxisTransform& transform);
AxisTransformRoute ReverseTransformRoute(const AxisTransformRoute& route);

struct LoopAxisMapping {
  std::vector<pir::Value> input_values;
  std::vector<pir::Value> output_values;
  std::unordered_map<pir::Value, int> outputs_use_count;
  std::vector<symbol::DimExpr> loop;
  size_t reduce_axis_num = 0;

  std::vector<AxisTransformRoute> input2loop;
  std::vector<AxisTransformRoute> loop2output;
  std::vector<AxisTransformRoute> loop2input;
  std::vector<AxisTransformRoute> output2loop;

  void SetReverseMapping();
  void DisableLoopAxisMapping();
  void SimplifyForwardMapping();

  std::string DebugStr() const;
};

LoopAxisMapping CreateLoopAxisMapping(pir::Operation* op);

class AxisTransformSimulator {
 public:
  AxisTransformSimulator() = delete;
  AxisTransformSimulator(const AxisTransformRoute& route,
                         const std::vector<symbol::DimExpr>& inshape);

  std::set<std::string> GetRelatedAxisIds(const std::vector<std::string>& ids);

  const AxisTransformRoute& route_;
  std::vector<symbol::DimExpr> out_shape_;
  std::vector<std::string> source_ids_;
  std::vector<std::string> target_ids_;
  std::unordered_map<std::string, symbol::DimExpr> axis_symbols_;
  std::map<std::string, std::set<std::string>> axis_relation_map_;

 private:
  void Simulate();

  int id_counter_ = 0;
  std::string UniqueAxisId() { return "I" + std::to_string(id_counter_++); }
};

AxisTransformRoute SimplifyTransformRoute(
    const AxisTransformRoute& route,
    const std::vector<symbol::DimExpr>& input_shape);

}  // namespace cinn::fusion
