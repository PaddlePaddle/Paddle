// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect::slice_uitls {
inline void CheckAndUpdateSliceAttrs(
    const ExprVec &in_dims,
    const std::vector<int64_t> &axes,
    ExprVec *starts_p,
    ExprVec *ends_p,
    std::vector<int64_t> *infer_flags = nullptr) {
  ExprVec &starts = *starts_p;
  ExprVec &ends = *ends_p;
  auto IsMaxInt = [](const symbol::DimExpr &expr) {
    return expr.isa<int64_t>() &&
           expr.Get<int64_t>() ==
               static_cast<int64_t>(std::numeric_limits<int>::max());
  };

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t start_i = 0;
    if (starts[i].isa<int64_t>()) {
      start_i = starts[i].Get<int64_t>();
    }
    int64_t end_i = 0;
    if (ends[i].isa<int64_t>()) {
      end_i = ends[i].Get<int64_t>();
    }

    // For both start and end can be negtive or positive, we need to handle the
    // following different arrangements.
    ends[i] = IsMaxInt(ends[i]) ? in_dims[axis] : ends[i];

    bool both_negative_or_positive =
        (start_i >= 0 && end_i >= 0) || (start_i <= 0 && end_i <= 0);
    bool start_negative_end_positive = start_i <= 0 && end_i >= 0;
    bool start_positive_end_negative = start_i >= 0 && end_i <= 0;

    if (both_negative_or_positive) {
      continue;
    } else if (start_negative_end_positive) {
      starts[i] = starts[i] + in_dims[axis];
    } else if (start_positive_end_negative) {
      starts[i] = starts[i] - in_dims[axis];
    } else {
      LOG(FATAL) << "Dead code";
    }
  }
}

inline ExprVec GetSliceDims(const ExprVec &in_dims,
                            const std::vector<int64_t> &axes,
                            const ExprVec &starts,
                            const ExprVec &ends,
                            std::vector<int64_t> *infer_flags = nullptr) {
  ExprVec slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    slice_dims[axis] = ends[i] - starts[i];
  }

  return slice_dims;
}

inline ExprVec GetDecreasedDims(const ExprVec &slice_dims,
                                const std::vector<int64_t> &decrease_axes) {
  ExprVec decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      int64_t axis = decrease_axes[i];
      decrease_flag[axis] = 1;
    }
    ExprVec new_shape;
    for (size_t i = 0; i < slice_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.emplace_back(slice_dims[i]);
      }
    }
    decreased_dims = new_shape;
  }
  return decreased_dims;
}

inline std::vector<int64_t> FormatSliceAxes(
    const std::vector<int64_t> &axes_raw, int64_t rank) {
  std::vector<int64_t> axes_vec(axes_raw.size(), 0);
  std::transform(
      axes_raw.begin(), axes_raw.end(), axes_vec.begin(), [rank](int64_t axis) {
        return axis >= 0 ? axis : std::max(int64_t(0), axis + rank);
      });
  return axes_vec;
}

inline ShapeOrData SliceRawInferSymbolicShape(
    const ShapeOrData &in_shapeordata,
    const ExprVec &starts_expr,
    const ExprVec &ends_expr,
    const std::vector<int64_t> &axes_raw,
    const std::vector<int64_t> &infer_flags_raw,
    const std::vector<int64_t> &decrease_axis) {
  ExprVec starts = starts_expr;
  ExprVec ends = ends_expr;
  std::vector<int64_t> infer_flags = [&infer_flags_raw, &axes_raw] {
    return infer_flags_raw.empty() ? std::vector<int64_t>(axes_raw.size(), 1)
                                   : infer_flags_raw;
  }();

  const auto &GetShapeDimExprs = [&]() -> symbol::ShapeOrDataDimExprs {
    const ExprVec &in_dims = in_shapeordata.shape();
    std::vector<int64_t> axes = FormatSliceAxes(axes_raw, in_dims.size());
    CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &infer_flags);
    ExprVec slice_dims =
        GetSliceDims(in_dims, axes, starts, ends, &infer_flags);
    ExprVec out_dims = GetDecreasedDims(slice_dims, decrease_axis);

    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  };

  // When `pd.slice` is operating on a tensor which is produced by a `pd.shape`
  // op, the reseult should be written into data.
  const auto &GetDataDimExprs = [&]() -> symbol::ShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> out_data;

    // Currently, we DO NOT support the case that any element in `axes` `starts`
    // or `ends` is a Symbol.
    auto vec_int64 = details::VecExpr2Int64(starts);
    IR_ENFORCE(vec_int64.has_value(),
               "for slice op, all the elements in `starts` must be int64_t");
    std::vector<int64_t> starts_int = vec_int64.value();

    vec_int64 = details::VecExpr2Int64(ends);
    IR_ENFORCE(vec_int64.has_value(),
               "for slice op, all the elements in `ends` must be int64_t");
    std::vector<int64_t> ends_int = vec_int64.value();

    const int64_t start =
        starts_int[0] < 0 ? starts_int[0] + in_shapeordata.data().value().size()
                          : starts_int[0];
    const int64_t end =
        static_cast<int64_t>(std::numeric_limits<int>::max()) == ends_int[0]
            ? in_shapeordata.data().value().size()
            : ends_int[0];

    for (int64_t i = start; i < end; i++) {
      out_data.push_back(in_shapeordata.data().value()[i]);
    }

    const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(shape, out_data)};
  };

  return in_shapeordata.data().has_value() ? GetDataDimExprs()
                                           : GetShapeDimExprs();
}
}  // namespace paddle::dialect::slice_uitls
