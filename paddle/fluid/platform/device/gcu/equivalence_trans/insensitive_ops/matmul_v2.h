/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kMatMulV2 = "matmul_v2";
const char *const kMatMulV2Grad = "matmul_v2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MatMulV2EquivalenceTrans) {
  auto *op = node->Op();
  GcuOp X = *(map_inputs["X"].at(0));
  GcuOp Y = *(map_inputs["Y"].at(0));
  auto x_shape = X.GetType().GetShape();
  auto y_shape = Y.GetType().GetShape();
  auto trans_x = PADDLE_GET_CONST(bool, op->GetAttr("trans_x"));
  auto trans_y = PADDLE_GET_CONST(bool, op->GetAttr("trans_y"));
  int64_t x_rank = x_shape.size();
  int64_t y_rank = y_shape.size();
  int64_t max_rank = std::max(x_rank, y_rank);
  int64_t rank_diff = std::abs(x_rank - y_rank);
  auto ptype = X.GetType().GetPrimitiveType();
  int64_t batch_dim;

  if (x_rank > y_rank) {
    if (trans_x || y_rank == 1) {
      std::vector<int64_t> broadcast_dims;
      std::vector<int64_t> bc_shape;
      if (y_rank == 1) {
        for (int64_t i = 0; i < rank_diff - 1; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        bc_shape.emplace_back(y_shape[0]);
        bc_shape.emplace_back(1);
        broadcast_dims.emplace_back(rank_diff - 1);
      } else {
        for (int64_t i = 0; i < rank_diff; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        for (int64_t i = 0; i < y_rank; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        int iter = 0;
        for (int64_t i = 0; i < x_rank; ++i) {
          if (i < rank_diff) {
            ++iter;
          } else {
            broadcast_dims.emplace_back(i);
          }
        }
      }
      builder::Type type(bc_shape, ptype);
      Y = builder::BroadcastInDim(Y, broadcast_dims, type);
    }
    if (y_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else if (x_rank < y_rank) {
    std::vector<int64_t> broadcast_dims;
    std::vector<int64_t> bc_shape;
    if (x_rank == 1) {
      for (int64_t i = 0; i < rank_diff - 1; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      bc_shape.emplace_back(1);
      bc_shape.emplace_back(x_shape[0]);
      broadcast_dims.emplace_back(rank_diff);
    } else {
      for (int64_t i = 0; i < rank_diff; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      for (int64_t i = 0; i < x_rank; i++) {
        bc_shape.emplace_back(x_shape[i]);
      }
      int iter = 0;
      for (int64_t i = 0; i < y_rank; ++i) {
        if (i < rank_diff) {
          ++iter;
        } else {
          broadcast_dims.emplace_back(i);
        }
      }
    }
    builder::Type type(bc_shape, ptype);
    X = builder::BroadcastInDim(X, broadcast_dims, type);
    if (x_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else {
    batch_dim = max_rank - 2;
    if (x_rank == y_rank && x_rank > 3) {
      auto x_brd_shape = x_shape;
      auto y_brd_shape = y_shape;
      std::vector<int64_t> x_brd_dims, y_brd_dims;
      for (int64_t i = 0; i < x_rank - 2; ++i) {
        x_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
        y_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
      }
      x_brd_dims.resize(x_rank);
      y_brd_dims.resize(y_rank);
      std::iota(x_brd_dims.begin(), x_brd_dims.end(), 0);
      std::iota(y_brd_dims.begin(), y_brd_dims.end(), 0);
      if (x_brd_shape != x_shape) {
        X = builder::BroadcastInDim(
            X, x_brd_dims, builder::Type(x_brd_shape, ptype));
      }
      if (y_brd_shape != y_shape) {
        Y = builder::BroadcastInDim(
            Y, y_brd_dims, builder::Type(y_brd_shape, ptype));
      }
    }
  }

  builder::DotDimensionNumbers dims_attr;
  std::vector<int64_t> lhs_batching_dimensions = {};
  std::vector<int64_t> rhs_batching_dimensions = {};
  std::vector<int64_t> lhs_contracting_dimensions = {};
  std::vector<int64_t> rhs_contracting_dimensions = {};
  if (x_rank == 1 && y_rank == 1) {
    lhs_contracting_dimensions.emplace_back(0);
    rhs_contracting_dimensions.emplace_back(0);
  } else if (x_rank <= y_rank || trans_x || y_rank == 1) {
    for (int64_t i = 0; i < max_rank - 1; ++i) {
      if (i < batch_dim) {
        lhs_batching_dimensions.emplace_back(i);
        rhs_batching_dimensions.emplace_back(i);
      } else {
        if (trans_x && x_rank != 1) {
          lhs_contracting_dimensions.emplace_back(i);
        } else {
          lhs_contracting_dimensions.emplace_back(i + 1);
        }
        if (trans_y && y_rank != 1) {
          rhs_contracting_dimensions.emplace_back(i + 1);
        } else {
          rhs_contracting_dimensions.emplace_back(i);
        }
      }
    }
  } else {
    lhs_contracting_dimensions.emplace_back(x_rank - 1);
    if (y_rank != 1) {
      if (trans_y) {
        rhs_contracting_dimensions.emplace_back(y_rank - 1);
      } else {
        rhs_contracting_dimensions.emplace_back(y_rank - 2);
      }
    } else {
      rhs_contracting_dimensions.emplace_back(0);
    }
  }

  std::vector<const char *> precision_config = {};
  dims_attr.set_lhs_batching_dimensions(lhs_batching_dimensions);
  dims_attr.set_rhs_batching_dimensions(rhs_batching_dimensions);
  dims_attr.set_lhs_contracting_dimensions(lhs_contracting_dimensions);
  dims_attr.set_rhs_contracting_dimensions(rhs_contracting_dimensions);
  auto dot = builder::DotGeneral(X, Y, dims_attr, precision_config);
  dot.SetAttribute("op_type", builder::Attribute("DotInference"));
  if (x_rank == 1 && y_rank == 1) {
    // auto type = dot.GetType().GetPrimitiveType();
    // std::vector<int64_t> new_shape;
    // new_shape.push_back(1);
    // builder::Type output_type(new_shape, type);
    // dot = builder::Reshape(dot, output_type);
  } else if (y_rank == 1) {
    auto shape = dot.GetType().GetShape();
    auto type = dot.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size() - 1; i++) {
      new_shape.push_back(shape[i]);
    }
    builder::Type output_type(new_shape, type);
    dot = builder::Reshape(dot, output_type);
  } else if (x_rank == 1) {
    auto shape = dot.GetType().GetShape();
    auto type = dot.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size(); i++) {
      if (i != shape.size() - 2) {
        new_shape.push_back(shape[i]);
      }
    }
    builder::Type output_type(new_shape, type);
    dot = builder::Reshape(dot, output_type);
  }

  auto result = std::make_shared<GcuOp>(dot);

  return result;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MatMulV2GradEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp X = *(map_inputs["X"].at(0));
  GcuOp Y = *(map_inputs["Y"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto x_shape = X.GetType().GetShape();
  auto y_shape = Y.GetType().GetShape();
  auto out_shape = out_grad.GetType().GetShape();
  int64_t x_rank = x_shape.size();
  int64_t y_rank = y_shape.size();
  int64_t out_rank = out_shape.size();
  int64_t max_rank = std::max(x_rank, y_rank);
  int64_t rank_diff = std::abs(x_rank - y_rank);
  auto ptype = X.GetType().GetPrimitiveType();
  auto trans_x = PADDLE_GET_CONST(bool, op->GetAttr("trans_x"));
  auto trans_y = PADDLE_GET_CONST(bool, op->GetAttr("trans_y"));
  int64_t batch_dim;
  GcuOp DX;
  GcuOp DY;
  // broadcast X, Y
  if (x_rank > y_rank) {
    if (trans_x || y_rank == 1) {
      std::vector<int64_t> broadcast_dims;
      std::vector<int64_t> bc_shape;
      if (y_rank == 1) {
        for (int64_t i = 0; i < rank_diff - 1; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        bc_shape.emplace_back(y_shape[0]);
        bc_shape.emplace_back(1);
        broadcast_dims.emplace_back(rank_diff - 1);
      } else {
        for (int64_t i = 0; i < rank_diff; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        for (int64_t i = 0; i < y_rank; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        int iter = 0;
        for (int64_t i = 0; i < x_rank; ++i) {
          if (i < rank_diff) {
            ++iter;
          } else {
            broadcast_dims.emplace_back(i);
          }
        }
      }
      builder::Type type(bc_shape, ptype);
      Y = builder::BroadcastInDim(Y, broadcast_dims, type);
    }
    if (y_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else if (x_rank < y_rank) {
    std::vector<int64_t> broadcast_dims;
    std::vector<int64_t> bc_shape;
    if (x_rank == 1) {
      for (int64_t i = 0; i < rank_diff - 1; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      bc_shape.emplace_back(1);
      bc_shape.emplace_back(x_shape[0]);
      broadcast_dims.emplace_back(rank_diff);
    } else {
      for (int64_t i = 0; i < rank_diff; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      for (int64_t i = 0; i < x_rank; i++) {
        bc_shape.emplace_back(x_shape[i]);
      }
      int iter = 0;
      for (int64_t i = 0; i < y_rank; ++i) {
        if (i < rank_diff) {
          ++iter;
        } else {
          broadcast_dims.emplace_back(i);
        }
      }
    }
    builder::Type type(bc_shape, ptype);
    X = builder::BroadcastInDim(X, broadcast_dims, type);
    if (x_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else {
    batch_dim = max_rank - 2;
    if (x_rank == y_rank && x_rank > 3) {
      auto x_brd_shape = x_shape;
      auto y_brd_shape = y_shape;
      std::vector<int64_t> x_brd_dims, y_brd_dims;
      for (int64_t i = 0; i < x_rank - 2; ++i) {
        x_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
        y_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
      }
      x_brd_dims.resize(x_rank);
      y_brd_dims.resize(y_rank);
      std::iota(x_brd_dims.begin(), x_brd_dims.end(), 0);
      std::iota(y_brd_dims.begin(), y_brd_dims.end(), 0);
      if (x_brd_shape != x_shape) {
        X = builder::BroadcastInDim(
            X, x_brd_dims, builder::Type(x_brd_shape, ptype));
      }
      if (y_brd_shape != y_shape) {
        Y = builder::BroadcastInDim(
            Y, y_brd_dims, builder::Type(y_brd_shape, ptype));
      }
    }
  }

  // reshape out@grad
  if (y_rank == 1 && x_rank == 1) {
    auto shape = out_grad.GetType().GetShape();
    auto type = out_grad.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    new_shape.emplace_back(1);
    // new_shape.emplace_back(1);
    builder::Type output_type(new_shape, type);
    out_grad = builder::Reshape(out_grad, output_type);

  } else if (y_rank == 1) {
    auto shape = out_grad.GetType().GetShape();
    auto type = out_grad.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size(); i++) {
      new_shape.emplace_back(shape[i]);
    }
    new_shape.emplace_back(1);
    builder::Type output_type(new_shape, type);
    out_grad = builder::Reshape(out_grad, output_type);
  } else if (x_rank == 1) {
    auto shape = out_grad.GetType().GetShape();
    auto type = out_grad.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size() - 1; i++) {
      new_shape.emplace_back(shape[i]);
    }
    new_shape.emplace_back(1);
    new_shape.emplace_back(shape[shape.size() - 1]);
    builder::Type output_type(new_shape, type);
    out_grad = builder::Reshape(out_grad, output_type);
  }

  // calculate DX, DY
  if (y_rank == 1 && x_rank == 1) {
    DX = out_grad * Y;
    DY = X * out_grad;
  } else {
    builder::DotDimensionNumbers dims_attr_dx;
    std::vector<int64_t> lhs_batching_dimensions_dx = {};
    std::vector<int64_t> rhs_batching_dimensions_dx = {};
    std::vector<int64_t> lhs_contracting_dimensions_dx = {};
    std::vector<int64_t> rhs_contracting_dimensions_dx = {};
    if (out_rank <= y_rank || trans_x || y_rank == 1) {
      for (int64_t i = 0; i < max_rank - 1; ++i) {
        if (i < batch_dim) {
          lhs_batching_dimensions_dx.emplace_back(i);
          rhs_batching_dimensions_dx.emplace_back(i);
        } else {
          lhs_contracting_dimensions_dx.emplace_back(i + 1);
          if (trans_y && y_rank != 1) {
            rhs_contracting_dimensions_dx.emplace_back(i);
          } else {
            rhs_contracting_dimensions_dx.emplace_back(i + 1);
          }
        }
      }
    } else {
      lhs_contracting_dimensions_dx.emplace_back(out_rank - 1);
      if (y_rank != 1) {
        if (trans_y) {
          rhs_contracting_dimensions_dx.emplace_back(y_rank - 2);
        } else {
          rhs_contracting_dimensions_dx.emplace_back(y_rank - 1);
        }
      } else {
        rhs_contracting_dimensions_dx.emplace_back(0);
      }
    }
    std::vector<const char *> precision_config = {};

    dims_attr_dx.set_lhs_batching_dimensions(lhs_batching_dimensions_dx);
    dims_attr_dx.set_rhs_batching_dimensions(rhs_batching_dimensions_dx);
    dims_attr_dx.set_lhs_contracting_dimensions(lhs_contracting_dimensions_dx);
    dims_attr_dx.set_rhs_contracting_dimensions(rhs_contracting_dimensions_dx);
    DX = builder::DotGeneral(out_grad, Y, dims_attr_dx, precision_config);
    DX.SetAttribute("op_type", builder::Attribute("DotBPI"));
    if (x_rank == y_rank && x_rank > 3) {
      auto dx_shape = DX.GetType().GetShape();
      auto true_dx_shape = x_shape;
      if (trans_x) {
        true_dx_shape[x_rank - 2] = x_shape[x_rank - 1];
        true_dx_shape[x_rank - 1] = x_shape[x_rank - 2];
      }
      if (dx_shape != true_dx_shape) {
        std::vector<int64_t> axis;
        for (int64_t i = 0; i < x_rank; ++i) {
          if (dx_shape[i] != true_dx_shape[i]) axis.push_back(i);
        }
        DX = builder::ReduceSum(DX, true, axis);
      }
    }

    builder::DotDimensionNumbers dims_attr_dy;
    std::vector<int64_t> lhs_batching_dimensions_dy = {};
    std::vector<int64_t> rhs_batching_dimensions_dy = {};
    std::vector<int64_t> lhs_contracting_dimensions_dy = {};
    std::vector<int64_t> rhs_contracting_dimensions_dy = {};
    for (int64_t i = 0; i < max_rank - 1; ++i) {
      if (i < batch_dim) {
        lhs_batching_dimensions_dy.emplace_back(i);
        rhs_batching_dimensions_dy.emplace_back(i);
      } else {
        if (trans_x && x_rank != 1) {
          lhs_contracting_dimensions_dy.emplace_back(i + 1);
        } else {
          lhs_contracting_dimensions_dy.emplace_back(i);
        }
        rhs_contracting_dimensions_dy.emplace_back(i);
      }
    }
    dims_attr_dy.set_lhs_batching_dimensions(lhs_batching_dimensions_dy);
    dims_attr_dy.set_rhs_batching_dimensions(rhs_batching_dimensions_dy);
    dims_attr_dy.set_lhs_contracting_dimensions(lhs_contracting_dimensions_dy);
    dims_attr_dy.set_rhs_contracting_dimensions(rhs_contracting_dimensions_dy);
    DY = builder::DotGeneral(X, out_grad, dims_attr_dy, precision_config);
    DY.SetAttribute("op_type", builder::Attribute("DotBPK"));
    if (x_rank == y_rank && x_rank > 3) {
      auto dy_shape = DY.GetType().GetShape();
      auto true_dy_shape = y_shape;
      if (trans_y) {
        true_dy_shape[y_rank - 2] = y_shape[y_rank - 1];
        true_dy_shape[y_rank - 1] = y_shape[y_rank - 2];
      }
      if (dy_shape != true_dy_shape) {
        std::vector<int64_t> axis;
        for (int64_t i = 0; i < x_rank; ++i) {
          if (dy_shape[i] != true_dy_shape[i]) axis.push_back(i);
        }
        DY = builder::ReduceSum(DY, true, axis);
      }
    }
  }

  // transpose back to original input shape(transpose_x or transpose_y)
  std::vector<int64_t> data_trans;
  for (int64_t i = 0; i < max_rank - 2; ++i) {
    data_trans.emplace_back(i);
  }
  data_trans.emplace_back(max_rank - 1);
  data_trans.emplace_back(max_rank - 2);
  if (trans_x && x_rank != 1) {
    DX = builder::Transpose(DX, data_trans);
  }
  if (trans_y && y_rank != 1) {
    DY = builder::Transpose(DY, data_trans);
  }
  // reduce sum when x_rank != y_rank
  if (x_rank > y_rank) {
    std::vector<int64_t> axis;
    if (batch_dim != 0) {
      for (int64_t i = 0; i < batch_dim; ++i) {
        axis.emplace_back(i);
      }
      DY = builder::ReduceSum(DY, false, axis);
    }
  } else if (x_rank < y_rank) {
    std::vector<int64_t> axis;
    if (batch_dim != 0) {
      for (int64_t i = 0; i < batch_dim; ++i) {
        axis.emplace_back(i);
      }
      DX = builder::ReduceSum(DX, false, axis);
    }
  }

  // reshape when (x_rank ==1 or y_rank==1) and (x_rank != y_rank)
  if (x_rank == 1 && y_rank != 1) {
    auto dx_shape = DX.GetType().GetShape();
    auto dx_type = DX.GetType().GetPrimitiveType();
    std::vector<int64_t> dx_new_shape;
    dx_new_shape.push_back(dx_shape[dx_shape.size() - 1]);
    builder::Type dx_output_type(dx_new_shape, dx_type);
    DX = builder::Reshape(DX, dx_output_type);
  } else if (y_rank == 1 && x_rank != 1) {
    auto dy_shape = DY.GetType().GetShape();
    auto dy_type = DY.GetType().GetPrimitiveType();
    std::vector<int64_t> dy_new_shape;
    dy_new_shape.push_back(dy_shape[0]);
    builder::Type dy_output_type(dy_new_shape, dy_type);
    DY = builder::Reshape(DY, dy_output_type);
  }

  // return result
  auto output_name_map = op->Outputs();
  bool out_x_grad = output_name_map.count("X@GRAD") != 0 &&
                    output_name_map["X@GRAD"].size() > 0;
  bool out_y_grad = output_name_map.count("Y@GRAD") != 0 &&
                    output_name_map["Y@GRAD"].size() > 0;
  if (!out_x_grad && !out_y_grad) {
    return nullptr;
  } else if (!out_x_grad) {
    return std::make_shared<GcuOp>(DY);
  } else if (!out_y_grad) {
    return std::make_shared<GcuOp>(DX);
  }
  std::vector<GcuOp> outputs;
  outputs.push_back(DX);
  outputs.push_back(DY);
  auto dx_shape = DX.GetType().GetShape();
  auto dy_shape = DY.GetType().GetShape();
  std::vector<std::vector<int64_t>> tuple_shape = {{dx_shape}, {dy_shape}};
  std::vector<builder::PrimitiveType> tuple_dtype;
  tuple_dtype.push_back(DX.GetType().GetPrimitiveType());
  tuple_dtype.push_back(DY.GetType().GetPrimitiveType());

  GcuType outputs_type(tuple_shape, tuple_dtype);
  std::vector<std::string> output_names{"X@GRAD", "Y@GRAD"};

  std::string output_names_attr = "";
  if (output_name_map.count(output_names[0]) > 0) {
    output_names_attr += output_name_map[output_names[0]][0];
  }
  for (size_t i = 1; i < output_names.size(); ++i) {
    if (output_name_map.count(output_names[i]) > 0) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
  }
  auto res = builder::Tuple(outputs, outputs_type);
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kMatMulV2, INSENSITIVE, MatMulV2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMatMulV2Grad,
                           INSENSITIVE,
                           MatMulV2GradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
