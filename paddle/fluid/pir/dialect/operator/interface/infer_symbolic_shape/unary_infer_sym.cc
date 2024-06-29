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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect {
using paddle::dialect::details::CreateShapeOrDataForXShape;

bool ArgmaxOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  bool flatten = GetBoolAttr(op, "flatten");
  bool keepdims = GetBoolAttr(op, "keepdims");

  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &axis_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  int axis =
      static_cast<int>(axis_shape_or_data.data().value()[0].Get<int64_t>());

  const std::vector<symbol::DimExpr> &input_sym_shape =
      input_shape_or_data.data().has_value()
          ? input_shape_or_data.data().value()
          : input_shape_or_data.shape();

  int rank = input_sym_shape.size();
  if (axis < 0) axis += rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    if (flatten) {
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(rank));
      } else {
        out_sym_shape.emplace_back(std::int64_t(0));
      }
    } else {
      for (int i = 0; i < axis; i++) {
        out_sym_shape.emplace_back(input_sym_shape[i]);
      }
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(1));
      }

      for (int i = axis + 1; i < rank; i++) {
        out_sym_shape.emplace_back(input_sym_shape[i]);
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ArgminOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return ArgmaxOpInferSymbolicShape(op, infer_context);
}

bool AsComplexOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.pop_back();
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool AsRealOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.push_back(symbol::DimExpr(2));
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool CummaxOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  infer_context->SetShapeOrDataForValue(op->result(1), operand_shape_or_data);
  return true;
}
bool CumminOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return CummaxOpInferSymbolicShape(op, infer_context);
}
bool CumprodOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  return true;
}
bool Cumprod_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CumprodOpInferSymbolicShape(op, infer_context);
}
bool CumsumOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  bool flatten = GetBoolAttr(op, "flatten");
  if (flatten) {
    symbol::DimExpr product{1};
    const auto &dim_exprs = operand_shape_or_data.shape();
    for (const auto &dim_expr : dim_exprs) {
      product = product * dim_expr;
    }
    const std::vector<symbol::DimExpr> out_dims = {product};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  } else {
    infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  }
  return true;
}
bool Cumsum_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CumsumOpInferSymbolicShape(op, infer_context);
}

bool DiagEmbedOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int dim1 = attributes.at("dim1").dyn_cast<pir::Int32Attribute>().data();
  int dim2 = attributes.at("dim2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
  int64_t offset_ = static_cast<int64_t>(std::abs(offset));
  symbol::DimExpr new_dim_len =
      symbol::DimExpr(offset_) + x_dims[x_dims.size() - 1];

  const auto &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = x_dims;
    out_dims.pop_back();
    out_dims.insert(out_dims.begin() + std::min(dim1_, dim2_), new_dim_len);
    out_dims.insert(out_dims.begin() + std::max(dim1_, dim2_), new_dim_len);
    return out_dims;
  }();
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool DiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis1 = attributes.at("axis1").dyn_cast<pir::Int32Attribute>().data();
  int axis2 = attributes.at("axis2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

  auto out_dims = x_dims;
  auto axis1_size = out_dims[axis1_];
  auto axis2_size = out_dims[axis2_];
  out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  symbol::DimExprBuilder builder;
  symbol::DimExpr zero{0};
  symbol::DimExpr res_shape;
  symbol::DimExpr offset_sym{offset};
  if (offset == 0) {
    res_shape = builder.Min(axis1_size, axis2_size);
  } else if (offset > 0) {
    if (axis2_size.isa<int64_t>()) {
      res_shape = (axis2_size.dyn_cast<int64_t>() - offset) > 0
                      ? builder.Min(axis1_size, axis2_size - offset_sym)
                      : zero;
    } else {
      res_shape = infer_context->GetNextSymName();
    }
  } else {
    if (axis1_size.isa<int64_t>()) {
      res_shape = (axis1_size.dyn_cast<int64_t>() + offset) > 0
                      ? builder.Min(axis1_size + offset_sym, axis2_size)
                      : zero;
    } else {
      res_shape = infer_context->GetNextSymName();
    }
  }
  out_dims.push_back(symbol::SimplifyDimExpr(res_shape));

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool DistributeFpnProposalsOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &attributes = op->attributes();
  int32_t min_level =
      attributes.at("min_level").dyn_cast<pir::Int32Attribute>().data();
  int32_t max_level =
      attributes.at("max_level").dyn_cast<pir::Int32Attribute>().data();
  int32_t num_levels = max_level - min_level + 1;
  int64_t batch_size = 1;

  symbol::DimExpr num_rois =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .shape()
          .at(0);

  const auto &multi_rois_out_shape = [&]() {
    symbol::TensorListShapeOrDataDimExprs multi_rois_out_shape;
    if (num_levels == 1) {
      multi_rois_out_shape.emplace_back(
          symbol::TensorShapeOrDataDimExprs({num_rois, 4}));
    } else {
      symbol::DimExpr last_dim = num_rois;
      for (int i = 0; i < num_levels - 1; i++) {
        const auto &next_sym_name = infer_context->GetNextSymName();
        std::vector<symbol::DimExpr> level_dim = {next_sym_name, 4};
        multi_rois_out_shape.emplace_back(
            symbol::TensorShapeOrDataDimExprs(level_dim));
        last_dim = last_dim - level_dim[0];
      }
      multi_rois_out_shape.emplace_back(symbol::TensorShapeOrDataDimExprs(
          {infer_context->GetNextSymName(), 4}));
    }

    return multi_rois_out_shape;
  }();

  const auto &rois_num_per_level_out_shape = [&]() {
    symbol::TensorListShapeOrDataDimExprs rois_num_per_level_out_shape;
    rois_num_per_level_out_shape.resize(
        num_levels, symbol::TensorShapeOrDataDimExprs({batch_size}));
    return rois_num_per_level_out_shape;
  }();

  const auto &restore_ind = [&]() {
    return symbol::TensorShapeOrDataDimExprs({num_rois, 1});
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), multi_rois_out_shape);
  infer_context->SetShapeOrDataForValue(op->result(1),
                                        rois_num_per_level_out_shape);
  infer_context->SetShapeOrDataForValue(op->result(2), restore_ind);
  return true;
}

bool EinsumOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool KthvalueOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  bool keepdim = GetBoolAttr(op, "keepdim");

  const auto &input_dims = operand_shape_or_data.shape();
  const int &dim_size = input_dims.size();
  if (axis < 0) axis += dim_size;
  std::vector<symbol::DimExpr> out_dims;
  for (int i = 0; i < axis; i++) {
    out_dims.emplace_back(input_dims[i]);
  }
  if (keepdim && dim_size > 0) {
    out_dims.emplace_back(symbol::DimExpr(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    out_dims.emplace_back(input_dims[i]);
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  infer_context->SetShapeOrDataForValue(op->result(1), shape_data);
  return true;
}

bool LogcumsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  // same as CumsumOpInferSymbolicShape
  return CumsumOpInferSymbolicShape(op, infer_context);
}

bool LogsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  std::vector<int> axis_in = details::GetVectorAttr<int>(op, "axis");
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int &t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  bool reduce_all = axis.size() == 0 ? true : false;
  return details::ReduceInferDim(op, infer_context, axis, keepdim, reduce_all);
}

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");

  const std::vector<int64_t> axis = [&] {
    pir::Operation *axis_gen_op = op->operand_source(1).defining_op();
    std::vector<int64_t> axis_vec;
    if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
      axis_vec = details::GetVectorAttr(
          axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    } else {
      // TODO(lanxianghit): there's other source: pir::VectorType,
      // paddle::dialect::DenseTensorType, but after PRIM, maybe always
      // FullIntArrayOp, to be confirmed
      PADDLE_THROW(
          phi::errors::Unimplemented("MaxOpInferSymbolicShape: 'axis' only "
                                     "support FullIntArrayOp's result now."));
    }
    return axis_vec;
  }();

  bool reduce_all = axis.size() == 0 ? true : false;

  return details::ReduceInferDim(op, infer_context, axis, keepdim, reduce_all);
}

bool MinOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  return MaxOpInferSymbolicShape(op, infer_context);
}

bool NonzeroOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int rank = x_shape.size();

  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      phi::errors::InvalidArgument(
          "Input(x) should have number of dimension at least 1."));

  std::string sym_name = infer_context->GetNextSymName();
  std::vector<symbol::DimExpr> out_shape{symbol::DimExpr{sym_name},
                                         symbol::DimExpr{rank}};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool PadOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  // input(0): Tensor x
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    phi::errors::InvalidArgument(
                        "InferSymbolicShape of PadOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();

  // input(1): int[] paddings
  std::vector<int> paddings =
      paddle::dialect::details::GetVectorAttr<int>(op, "paddings");
  PADDLE_ENFORCE_EQ(rank * 2,
                    paddings.size(),
                    phi::errors::InvalidArgument(
                        "The size of paddings should be 2 * input's rank. But "
                        "got paddings.size() = %d, input's rank = %d.",
                        paddings.size(),
                        rank));

  // output
  const auto &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      out_dims.push_back(x_dims_sym[i] + paddings[2 * i] + paddings[2 * i + 1]);
    }
    return out_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dims));

  return true;
}

bool Pad3dOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    5,
                    common::errors::InvalidArgument(
                        "The size of Input(X)'s dimension should be equal to "
                        "5, but received %d. ",
                        x_shape.size()));
  const auto &paddings_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  if (!paddings_shape.data().has_value()) {
    std::stringstream ss;
    ss << paddings_shape;
    PADDLE_THROW(
        common::errors::InvalidArgument("The data of paddings's symbol shape "
                                        "should have value, but now got [%s].",
                                        ss.str()));
  }
  const std::string &data_format =
      op->attribute<pir::StrAttribute>("data_format").AsString();

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = x_shape;
    const auto &paddings = paddings_shape.data().value();
    PADDLE_ENFORCE_EQ(paddings.size(),
                      6,
                      common::errors::InvalidArgument(
                          "Shape of Input(Paddings) should be equal to "
                          "[6], but received [%d].",
                          paddings.size()));
    if (data_format == "NCDHW") {
      out_dims[1] = x_shape[1];
      out_dims[2] = x_shape[2] + paddings[4] + paddings[5];
      out_dims[3] = x_shape[3] + paddings[2] + paddings[3];
      out_dims[4] = x_shape[4] + paddings[0] + paddings[1];
    } else {
      out_dims[1] = x_shape[1] + paddings[4] + paddings[5];
      out_dims[2] = x_shape[2] + paddings[2] + paddings[3];
      out_dims[3] = x_shape[3] + paddings[0] + paddings[1];
      out_dims[4] = x_shape[4];
    }
    return out_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dims));

  return true;
}

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keep_dim");
  bool reduce_all = GetBoolAttr(op, "reduce_all");

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = details::GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("ProdOpInferSymbolicShape: 'axis' only "
                                   "support FullIntArrayOp's result now."));
  }

  return true;
}

bool RepeatInterleaveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const auto &attributes = op->attributes();
  int repeats = attributes.at("repeats").dyn_cast<pir::Int32Attribute>().data();
  // what should I do if axis is null
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const std::vector<symbol::DimExpr> &in_dims_sym = [&] {
    std::vector<symbol::DimExpr> dims;
    if (operand_shape_or_data.data().has_value()) {
      dims = operand_shape_or_data.data().value();
    } else {
      dims = operand_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = in_dims_sym.size();
  if (axis < 0) axis += x_rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < x_rank; i++) {
      if (i == axis) {
        out_sym_shape.push_back(in_dims_sym[i] * repeats);
      } else {
        out_sym_shape.push_back(in_dims_sym[i]);
      }
    }
    return out_sym_shape;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{
          symbol::TensorShapeOrDataDimExprs(out_sym_shape)});

  return true;
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_dim_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &shape_dim_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &GetProduct = [&](const auto &dim_exprs, const auto &Filter) {
    symbol::DimExpr product{1};
    for (const auto &dim_expr : dim_exprs) {
      if (Filter(dim_expr)) {
        product = product * dim_expr;
      }
    }
    return product;
  };

  const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
    }
    return true;
  };

  const auto &IsZero = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() == static_cast<int64_t>(0);
    }
    return false;
  };

  const std::vector<symbol::DimExpr> out_dims = [&] {
    const auto &original_shape =
        infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();

    const auto &numel =
        GetProduct(original_shape, [](const auto &) { return true; });

    ExprVec target_shape = details::GetExprVecFromData(shape_dim_expr);
    const auto &product_exclude_minus_one =
        GetProduct(target_shape, IsNotMinusOne);

    const auto &input_dims = target_shape;

    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); ++i) {
      auto out_dim_expr = IsNotMinusOne(input_dims[i])
                              ? input_dims[i]
                              : (numel / product_exclude_minus_one);
      out_dim_expr = IsZero(input_dims[i]) ? original_shape[i] : out_dim_expr;
      out_dims.emplace_back(out_dim_expr);
    }

    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data = [&] {
    if (x_dim_expr.data().has_value()) {
      return symbol::TensorShapeOrDataDimExprs(out_dims,
                                               x_dim_expr.data().value());
    }
    return symbol::TensorShapeOrDataDimExprs(out_dims);
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      CreateShapeOrDataForXShape(
          infer_context->GetShapeOrDataForValue(op->operand_source(0))));
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReshapeOpInferSymbolicShape(op, infer_context);
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &out_data = operand_shape_or_data.shape();
  const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
  symbol::ShapeOrDataDimExprs shape_or_data{
      symbol::TensorShapeOrDataDimExprs(shape, out_data)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_or_data);
  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ShapeOpInferSymbolicShape(op, infer_context);
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  pir::Value operand_starts = op->operand_source(1);
  pir::Value operand_ends = op->operand_source(2);
  pir::Value res = op->result(0);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  const symbol::ShapeOrDataDimExprs &starts_shape_data =
      infer_context->GetShapeOrDataForValue(operand_starts);
  const symbol::ShapeOrDataDimExprs &ends_shape_data =
      infer_context->GetShapeOrDataForValue(operand_ends);

  std::vector<int64_t> axes_vec = details::GetVectorAttr(op, "axes");

  // // Currently, we DO NOT support any element in `starts` is a Symbol.
  ExprVec starts = slice_utils::GetExprVecFromData(starts_shape_data);
  ExprVec ends = slice_utils::GetExprVecFromData(ends_shape_data);

  std::vector<int64_t> infer_flags = details::GetVectorAttr(op, "infer_flags");

  const std::vector<int64_t> decrease_axis =
      details::GetVectorAttr(op, "decrease_axis");

  infer_context->SetShapeOrDataForValue(
      res,
      slice_utils::SliceRawInferSymbolicShape(operand_shape_or_data,
                                              starts,
                                              ends,
                                              axes_vec,
                                              infer_flags,
                                              decrease_axis));

  return true;
}

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  // input
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    phi::errors::InvalidArgument(
                        "InferSymbolicShape of SplitOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  CHECK(op->operand_source(2).defining_op()->isa<paddle::dialect::FullOp>());

  int64_t axis = op->operand_source(2)
                     .defining_op<paddle::dialect::FullOp>()
                     .attributes()
                     .at("value")
                     .dyn_cast<paddle::dialect::ScalarAttribute>()
                     .data()
                     .to<int64_t>();
  size_t rank = x_dims_sym.size();
  axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

  // sections
  const std::vector<symbol::DimExpr> &sections_sym = [&] {
    const auto &sections_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    std::vector<symbol::DimExpr> sections_sym;
    if (sections_shape_or_data.data().has_value()) {
      sections_sym = sections_shape_or_data.data().value();
    } else {
      sections_sym = sections_shape_or_data.shape();
    }
    return sections_sym;
  }();

  // output
  const symbol::TensorListShapeOrDataDimExprs &output_shape_data_list = [&] {
    const auto &GetSum = [&](const auto &dim_exprs, const auto &Filter) {
      symbol::DimExpr sum{0};
      for (const auto &dim_expr : dim_exprs) {
        if (Filter(dim_expr)) {
          sum = sum + dim_expr;
        }
      }
      return sum;
    };
    const auto &All = [&](const auto &dim_exprs, const auto &Cond) {
      for (const auto &dim_expr : dim_exprs) {
        if (!Cond(dim_expr)) {
          return false;
        }
      }
      return true;
    };
    const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
      if (dim_expr.isa<int64_t>()) {
        return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
      }
      return true;
    };
    const auto &sum_exclude_minus_one = GetSum(sections_sym, IsNotMinusOne);

    const bool &all_sections_sym_not_minus_one =
        All(sections_sym, IsNotMinusOne);
    if (all_sections_sym_not_minus_one) {
      infer_context->AddEqualCstr(x_dims_sym[axis], sum_exclude_minus_one);
    }

    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    std::vector<symbol::DimExpr> output_dims_sym = x_dims_sym;
    if (!all_sections_sym_not_minus_one && sections_sym.size() == 1) {
      VLOG(3) << "[SplitOp]-1 is the only split section. The output shape is "
                 "identical to the input shape.";
      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
      return shape_data_list;
    }
    for (uint32_t idx = 0; idx < sections_sym.size(); idx++) {
      const auto &section_sym = sections_sym[idx];
      output_dims_sym[axis] = IsNotMinusOne(section_sym)
                                  ? section_sym
                                  : x_dims_sym[axis] - sum_exclude_minus_one;

      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool SplitWithNumOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &axis_shape_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  PADDLE_ENFORCE_EQ(
      axis_shape_data.data().has_value(),
      true,
      phi::errors::InvalidArgument(
          "In InferSymbolicShape, axis of SplitWithNumOp is null"));
  const std::vector<symbol::DimExpr> &axis_data =
      axis_shape_data.data().value();
  PADDLE_ENFORCE_EQ(
      axis_data.size() == 1,
      true,
      phi::errors::InvalidArgument(
          "In SplitWithNumOp, data of axis should be one dimension"));
  if (!axis_data[0].isa<int64_t>()) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The type of axis must be int64_t"));
  }
  int64_t axis = axis_data[0].dyn_cast<int64_t>();

  const auto &attributes = op->attributes();
  int num = attributes.at("num").dyn_cast<pir::Int32Attribute>().data();
  const auto &x_s_or_d =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  int rank = x_s_or_d.shape().size();
  axis = axis < 0 ? axis + rank : axis;

  symbol::DimExpr input_axis_dim = x_s_or_d.shape().at(axis);
  symbol::DimExpr axis_shape = input_axis_dim / symbol::DimExpr{num};

  const auto &out_s_d = [&] {
    std::vector<symbol::DimExpr> out_s_d;
    for (size_t i = 0; i < x_s_or_d.shape().size(); ++i) {
      const auto &sym_dim =
          axis == static_cast<int64_t>(i) ? axis_shape : x_s_or_d.shape()[i];
      out_s_d.push_back(sym_dim);
    }
    return symbol::TensorShapeOrDataDimExprs(out_s_d);
  }();

  symbol::TensorListShapeOrDataDimExprs outs_s_d(num, out_s_d);
  infer_context->SetShapeOrDataForValue(op->result(0),
                                        symbol::ShapeOrDataDimExprs{outs_s_d});
  return true;
}

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  bool reduce_all = false;

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = details::GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    if (axis.size() == 0) {
      reduce_all = true;
    }
    return details::ReduceInferDim(
        op, infer_context, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("SumOpInferSymbolicShape: 'axis' only "
                                   "support FullIntArrayOp's result now."));
  }

  return true;
}

bool TileOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_x = op->operand_source(0);
  symbol::ShapeOrDataDimExprs x_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_x);
  pir::Value operand_repeat_times = op->operand_source(1);
  symbol::ShapeOrDataDimExprs repeat_times_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_repeat_times);

  std::vector<symbol::DimExpr> x_dimexpr;
  if (x_shape_or_data.data().has_value()) {
    x_dimexpr = x_shape_or_data.data().value();
  } else {
    x_dimexpr = x_shape_or_data.shape();
  }

  std::vector<symbol::DimExpr> repeat_times_dimexpr;
  if (repeat_times_shape_or_data.data().has_value()) {
    repeat_times_dimexpr = repeat_times_shape_or_data.data().value();
  } else {
    repeat_times_dimexpr = repeat_times_shape_or_data.shape();
  }
  if (repeat_times_dimexpr.empty()) {
    repeat_times_dimexpr = std::vector<symbol::DimExpr>(x_dimexpr.size(), 1);
  }

  auto out_rank = std::max(static_cast<size_t>(x_dimexpr.size()),
                           repeat_times_dimexpr.size());
  std::vector<symbol::DimExpr> out_shape(out_rank);
  if (x_dimexpr.size() > repeat_times_dimexpr.size()) {
    auto diff = x_dimexpr.size() - repeat_times_dimexpr.size();
    repeat_times_dimexpr.insert(repeat_times_dimexpr.begin(), diff, 1);
  } else {
    auto diff = repeat_times_dimexpr.size() - x_dimexpr.size();
    x_dimexpr.insert(x_dimexpr.begin(), diff, 1);
  }

  for (size_t i = 0; i < repeat_times_dimexpr.size(); ++i) {
    out_shape[i] = x_dimexpr[i] * repeat_times_dimexpr[i];
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool TopkOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  symbol::ShapeOrDataDimExprs x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  symbol::ShapeOrDataDimExprs k_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  const std::vector<symbol::DimExpr> &in_dims_sym = [&] {
    std::vector<symbol::DimExpr> dims;
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = in_dims_sym.size();

  int k = k_shape_or_data.data().value()[0].Get<int64_t>();

  if (axis < 0) axis += x_rank;
  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    for (int i = 0; i < x_rank; ++i) {
      if (i == axis) {
        out_sym_shape.push_back(symbol::DimExpr(k));
      } else {
        out_sym_shape.push_back(in_dims_sym[i]);
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  infer_context->SetShapeOrDataForValue(op->result(1), shape_data);

  return true;
}

bool TopkV1OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return TopkOpInferSymbolicShape(op, infer_context);
}

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<pir::Attribute> perm =
      op->attributes().at("perm").dyn_cast<pir::ArrayAttribute>().AsVector();
  if (perm.size() == 1) {
    // perm must be [0], which means nothing to do with input, just copy the
    // info from input
    infer_context->SetShapeOrDataForValue(
        op->result(0),
        infer_context->GetShapeOrDataForValue(op->operand_source(0)));
    return true;
  }
  const std::vector<symbol::DimExpr> &x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto &x_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(0));
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = x_dims.size();

  const std::vector<int32_t> formatted_axis = [x_rank, &perm] {
    std::vector<int32_t> out(perm.size(), 0);
    std::transform(perm.begin(),
                   perm.end(),
                   out.begin(),
                   [](pir::Attribute &p) -> int32_t {
                     return p.dyn_cast<pir::Int32Attribute>().data();
                   });

    // format the negative axis
    std::for_each(out.begin(), out.end(), [x_rank](int32_t &v) {
      if (v < 0) {
        v += x_rank;
      }
    });
    return out;
  }();

  int axis_size = static_cast<int>(formatted_axis.size());

  std::vector<symbol::DimExpr> out_dims(x_dims);
  for (int i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[formatted_axis[i]];
  }

  infer_context->SetShapeOrDataForValue(op->result(0),
                                        ShapeOrData{TensorExprs(out_dims)});

  return true;
}

bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return TransposeOpInferSymbolicShape(op, infer_context);
}

bool SqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_ENFORCE_EQ(
      op->num_operands(),
      2,
      phi::errors::InvalidArgument(
          "SqueezeOpInferSymbolicShape ONLY support num_operands() == 2 "
          "now, but got %d operands",
          op->num_operands()));

  auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  auto axes_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> in_dims_sym;
  if (x_shape_or_data.data().has_value()) {
    in_dims_sym = x_shape_or_data.data().value();
  } else {
    in_dims_sym = x_shape_or_data.shape();
  }

  std::vector<symbol::DimExpr> squeeze_dims_sym;
  if (axes_shape_or_data.data().has_value()) {
    squeeze_dims_sym = axes_shape_or_data.data().value();
  } else {
    squeeze_dims_sym = axes_shape_or_data.shape();
  }

  std::vector<int> squeeze_dims;
  for (auto squeeze_dim : squeeze_dims_sym) {
    PADDLE_ENFORCE_EQ(
        squeeze_dim.Has<std::int64_t>(),
        true,
        phi::errors::InvalidArgument(
            "in SqueezeOpInferSymbolicShape, axes must be known int type, "
            "but got: %s",
            symbol::ToString(squeeze_dim)));
    squeeze_dims.emplace_back(
        static_cast<int>(squeeze_dim.Get<std::int64_t>()));
  }

  // GetOutputSqueezeShape
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims_sym.size(), false);
  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (size_t i = 0; i < in_dims_sym.size(); ++i) {
      // TODO(lanxianghit): if symbol here, maybe we need the result of dim expr
      // simplification
      if (in_dims_sym[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      if (in_dims_sym.size() == 0) {
        continue;
      }
      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims_sym.size()
                                        : squeeze_dims[i];

      if (!should_squeeze[current]) {
        // At compile time, dim of SYMBOL is allowed to squeeze?
        if (in_dims_sym[current] == 1) {
          should_squeeze[current] = true;
        } else if (!in_dims_sym[current].Has<std::int64_t>()) {
          should_squeeze[current] = true;
        } else {
          should_squeeze[current] = true;
        }
      }
    }
  }

  // Make output dimensions
  std::vector<symbol::DimExpr> output_shape_sym;
  for (size_t i = 0; i < in_dims_sym.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape_sym.emplace_back(in_dims_sym[i]);
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(output_shape_sym)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  infer_context->SetShapeOrDataForValue(
      op->result(1), CreateShapeOrDataForXShape(x_shape_or_data));

  return true;
}
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return SqueezeOpInferSymbolicShape(op, infer_context);
}

bool UnbindOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  // input
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      phi::errors::InvalidArgument(
          "InferSymbolicShape of UnbindOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  int axis = op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();
  int rank = x_dims_sym.size();
  axis = axis >= 0 ? axis : axis + rank;

  // output
  const symbol::TensorListShapeOrDataDimExprs &output_shape_data_list = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    std::vector<symbol::DimExpr> output_dims_sym = x_dims_sym;

    const symbol::DimExpr &unbound_dim = x_dims_sym.at(axis);
    PADDLE_ENFORCE_EQ(unbound_dim.isa<int64_t>(),
                      true,
                      phi::errors::InvalidArgument(
                          "InferSymbolicShape of UnbindOp only support unbound "
                          "dim with constant length!"));
    output_dims_sym.erase(output_dims_sym.begin() + axis);
    const int64_t unbound_dim_length = unbound_dim.dyn_cast<int64_t>();

    for (uint32_t idx = 0; idx < unbound_dim_length; idx++) {
      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool UniqueOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      phi::errors::InvalidArgument(
          "InferSymbolicShape of UniqueOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();
  std::vector<int> axes =
      paddle::dialect::details::GetVectorAttr<int>(op, "axis");

  symbol::DimExpr unique_dim_sym =
      infer_context->GetNextSymName();  // unknown until runtime

  const std::vector<symbol::DimExpr> &counts_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.push_back(unique_dim_sym);
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &index_dims = counts_dims;

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    if (axes.empty()) {
      return counts_dims;
    }
    std::vector<symbol::DimExpr> out_dims = x_dims_sym;
    int axis = axes[0];
    axis = axis >= 0 ? axis : axis + rank;
    out_dims[axis] = unique_dim_sym;
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &inverse_dims = [&] {
    std::vector<symbol::DimExpr> inverse_dims;
    if (axes.empty()) {
      // flatten before unique
      symbol::DimExpr product{1};
      for (const auto &x_dim : x_dims_sym) {
        product = product * x_dim;
      }
      inverse_dims.push_back(product);
    } else {
      int axis = axes[0];
      axis = axis >= 0 ? axis : axis + rank;
      inverse_dims.push_back(x_dims_sym[axis]);
    }
    return inverse_dims;
  }();

  bool return_index = GetBoolAttr(op, "return_index");
  bool return_inverse = GetBoolAttr(op, "return_inverse");
  bool return_counts = GetBoolAttr(op, "return_counts");

  symbol::ShapeOrDataDimExprs empty{symbol::TensorShapeOrDataDimExprs{}};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs{out_dims});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      return_index ? symbol::TensorShapeOrDataDimExprs{index_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      return_inverse ? symbol::TensorShapeOrDataDimExprs{inverse_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(3),
      return_counts ? symbol::TensorShapeOrDataDimExprs{counts_dims} : empty);

  return true;
}

bool UniqueConsecutiveOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(
      x_shape_or_data.data().has_value(),
      false,
      phi::errors::InvalidArgument(
          "InferSymbolicShape of UniqueConsecutiveOp only support input with "
          "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();
  const size_t rank = x_dims_sym.size();
  std::vector<int> axes =
      paddle::dialect::details::GetVectorAttr<int>(op, "axis");

  symbol::DimExpr unique_dim_sym =
      infer_context->GetNextSymName();  // unknown until runtime

  const std::vector<symbol::DimExpr> &counts_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims.push_back(unique_dim_sym);
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    if (axes.empty()) {
      return counts_dims;
    }
    std::vector<symbol::DimExpr> out_dims = x_dims_sym;
    int axis = axes[0];
    axis = axis >= 0 ? axis : axis + rank;
    out_dims[axis] = unique_dim_sym;
    return out_dims;
  }();

  const std::vector<symbol::DimExpr> &inverse_dims = [&] {
    std::vector<symbol::DimExpr> inverse_dims;
    if (axes.empty()) {
      // flatten before unique
      symbol::DimExpr product{1};
      for (const auto &x_dim : x_dims_sym) {
        product = product * x_dim;
      }
      inverse_dims.push_back(product);
    } else {
      int axis = axes[0];
      axis = axis >= 0 ? axis : axis + rank;
      inverse_dims.push_back(x_dims_sym[axis]);
    }
    return inverse_dims;
  }();

  bool return_inverse = GetBoolAttr(op, "return_inverse");
  bool return_counts = GetBoolAttr(op, "return_counts");

  symbol::ShapeOrDataDimExprs empty{symbol::TensorShapeOrDataDimExprs{}};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs{out_dims});
  infer_context->SetShapeOrDataForValue(
      op->result(1),
      return_inverse ? symbol::TensorShapeOrDataDimExprs{inverse_dims} : empty);
  infer_context->SetShapeOrDataForValue(
      op->result(2),
      return_counts ? symbol::TensorShapeOrDataDimExprs{counts_dims} : empty);

  return true;
}

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  PADDLE_ENFORCE_EQ(
      op->num_operands(),
      2,
      phi::errors::InvalidArgument(
          "UnsqueezeOp InferSymbolicShape ONLY support num_operands() == 2 "
          "now, but got %d operands",
          op->num_operands()));

  auto x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  auto axes_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> x_sym_shape;
  if (x_shape_or_data.data().has_value()) {
    x_sym_shape = x_shape_or_data.data().value();
  } else {
    x_sym_shape = x_shape_or_data.shape();
  }
  int x_dims_size = x_sym_shape.size();

  std::vector<symbol::DimExpr> axes_sym;
  if (axes_shape_or_data.data().has_value()) {
    axes_sym = axes_shape_or_data.data().value();
  } else {
    axes_sym = axes_shape_or_data.shape();
  }
  int axes_sym_size = axes_sym.size();

  // GetUnsqueezeShape
  int output_rank = x_dims_size + axes_sym_size;
  std::vector<symbol::DimExpr> result_sym_dims(output_rank, 0);

  int cur_output_rank = x_dims_size;
  for (auto axis_expr : axes_sym) {
    PADDLE_ENFORCE_EQ(
        axis_expr.Has<std::int64_t>(),
        true,
        phi::errors::InvalidArgument(
            "in UnsqueezeOpInferSymbolicShape, axes must be known int type, "
            "but got: %s",
            symbol::ToString(axis_expr)));
    int axis = static_cast<int>(axis_expr.Get<std::int64_t>());
    int cur = axis < 0 ? axis + cur_output_rank + 1 : axis;

    // Move old axis, and insert new axis
    for (int i = cur_output_rank; i >= cur; --i) {
      if (result_sym_dims[i] == 1) {
        // Move axis
        result_sym_dims[i + 1] = 1;
        result_sym_dims[i] = 0;
      }
    }
    result_sym_dims[cur] = 1;
    // Add the output size.
    cur_output_rank++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_rank; ++out_idx) {
    if (result_sym_dims[out_idx] == 0) {
      result_sym_dims[out_idx] = x_sym_shape[in_idx++];
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(result_sym_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  infer_context->SetShapeOrDataForValue(
      op->result(1), CreateShapeOrDataForXShape(x_shape_or_data));

  return true;
}
bool Unsqueeze_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return UnsqueezeOpInferSymbolicShape(op, infer_context);
}

}  // namespace paddle::dialect
