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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/multiary_infer_sym.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

bool BicubicInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &attributes = op->attributes();

  const std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();
  int out_d = attributes.at("out_d").dyn_cast<pir::Int32Attribute>().data();
  int out_h = attributes.at("out_h").dyn_cast<pir::Int32Attribute>().data();
  int out_w = attributes.at("out_w").dyn_cast<pir::Int32Attribute>().data();

  std::vector<int> size_tensor;
  if (out_d != -1) size_tensor.push_back(out_d);
  if (out_h != -1) size_tensor.push_back(out_h);
  if (out_w != -1) size_tensor.push_back(out_w);

  const DataLayout data_layout = common::StringToDataLayout(data_format);

  if (x.shape().size() == 3) {
    // shape check for 1D interpolate for input tensor shape NCHW
    if (!size_tensor.empty()) {
      // top priority size
      std::vector<symbol::DimExpr> dim_out;
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {x.shape()[0], x.shape()[1], symbol::DimExpr{out_w}};
      } else {
        dim_out = {x.shape()[0], symbol::DimExpr{out_w}, x.shape()[2]};
      }

      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(dim_out)};

      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);
      return true;
    }

    symbol::DimExpr out_w_tmp{0};
    const auto &next_sym = infer_context->GetNextSymName();
    out_w_tmp = symbol::DimExpr(next_sym);

    std::vector<symbol::DimExpr> dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {x.shape()[0], x.shape()[1], out_w_tmp};
    } else {
      dim_out = {x.shape()[0], out_w_tmp, x.shape()[2]};
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  } else if (x.shape().size() == 4) {
    // shape check for 2D interpolate for input tensor shape NCHW
    if (!size_tensor.empty()) {
      // top priority size
      std::vector<symbol::DimExpr> dim_out;
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {x.shape()[0],
                   x.shape()[1],
                   symbol::DimExpr{out_h},
                   symbol::DimExpr{out_w}};
      } else {
        dim_out = {x.shape()[0],
                   symbol::DimExpr{out_h},
                   symbol::DimExpr{out_w},
                   x.shape()[3]};
      }

      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(dim_out)};
      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);
      return true;
    }

    symbol::DimExpr out_h_tmp{0};
    symbol::DimExpr out_w_tmp{0};
    const auto &next_sym = infer_context->GetNextSymName();
    out_h_tmp = symbol::DimExpr(next_sym);
    out_w_tmp = symbol::DimExpr(next_sym);

    std::vector<symbol::DimExpr> dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {x.shape()[0], x.shape()[1], out_h_tmp, out_w_tmp};
    } else {
      dim_out = {x.shape()[0], out_h_tmp, out_w_tmp, x.shape()[3]};
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  } else if (x.shape().size() == 5) {
    // shape check for 3D interpolate for input tensor shape NCDHW
    if (!size_tensor.empty()) {
      // top priority size
      std::vector<symbol::DimExpr> dim_out;
      if (data_layout == DataLayout::kNCHW) {
        dim_out = {x.shape()[0],
                   x.shape()[1],
                   symbol::DimExpr{out_d},
                   symbol::DimExpr{out_h},
                   symbol::DimExpr{out_w}};
      } else {
        dim_out = {x.shape()[0],
                   symbol::DimExpr{out_d},
                   symbol::DimExpr{out_h},
                   symbol::DimExpr{out_w},
                   x.shape()[4]};
      }

      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(dim_out)};

      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);
      return true;
    }

    symbol::DimExpr out_d_tmp{0};
    symbol::DimExpr out_h_tmp{0};
    symbol::DimExpr out_w_tmp{0};
    const auto &next_sym = infer_context->GetNextSymName();
    out_d_tmp = symbol::DimExpr(next_sym);
    out_h_tmp = symbol::DimExpr(next_sym);
    out_w_tmp = symbol::DimExpr(next_sym);

    std::vector<symbol::DimExpr> dim_out;

    if (data_layout == DataLayout::kNCHW) {
      dim_out = {x.shape()[0], x.shape()[1], out_d_tmp, out_h_tmp, out_w_tmp};
    } else {
      dim_out = {x.shape()[0], out_d_tmp, out_h_tmp, out_w_tmp, x.shape()[4]};
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(dim_out)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;

  } else {
    PADDLE_THROW(phi::errors::Fatal("Input(X) dimension must be 3, 4 or 5!"));
  }

  return true;
}

bool BilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool CrossEntropyWithSoftmaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &input_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const symbol::ShapeOrDataDimExprs &index_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &input_dim = input_shape.shape();
  const auto &index_dim = index_shape.shape();
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  if (axis < 0) axis += input_shape.shape().size();
  bool soft_label =
      attributes.at("soft_label").dyn_cast<pir::BoolAttribute>().data();

  PADDLE_ENFORCE(!soft_label || input_dim.size() == index_dim.size(),
                 phi::errors::InvalidArgument(
                     "The input and index should have the same rank when "
                     "soft_label is true. But received input rank(%d) and "
                     "index rank(%d)",
                     input_dim.size(),
                     index_dim.size()));

  auto softmax_dim = index_dim;
  auto out_dim = index_dim;

  if (index_dim.size() == input_dim.size()) {
    if (!soft_label) {
      out_dim.erase(out_dim.begin() + axis);
    } else {
      out_dim[axis] = 1;
    }
    softmax_dim[axis] = input_dim[axis];
  } else {
    softmax_dim.insert(softmax_dim.begin() + axis, input_dim[axis]);
    if (soft_label) {
      out_dim.insert(out_dim.begin() + axis, 1);
    }
  }

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(softmax_dim));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(out_dim));

  return true;
}

bool CrossEntropyWithSoftmax_OpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return CrossEntropyWithSoftmaxOpInferSymbolicShape(op, infer_context);
}

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const auto &shape_data_list =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  size_t rank = shape_data_list[0].shape().size();

  int64_t axis = 0;

  auto SetShapeOrDataForAxis = [&](int axis_value) {
    std::vector<symbol::DimExpr> data{axis_value};
    symbol::TensorShapeOrDataDimExprs shape_or_data(
        std::vector<symbol::DimExpr>{}, data);
    infer_context->SetShapeOrDataForValue(op->operand_source(1), shape_or_data);
  };

  if (infer_context->HasShapeOrDataForValue(op->operand_source(1)) &&
      (infer_context->GetShapeOrDataForValue(op->operand_source(1)))
          .data()
          .has_value()) {
    const auto &axis_shape_or_data =
        infer_context->GetShapeOrDataForValue(op->operand_source(1));
    axis =
        static_cast<int>(axis_shape_or_data.data().value()[0].Get<int64_t>());
  } else {
    if (op->operand_source(1).defining_op() &&
        op->operand_source(1).defining_op()->isa<paddle::dialect::FullOp>()) {
      axis = op->operand_source(1)
                 .defining_op<paddle::dialect::FullOp>()
                 .attributes()
                 .at("value")
                 .dyn_cast<paddle::dialect::ScalarAttribute>()
                 .data()
                 .to<int64_t>();
      SetShapeOrDataForAxis(axis);
    } else {
      pir::Value res = op->result(0);
      infer_context->SetStaticShapeForValue(res);
      // update axis value
      auto res_shape = infer_context->GetShapeOrDataForValue(res);
      for (size_t i = 0; i < rank; ++i) {
        auto res_shape_dim = res_shape.shape()[i];
        auto shape_data_dim = shape_data_list[0].shape()[i];
        if (!res_shape_dim.isa<int64_t>()) break;
        if (!shape_data_dim.isa<int64_t>()) break;
        if (res_shape_dim.Get<int64_t>() > shape_data_dim.Get<int64_t>()) {
          SetShapeOrDataForAxis(i);
        }
      }
      return true;
    }
  }
  axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

  if (shape_data_list[0].data().has_value()) {
    if (rank == 1) {
      const auto &s_or_d =
          infer_context->GetShapeOrDataForValue(operand_source);
      ExprVec data = details::GetExprVecFromData(s_or_d);

      const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(shape, data)};
      pir::Value res = op->result(0);
      infer_context->SetShapeOrDataForValue(res, shape_data);

      return true;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          op->name() +
          " 's InferSymbolicShape can NOT deal with rank > 1 now."));
    }
    std::vector<symbol::DimExpr> data;
    data.reserve(shape_data_list.size());
    for (auto &data_elem : shape_data_list) {
      data.push_back(data_elem.data().value()[0]);
    }
    const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shape, data)};
    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);

    return true;
  }

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list[0].shape();
    for (size_t i = 0; i < rank; ++i) {
      if (i != static_cast<size_t>(axis)) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
        continue;
      }
      for (size_t j = 1; j < shape_data_list.size(); ++j) {
        out_dims[axis] = out_dims[axis] + shape_data_list[j].shape()[axis];
      }
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(1);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);

  const auto &out_shape = operand_shape_or_data.data().has_value()
                              ? operand_shape_or_data.data().value()
                              : operand_shape_or_data.shape();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool FlashAttnOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &q =
      infer_context->GetShapeOrDataForValue(operand_source);

  const symbol::ShapeOrDataDimExprs &k =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const symbol::ShapeOrDataDimExprs &v =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));

  PADDLE_ENFORCE_EQ(q.shape().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  infer_context->AddEqualCstr(q.shape()[0], k.shape()[0]);
  infer_context->AddEqualCstr(q.shape()[0], v.shape()[0]);
  infer_context->AddEqualCstr(k.shape()[1], v.shape()[1]);

  if (op->operand_source(4)) {
    const symbol::ShapeOrDataDimExprs &attn_mask =
        infer_context->GetShapeOrDataForValue(op->operand_source(4));
    infer_context->AddEqualCstr(attn_mask.shape()[0], q.shape()[0]);
    infer_context->AddEqualCstr(attn_mask.shape()[2], q.shape()[1]);
    infer_context->AddEqualCstr(attn_mask.shape()[3], k.shape()[1]);
  }

  std::vector<symbol::DimExpr> out_shape = q.shape();

  out_shape.back() = v.shape().back();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));

  // GPU has round for seqlen, but XPU has not. Here we align with the GPU
  // version.
  auto round_multiple = [](symbol::DimExpr x) {
    auto m = symbol::DimExpr{128};
    auto m_minus_one = symbol::DimExpr{127};
    return (x + m_minus_one) / m * m;
  };
  auto batch_size_expr = q.shape()[0];
  auto num_heads_expr = q.shape()[2];
  auto seqlen_q_rounded_expr = round_multiple(q.shape()[1]);
  auto seqlen_k_rounded_expr = round_multiple(k.shape()[1]);
  if (op->result(1)) {
    std::vector<symbol::DimExpr> softmax_shape{batch_size_expr,
                                               num_heads_expr,
                                               seqlen_q_rounded_expr,
                                               seqlen_k_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(1), symbol::TensorShapeOrDataDimExprs(softmax_shape));
  }
  if (op->result(2)) {
    std::vector<symbol::DimExpr> softmax_lse_shape{
        batch_size_expr, num_heads_expr, seqlen_q_rounded_expr};
    infer_context->SetShapeOrDataForValue(
        op->result(2), symbol::TensorShapeOrDataDimExprs(softmax_lse_shape));
  }
  if (op->result(3)) {
    std::vector<symbol::DimExpr> seed_offset_shape{symbol::DimExpr{2}};
    infer_context->SetShapeOrDataForValue(
        op->result(3), symbol::TensorShapeOrDataDimExprs(out_shape));
  }
  return true;
}

bool GroupNormOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  infer_context->SetShapeOrDataForValue(op->result(0), x_shape);

  const symbol::DimExpr &batch_size = x_shape.shape()[0];
  int groups = op->attribute<pir::Int32Attribute>("groups").data();
  symbol::TensorShapeOrDataDimExprs mean_shape(
      std::vector<symbol::DimExpr>{batch_size, groups});
  if (op->result(1)) {
    infer_context->SetShapeOrDataForValue(op->result(1), mean_shape);
  }
  if (op->result(2)) {
    infer_context->SetShapeOrDataForValue(op->result(2), mean_shape);
  }
  return true;
}

bool LinspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &num_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(2));
  const auto step = [&] {
    symbol::DimExpr expr;
    if (num_shape_or_data.data().has_value()) {
      expr = num_shape_or_data.data().value()[0];
    } else {
      expr = num_shape_or_data.shape()[0];
    }
    return expr;
  }();
  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims{step};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool LinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool LogspaceOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return LinspaceOpInferSymbolicShape(op, infer_context);
}

bool NearestInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool MemoryEfficientAttentionOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &q_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape();
  const auto &k_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1)).shape();
  const auto &v_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(2)).shape();
  PADDLE_ENFORCE_EQ(
      q_shape.size(),
      4,
      phi::errors::InvalidArgument("Query should be a 4-D tensor"
                                   "But received Query dimension(%d)",
                                   q_shape.size()));
  PADDLE_ENFORCE_EQ(
      k_shape.size(),
      4,
      phi::errors::InvalidArgument("Key should be a 4-D tensor"
                                   "But received Key dimension(%d)",
                                   k_shape.size()));
  PADDLE_ENFORCE_EQ(
      v_shape.size(),
      4,
      phi::errors::InvalidArgument("Value should be a 4-D tensor"
                                   "But received Value dimension(%d)",
                                   v_shape.size()));

  const auto &query_batch_size = q_shape[0];
  const auto &query_seq_length = q_shape[1];
  const auto &query_num_head = q_shape[2];
  const auto &query_head_size = q_shape[3];

  const auto &key_batch_size = k_shape[0];
  const auto &key_seq_length = k_shape[1];
  const auto &key_num_head = k_shape[2];
  const auto &key_head_size = k_shape[3];

  const auto &value_batch_size = v_shape[0];
  const auto &value_seq_length = v_shape[1];
  const auto &value_num_head = v_shape[2];
  const auto &value_head_size = v_shape[3];

  infer_context->AddEqualCstr(query_batch_size, key_batch_size);
  infer_context->AddEqualCstr(key_batch_size, value_batch_size);

  infer_context->AddEqualCstr(query_num_head, key_num_head);
  infer_context->AddEqualCstr(key_num_head, value_num_head);

  infer_context->AddEqualCstr(query_head_size, key_head_size);

  infer_context->AddEqualCstr(key_seq_length, value_seq_length);

  const std::vector<symbol::DimExpr> out_dims{
      query_batch_size, query_seq_length, query_num_head, value_head_size};
  const std::vector<symbol::DimExpr> logsumexp_dims{query_num_head,
                                                    query_batch_size};
  const std::vector<symbol::DimExpr> seed_and_offset_dims{2};

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(1), symbol::TensorShapeOrDataDimExprs(logsumexp_dims));
  infer_context->SetShapeOrDataForValue(
      op->result(2), symbol::TensorShapeOrDataDimExprs(seed_and_offset_dims));

  return true;
}

bool RoiAlignOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const auto &x = op->operand_source(0);
  const auto &boxes = op->operand_source(1);

  const auto &num_boxes =
      infer_context->GetShapeOrDataForValue(boxes).shape()[0];
  symbol::DimExpr channel_num =
      infer_context->GetShapeOrDataForValue(x).shape()[1];

  int32_t out_h = op->attribute<pir::Int32Attribute>("pooled_height").data();
  int32_t out_w = op->attribute<pir::Int32Attribute>("pooled_width").data();

  std::vector<symbol::DimExpr> out_dim = {num_boxes, channel_num, out_h, out_w};
  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_dim));
  return true;
}

bool MeshgridOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(op->operand_source(0))
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const symbol::ShapeOrDataDimExprs sym_shape_dim_exprs = [&] {
    symbol::TensorListShapeOrDataDimExprs shape_dim_exprs_list;
    std::vector<symbol::DimExpr> vec;

    for (auto &shape_data : shape_data_list) {
      if (shape_data.shape().size() == 0) {
        vec.emplace_back(1);
      } else {
        vec.emplace_back(shape_data.shape()[0]);
      }
    }

    auto shape_dim_exprs = symbol::TensorShapeOrDataDimExprs(vec);

    for (size_t i = 0; i < shape_data_list.size(); i++) {
      shape_dim_exprs_list.emplace_back(shape_dim_exprs);
    }

    return symbol::ShapeOrDataDimExprs(shape_dim_exprs_list);
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, sym_shape_dim_exprs);
  return true;
}

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      infer_context->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  int rank = shape_data_list[0].shape().size();
  if (axis < 0) axis += rank + 1;

  const symbol::ShapeOrDataDimExprs shape_data = [&] {
    std::vector<symbol::DimExpr> shape_dim_exprs;
    std::vector<symbol::DimExpr> data_dim_exprs;
    for (const auto &shape_data : shape_data_list) {
      if (shape_data.data().has_value() && axis == 0) {
        data_dim_exprs.emplace_back(shape_data.data().value()[0]);
      }
    }

    if (!data_dim_exprs.empty()) {
      shape_dim_exprs.emplace_back(
          static_cast<std::int64_t>(shape_data_list.size()));
    } else {
      for (int i = 0; i < rank; ++i) {
        details::BuildCstrEqForTensorListAlongAxis(
            infer_context, shape_data_list, i);
      }
      shape_dim_exprs.insert(shape_dim_exprs.begin() + axis,
                             static_cast<std::int64_t>(shape_data_list.size()));
    }
    if (data_dim_exprs.empty()) {
      return symbol::ShapeOrDataDimExprs(
          symbol::TensorShapeOrDataDimExprs(shape_dim_exprs));
    }
    return symbol::ShapeOrDataDimExprs(
        symbol::TensorShapeOrDataDimExprs(shape_dim_exprs, data_dim_exprs));
  }();

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool TrilinearInterpOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return BicubicInterpOpInferSymbolicShape(op, infer_context);
}

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      infer_context->GetShapeOrDataForValue(op->operand_source(0)));

  const std::vector<pir::Value> &operands = {op->operand_source(0),
                                             op->operand_source(1)};

  size_t rank = infer_context->GetShapeOrDataForValue(op->operand_source(0))
                    .shape()
                    .size();

  for (size_t i = 0; i < rank; ++i) {
    paddle::dialect::details::BuildCstrEqForTensorListAlongAxis(
        infer_context, operands, i);
  }

  return true;
}

bool Where_OpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return WhereOpInferSymbolicShape(op, infer_context);
}

}  // namespace paddle::dialect
