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

#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"

#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/utils.h"

namespace paddle::dialect {

template <>
std::vector<pir::Value> RelevantInputsImpl<AddGroupNormSiluOp>(
    pir::Operation* op) {
  auto concrete_op = op->dyn_cast<AddGroupNormSiluOp>();
  return {concrete_op.x(), concrete_op.residual()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<AddGroupNormSiluOp>(
    pir::Operation* op) {
  auto concrete_op = op->dyn_cast<AddGroupNormSiluOp>();
  return {concrete_op.y(), concrete_op.residual_out()};
}

template <>
common::DataLayout PreferLayoutImpl<AddGroupNormSiluOp>(pir::Operation* op) {
  // Note(bukejiyu): add_group_norm_silu only supports NHWC layout now.
  return common::DataLayout::NHWC;
}

template <>
common::DataLayout PreferLayoutImpl<Conv2dOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }

  auto concrete_op = op->dyn_cast<Conv2dOp>();
  if (auto in = concrete_op.input()) {
    if (auto in_type = in.type()) {
      if (in_type.isa<DenseTensorType>()) {
        if (auto tensor_type = in_type.dyn_cast<DenseTensorType>()) {
          if (tensor_type.dtype().isa<pir::Float16Type>()) {
            return common::DataLayout::NHWC;
          }
        }
      }
    }
  }

  return common::StringToDataLayout(data_format_attr.AsString());
}

template <>
common::DataLayout PreferLayoutImpl<Conv2dTransposeOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }

  auto concrete_op = op->dyn_cast<Conv2dTransposeOp>();
  if (auto in = concrete_op.x()) {
    if (auto in_type = in.type()) {
      if (in_type.isa<DenseTensorType>()) {
        if (auto tensor_type = in_type.dyn_cast<DenseTensorType>()) {
          if (tensor_type.dtype().isa<pir::Float16Type>()) {
            return common::DataLayout::NHWC;
          }
        }
      }
    }
  }

  return common::StringToDataLayout(data_format_attr.AsString());
}

template <>
bool CanBeModifiedImpl<Conv2dOp>(pir::Operation* op) {
  return false;
}

template <>
common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }

  auto original_layout =
      common::StringToDataLayout(data_format_attr.AsString());

  if (op->HasAttribute(kForceBackendAttr) &&
      op->attributes()
              .at(kForceBackendAttr)
              .dyn_cast<pir::StrAttribute>()
              .AsString() == "gpu") {
    return common::DataLayout::NHWC;
  }

  auto concrete_op = op->dyn_cast<FusedConv2dAddActOp>();
  if (auto in = concrete_op.input()) {
    if (auto in_type = in.type()) {
      if (in_type.isa<paddle::dialect::DenseTensorType>()) {
        if (auto tensor_type =
                in_type.dyn_cast<paddle::dialect::DenseTensorType>()) {
          if (!tensor_type.dtype().isa<pir::Float16Type>()) {
            return original_layout;
          }
        }
      }
    }
  }

  constexpr int CUDNN_ALIGNMENT = 8;

  if (auto filter = concrete_op.filter()) {
    if (auto filter_type = filter.type()) {
      if (filter_type.isa<DenseTensorType>()) {
        if (auto tensor_type = filter_type.dyn_cast<DenseTensorType>()) {
          if (tensor_type.dtype().isa<pir::Float16Type>()) {
            auto dims = tensor_type.dims();
            if (dims.size() == 4 && (dims[0] % CUDNN_ALIGNMENT == 0) &&
                (dims[1] % CUDNN_ALIGNMENT == 0)) {
              return common::DataLayout::NHWC;
            }
          }
        }
      }
    }
  }

  return original_layout;
}

template <>
bool CanBeModifiedImpl<FusedConv2dAddActOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }
  auto cur_layout = common::StringToDataLayout(data_format_attr.AsString());
  auto prefer_layout = PreferLayoutImpl<FusedConv2dAddActOp>(op);
  auto can_be_modified = cur_layout != prefer_layout;

  for (auto value : RelevantOutputsImpl<FusedConv2dAddActOp>(op)) {
    // TODO(lyk) if value was used in another block, we cannot rewrite this op
    for (auto it = value.use_begin(); it != value.use_end(); ++it) {
      if (it->owner()->GetParent() != op->GetParent()) {
        return false;
      }
    }
  }

  return can_be_modified;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<GroupNormOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<GroupNormOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<GroupNormOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<GroupNormOp>();
  return {concrete_op.y()};
}

template <>
std::vector<pir::Value> RelevantInputsImpl<ReshapeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<ReshapeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  return {concrete_op.out()};
}

template <>
bool CanBeModifiedImpl<ReshapeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  auto shape = concrete_op.shape();
  auto x = concrete_op.x();
  if (!x || !(x.defining_op()->isa<pir::ParameterOp>())) return false;
  if (!shape || !(shape.defining_op()->isa<FullIntArrayOp>())) return false;

  auto full_int_op = shape.defining_op()->dyn_cast<FullIntArrayOp>();
  auto value_attr =
      full_int_op.attribute("value").dyn_cast<pir::ArrayAttribute>();

  std::vector<int32_t> value_int32;
  for (size_t i = 0; i < value_attr.size(); ++i) {
    auto attr = value_attr.at(i);
    value_int32.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
  }
  return value_int32 == std::vector<int32_t>{1, -1, 1, 1};
}

template <>
void RewriteByLayoutImpl<SqueezeOp>(pir::Operation* op,
                                    common::DataLayout new_layout) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function", op->name()));
  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<SqueezeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<SqueezeOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<SqueezeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<SqueezeOp>();
  return {concrete_op.out()};
}

template <>
bool CanBeModifiedImpl<SqueezeOp>(pir::Operation* op) {
  return false;
}

template <>
bool CanBeModifiedImpl<AddOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<AddOp>();
  if (auto x = concrete_op.x(), y = concrete_op.y(); x && y) {
    if (auto xt = x.type(), yt = y.type(); xt && yt) {
      if (auto xdt = xt.dyn_cast<pir::DenseTensorType>(),
          ydt = yt.dyn_cast<pir::DenseTensorType>();
          xdt && ydt) {
        if (xdt.dims().size() != ydt.dims().size()) {
          return false;
        }
      }
    }
  }
  return true;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<ConcatOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ConcatOp>();
  return {concrete_op.x()};
}

template <>
void RewriteByLayoutImpl<ConcatOp>(pir::Operation* op,
                                   common::DataLayout new_layout) {
  // we must the value of concat axis, but this is an input
  // which is really hard to process.
  // here we handle the simple case like pd_op.full and throw
  // error in other cases.
  auto concrete_op = op->dyn_cast<ConcatOp>();
  auto axis = concrete_op.axis();
  if (!axis || !(axis.defining_op()->isa<FullOp>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Concat's axis must be processed when rewrite by layout."));
  }

  // TODO(lyk): we must assert this full int array op has one user which is
  // reshape
  auto axis_op = axis.defining_op()->dyn_cast<FullOp>();
  int axis_value =
      axis_op.attribute("value").dyn_cast<ScalarAttribute>().data().to<int>();

  // The layout of the tensor type is unreliable, since its always
  // NCHW, which is a default value. So we cannot deduct the new
  // axis by new layout, since we do not know if the layout changed.
  // So we simply assume the old layout must be NCHW, new layout must
  // be NHWC.
  PADDLE_ENFORCE_EQ(
      axis_value,
      1,
      common::errors::InvalidArgument(
          "Concat's axis was expected as 1, but got %d", axis_value));
  axis.defining_op()->set_attribute(
      "value",
      ScalarAttribute::get(pir::IrContext::Instance(), phi::Scalar(3)));

  // infer new meta for concat
  RewriteByInfermeta<ConcatOp>(op, new_layout);
}

template <>
void RewriteByLayoutImpl<ReshapeOp>(pir::Operation* op,
                                    common::DataLayout new_layout) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();

  auto shape = concrete_op.shape();
  if (!shape || !(shape.defining_op()->isa<FullIntArrayOp>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Reshape's shape must be processed when rewrite by layout."));
  }

  auto full_int_op = shape.defining_op()->dyn_cast<FullIntArrayOp>();
  auto value_attr =
      full_int_op.attribute("value").dyn_cast<pir::ArrayAttribute>();

  PADDLE_ENFORCE_EQ(value_attr.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Reshape's shape size was expected as 4, but got %d",
                        value_attr.size()));
  std::vector<pir::Attribute> new_value_attr;
  if (new_layout == common::DataLayout::kNHWC) {
    new_value_attr = std::vector<pir::Attribute>{
        value_attr[0], value_attr[2], value_attr[3], value_attr[1]};
  } else if (new_layout == common::DataLayout::kNCHW) {
    new_value_attr = std::vector<pir::Attribute>{
        value_attr[0], value_attr[3], value_attr[1], value_attr[2]};
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Layout only supports NHWC and NCHW in LayoutTransformationInterface"));
  }

  shape.defining_op()->set_attribute(
      "value",
      pir::ArrayAttribute::get(pir::IrContext::Instance(), new_value_attr));

  // infer new meta for concat
  RewriteByInfermeta<ReshapeOp>(op, new_layout);
}

template <>
void RewriteByLayoutImpl<ArgmaxOp>(pir::Operation* op,
                                   common::DataLayout new_layout) {
  auto concrete_op = op->dyn_cast<ArgmaxOp>();
  auto axis = concrete_op.axis();
  if (!axis || !(axis.defining_op()->isa<FullOp>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Argmax's axis must be processed when rewrite by layout."));
  }

  auto axis_op = axis.defining_op()->dyn_cast<FullOp>();
  int axis_value =
      axis_op.attribute("value").dyn_cast<ScalarAttribute>().data().to<int>();

  PADDLE_ENFORCE_EQ(
      axis_value,
      1,
      common::errors::InvalidArgument(
          "Argmax's axis was expected as 1, but got %d", axis_value));
  axis.defining_op()->set_attribute(
      "value",
      ScalarAttribute::get(pir::IrContext::Instance(), phi::Scalar(3)));

  // infer new meta for argmax
  RewriteByInfermeta<ArgmaxOp>(op, new_layout);
}

template <>
void RewriteByLayoutImpl<pir::CombineOp>(pir::Operation* op,
                                         common::DataLayout new_layout) {
  auto concrete_op = op->dyn_cast<pir::CombineOp>();
  auto out = concrete_op.out();
  if (!out) return;
  std::vector<pir::Type> new_out_type;
  for (auto v : op->operands_source()) {
    new_out_type.push_back(v.type());
  }
  auto new_out_type_v =
      pir::VectorType::get(pir::IrContext::Instance(), new_out_type);
  out.set_type(new_out_type_v);

  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<Pool2dOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<Pool2dOp>();
  return {concrete_op.x()};
}

template <>
common::DataLayout PreferLayoutImpl<Pool2dOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<Pool2dOp>();
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  auto origin_format = common::StringToDataLayout(data_format_attr.AsString());
  auto input =
      concrete_op.x().type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto full_int_op =
      concrete_op.kernel_size().defining_op()->dyn_cast<FullIntArrayOp>();
  if (!full_int_op || !input) return origin_format;

  // get pooling type
  std::string pool_type = concrete_op.attribute("pooling_type")
                              .dyn_cast<pir::StrAttribute>()
                              .AsString();

  // get input dims h, w, c
  int32_t h, w, c;
  if (origin_format == common::DataLayout::kNHWC) {
    h = input.dims().at(1);
    w = input.dims().at(2);
    c = input.dims().at(3);
  } else if (origin_format == common::DataLayout::kNCHW) {
    h = input.dims().at(2);
    w = input.dims().at(3);
    c = input.dims().at(1);
  } else {
    return origin_format;
  }

  // get kernel_size
  auto value_attr =
      full_int_op.attribute("value").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> kernel_size;
  for (size_t i = 0; i < value_attr.size(); ++i) {
    auto attr = value_attr.at(i);
    kernel_size.push_back(attr.dyn_cast<pir::Int64Attribute>().data());
  }

  auto PreferLayout = [](int c,
                         int h,
                         int w,
                         std::vector<int64_t> kernel_size,
                         std::string pool_type) {
    auto AllEqual = [&](const std::vector<int64_t>& vec) {
      return vec.empty() || std::all_of(vec.begin(), vec.end(), [&](int64_t x) {
               return x == vec[0];
             });
    };
    // TODO(liujinnan): need to test the prefer layout if kernel_size is not
    // aligned.
    if (!AllEqual(kernel_size)) return common::DataLayout::kNCHW;

    int k = kernel_size[0];
    // kernel size is all 1, prefer NCHW.
    if (k == 1) return common::DataLayout::kNCHW;

    if (pool_type == "max") {
      if (h * w <= 64 * 64) {
        if (k <= 3) return common::DataLayout::kNHWC;
        return common::DataLayout::kNCHW;
      } else {
        if (c <= 16) {
          if (k <= 5) return common::DataLayout::kNHWC;
          return common::DataLayout::kNCHW;
        }
        // when c > 16, all kernel_size return NHWC
        return common::DataLayout::kNHWC;
      }
    } else if (pool_type == "avg") {
      if (h * w <= 64 * 64) {
        if (c <= 16) {
          if (k < 7)
            return common::DataLayout::kNCHW;
          else
            return common::DataLayout::kNHWC;
        } else if (c > 16 && c <= 32) {
          if (k <= 7)
            return common::DataLayout::kNHWC;
          else
            return common::DataLayout::kNCHW;
        } else if (c > 32 && c <= 64) {
          if (k < 5)
            return common::DataLayout::kNCHW;
          else
            return common::DataLayout::kNHWC;
        } else if (c > 64 && c <= 128) {
          if (k < 7)
            return common::DataLayout::kNHWC;
          else
            return common::DataLayout::kNCHW;
        }
        // when c > 128, all kernel_size return NHWC
        return common::DataLayout::kNHWC;
      } else {
        if (c < 64) {
          return common::DataLayout::kNHWC;
        } else {
          if (k < 7)
            return common::DataLayout::kNHWC;
          else
            return common::DataLayout::kNCHW;
        }
      }
    }
    return common::DataLayout::kNCHW;
  };
  return PreferLayout(c, h, w, kernel_size, pool_type);
}

}  // namespace paddle::dialect
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LayoutTransformationInterface)
