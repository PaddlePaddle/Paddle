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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/phi/common/complex.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/parameter.h"

namespace paddle::dialect {

pir::Value builtin_combine(const std::vector<pir::Value>& x) {
  // Auto Parallel condition
  ProcessMeshAttribute op_mesh;
  if (HasDistInput(x, &op_mesh)) {
    CvtAllInputsToDist(x, op_mesh);
  }
  auto combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  return combine_op.out();
}

std::vector<pir::Value> builtin_split(const pir::Value& x) {
  auto split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(x);
  return split_op.outputs();
}

std::vector<pir::Value> add_n_grad(const std::vector<pir::Value>& inputs,
                                   const pir::Value& out_grad) {
  std::vector<pir::Value> inputs_grad;
  for (size_t i = 0; i < inputs.size(); i++) {
    paddle::dialect::ScaleOp scale_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleOp>(
            out_grad, 1.0, 0.0, true);
    inputs_grad.push_back(scale_op.result(0));
  }
  return inputs_grad;
}

pir::Value full(const std::vector<int64_t>& shape,
                double real,
                double imag,
                phi::DataType dtype,
                const phi::Place& place) {
  CheckDataType(dtype, "dtype", "full");
  if (dtype == phi::DataType::COMPLEX64) {
    dtype = phi::DataType::FLOAT32;
  } else {
    dtype = phi::DataType::FLOAT64;
  }
  pir::Value real_tmp = full(shape, real, dtype, place);
  pir::Value imag_tmp = full(shape, imag, dtype, place);
  return paddle::dialect::complex(real_tmp, imag_tmp);
}

pir::Value zeros_like(const pir::Value& x,
                      const phi::DataType dtype,
                      const Place& place) {
  return paddle::dialect::full_like(x, 0, dtype, place);
}

pir::Value parameter(const std::string& name) {
  pir::Parameter* param = ApiBuilder::Instance().GetParameter(name);
  pir::ParameterOp parameter_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::ParameterOp>(
          name, param->type());
  return parameter_op.result(0);
}

void set_parameter(const pir::Value& parameter, const std::string& name) {
  pir::Parameter* param = ApiBuilder::Instance().GetParameter(name);
  if (param) {
    PADDLE_ENFORCE_EQ(param->type(),
                      parameter.type(),
                      common::errors::InvalidArgument(
                          "Duplicate parameter %s with different type.", name));
  } else {
    std::unique_ptr<pir::Parameter> param_new(
        new pir::Parameter(nullptr, 0, parameter.type()));
    ApiBuilder::Instance().SetParameter(name, std::move(param_new));
    ApiBuilder::Instance().GetBuilder()->Build<pir::SetParameterOp>(parameter,
                                                                    name);
  }
}

void update_parameter(const pir::Value& parameter, const std::string& name) {
  pir::Parameter* param = ApiBuilder::Instance().GetParameter(name);
  PADDLE_ENFORCE_NOT_NULL(param,
                          common::errors::InvalidArgument(
                              "Parameter %s not exist, can not update.", name));
  std::unique_ptr<pir::Parameter> param_new(
      new pir::Parameter(nullptr, 0, parameter.type()));
  ApiBuilder::Instance().SetParameter(name, std::move(param_new));
  ApiBuilder::Instance().GetBuilder()->Build<pir::SetParameterOp>(parameter,
                                                                  name);
}

void shadow_output(const pir::Value& persist_value, const std::string& name) {
  auto& builder = ApiBuilder::Instance().GetBuilder();
  auto op = builder->Build<pir::ShadowOutputOp>(persist_value, name);
  if (auto dist_interface =
          persist_value.type().dyn_cast<DistTypeInterface>()) {
    op->set_attribute(
        kAttrOpDistAttr,
        OperationDistAttribute::get(builder->ir_context(),
                                    dist_interface.process_mesh_attr(),
                                    {dist_interface.tensor_dist_attr()},
                                    {}));
  }
}

pir::Value embedding_grad(const pir::Value& x,
                          const pir::Value& weight,
                          const pir::Value& out_grad,
                          int64_t padding_idx,
                          bool sparse) {
  if (weight.type().isa<paddle::dialect::DenseTensorType>()) {
    if (sparse) {
      auto embedding_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::EmbeddingSparseGradOp>(
                  x, weight, out_grad, padding_idx);
      return embedding_grad_op.weight_grad();
    } else {
      auto embedding_grad_op = ApiBuilder::Instance()
                                   .GetBuilder()
                                   ->Build<paddle::dialect::EmbeddingGradOp>(
                                       x, weight, out_grad, padding_idx);
      return embedding_grad_op.weight_grad();
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Now we do not support sparse weight embedding_grad."));
  }
}

pir::Value split_with_num_grad(const std::vector<pir::Value>& out_grad,
                               int axis) {
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::Value split_with_num_grad(const std::vector<pir::Value>& out_grad,
                               const pir::Value& axis) {
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::Value ones(const std::vector<int64_t>& shape,
                phi::DataType dtype,
                const Place& place) {
  return paddle::dialect::full(shape, 1, dtype, place);
}

pir::Value ones_like(pir::Value x_, phi::DataType dtype, const Place& place) {
  return paddle::dialect::full_like(x_, 1, dtype, place);
}

pir::Value zeros(const std::vector<int64_t>& shape,
                 phi::DataType dtype,
                 const Place& place) {
  return paddle::dialect::full(shape, 0, dtype, place);
}

pir::Value create_array(phi::DataType dtype) {
  auto create_array_op = ApiBuilder::Instance()
                             .GetBuilder()
                             ->Build<paddle::dialect::CreateArrayOp>(dtype);
  return create_array_op.out();
}

pir::Value create_array_like(pir::Value input, float value) {
  auto create_array_like_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::CreateArrayLikeOp>(input, value);
  return create_array_like_op.out();
}

pir::Value array_length(pir::Value x) {
  auto array_length_op = ApiBuilder::Instance()
                             .GetBuilder()
                             ->Build<paddle::dialect::ArrayLengthOp>(x);
  return array_length_op.out();
}

pir::Value array_read(pir::Value array, pir::Value i) {
  auto array_read_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArrayReadOp>(
          array, i);
  return array_read_op.out();
}

pir::Value fetch(pir::Value value, std::string name, int col) {
  auto fetch_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::FetchOp>(
          value, name, col);
  return fetch_op.out();
}

pir::Value array_write_(pir::Value array, pir::Value x, pir::Value i) {
  auto array_write_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ArrayWrite_Op>(array, x, i);
  return array_write_op.out();
}

std::tuple<pir::Value, pir::Value> array_to_tensor(pir::Value x,
                                                   int axis,
                                                   bool use_stack) {
  auto array_to_tensor =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ArrayToTensorOp>(x, axis, use_stack);
  return std::make_tuple(array_to_tensor.result(0), array_to_tensor.result(1));
}

pir::Value tensor_to_array(pir::Value x,
                           pir::Value out_grad,
                           int axis,
                           bool use_stack) {
  auto tensor_to_array = ApiBuilder::Instance()
                             .GetBuilder()
                             ->Build<paddle::dialect::TensorToArrayOp>(
                                 x, out_grad, axis, use_stack);
  return tensor_to_array.result(0);
}

pir::Value add_n_array(const std::vector<pir::Value>& inputs) {
  auto inputs_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs);
  paddle::dialect::AddNArrayOp add_n_array_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AddNArrayOp>(
          inputs_combine_op.out());
  return add_n_array_op.result(0);
}

pir::Value slice_array(pir::Value input, pir::Value starts, pir::Value ends) {
  auto op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SliceArrayOp>(
          input, starts, ends);
  return op.result(0);
}

pir::Value slice_array_dense(pir::Value input, pir::Value starts) {
  auto op = ApiBuilder::Instance()
                .GetBuilder()
                ->Build<paddle::dialect::SliceArrayDenseOp>(input, starts);
  return op.result(0);
}

pir::Value assign(const pir::Value& x) {
  CheckValueDataType(x, "x", "assign");
  if (x.type().isa<paddle::dialect::DenseTensorType>()) {
    paddle::dialect::AssignOp assign_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::AssignOp>(
            x);
    return assign_op.result(0);
  } else if (x.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    paddle::dialect::AssignArrayOp assign_array_op =
        ApiBuilder::Instance()
            .GetBuilder()
            ->Build<paddle::dialect::AssignArrayOp>(x);
    return assign_array_op.result(0);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently, assign only supports DenseTensorType and "
        "DenseTensorArrayType."));
  }
}

std::tuple<pir::Value, pir::Value> fused_gemm_epilogue(pir::Value x,
                                                       pir::Value y,
                                                       pir::Value bias,
                                                       bool trans_x,
                                                       bool trans_y,
                                                       std::string activation) {
  // AMP Logic
  if (egr::Controller::Instance().GetCurrentAmpAttrs()->GetAmpLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP: fused_gemm_epilogue";
    auto op_name = phi::TransToFluidOpName("fused_gemm_epilogue");
    paddle::small_vector<std::vector<pir::Value>, egr::kSlotSmallVectorSize>
        amp_values_vector = {{x}, {y}, {bias}};
    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_values_vector);
    auto new_x =
        paddle::imperative::AmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y =
        paddle::imperative::AmpAutoCast("y", y, amp_dst_dtype, op_name);
    auto new_bias =
        paddle::imperative::AmpAutoCast("bias", bias, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return paddle::dialect::fused_gemm_epilogue(
          new_x, new_y, new_bias, trans_x, trans_y, activation);
    }
  }

  // Type Promotion Logic
  VLOG(5) << " No Type Promotion for fused_gemm_epilogue api. ";
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::AttributeMap attribute_map = {
      {"trans_x", pir::BoolAttribute::get(ctx, trans_x)},
      {"trans_y", pir::BoolAttribute::get(ctx, trans_y)},
      {"activation", pir::StrAttribute::get(ctx, activation)}};
  auto fused_gemm_epilogue_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedGemmEpilogueOp>(
              x, y, bias, attribute_map);
  if (!egr::Controller::Instance().HasGrad()) {
    SetStopGradient(fused_gemm_epilogue_op.result(0),
                    fused_gemm_epilogue_op.result(1));
  }
  return std::make_tuple(fused_gemm_epilogue_op.result(0),
                         fused_gemm_epilogue_op.result(1));
}

std::tuple<pir::Value, pir::Value, pir::Value> fused_gemm_epilogue_grad(
    pir::Value x,
    pir::Value y,
    paddle::optional<pir::Value> reserve_space,
    pir::Value out_grad,
    bool trans_x,
    bool trans_y,
    std::string activation_grad) {
  // AMP Logic
  if (egr::Controller::Instance().GetCurrentAmpAttrs()->GetAmpLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP: fused_gemm_epilogue_grad";
    auto op_name = phi::TransToFluidOpName("fused_gemm_epilogue_grad");
    paddle::small_vector<std::vector<pir::Value>, egr::kSlotSmallVectorSize>
        amp_values_vector = {{x}, {y}};
    if (reserve_space) {
      amp_values_vector.push_back({*reserve_space});
    }

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_values_vector);
    auto new_x =
        paddle::imperative::AmpAutoCast("x", x, amp_dst_dtype, op_name);
    auto new_y =
        paddle::imperative::AmpAutoCast("y", y, amp_dst_dtype, op_name);
    auto new_reserve_space = paddle::imperative::AmpAutoCast(
        "reserve_space", reserve_space, amp_dst_dtype, op_name);
    auto new_out_grad = paddle::imperative::AmpAutoCast(
        "out_grad", out_grad, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return paddle::dialect::fused_gemm_epilogue_grad(new_x,
                                                       new_y,
                                                       new_reserve_space,
                                                       new_out_grad,
                                                       trans_x,
                                                       trans_y,
                                                       activation_grad);
    }
  }

  // Type Promotion Logic
  VLOG(5) << " No Type Promotion for fused_gemm_epilogue api. ";
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::AttributeMap attribute_map = {
      {"trans_x", pir::BoolAttribute::get(ctx, trans_x)},
      {"trans_y", pir::BoolAttribute::get(ctx, trans_y)},
      {"activation_grad", pir::StrAttribute::get(ctx, activation_grad)}};
  paddle::optional<pir::Value> optional_reserve_space;
  if (!reserve_space) {
    optional_reserve_space = paddle::make_optional<pir::Value>(pir::Value());
  } else {
    optional_reserve_space = reserve_space;
  }
  auto fused_gemm_epilogue_grad_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::FusedGemmEpilogueGradOp>(
              x, y, optional_reserve_space.get(), out_grad, attribute_map);
  if (!egr::Controller::Instance().HasGrad()) {
    SetStopGradient(fused_gemm_epilogue_grad_op.result(0),
                    fused_gemm_epilogue_grad_op.result(1),
                    fused_gemm_epilogue_grad_op.result(2));
  }
  return std::make_tuple(fused_gemm_epilogue_grad_op.result(0),
                         fused_gemm_epilogue_grad_op.result(1),
                         fused_gemm_epilogue_grad_op.result(2));
}

pir::Value array_pop(pir::Value input, int index) {
  if (input.type().isa<paddle::dialect::DenseTensorArrayType>()) {
    paddle::dialect::ArrayPopOp array_pop_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArrayPopOp>(
            input, index);
    return array_pop_op.result(1);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "pop only supports DenseTensorArrayType."));
  }
}

std::vector<pir::Value> tensorrt_engine(
    const std::vector<pir::Value>& inputs,
    paddle::platform::EngineParams trt_params,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::vector<std::vector<int64_t>> outputs_shape,
    std::vector<phi::DataType> outputs_dtype,
    const std::string& converter_debug_info) {
  auto x =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(inputs).out();
  paddle::dialect::TensorRTEngineOp tensorrt_engine_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::TensorRTEngineOp>(x,
                                                     trt_params,
                                                     input_names,
                                                     output_names,
                                                     outputs_shape,
                                                     outputs_dtype,
                                                     converter_debug_info);
  auto out_split_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>(
      tensorrt_engine_op.result(0));
  return out_split_op.outputs();
}

pir::Operation* share_var(const std::vector<pir::Value>& x) {
  return ApiBuilder::Instance().GetBuilder()->Build<ShareVarOp>(x);
}

}  // namespace paddle::dialect
