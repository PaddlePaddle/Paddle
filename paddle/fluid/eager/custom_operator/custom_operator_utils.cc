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

#include "paddle/fluid/eager/custom_operator/custom_operator_utils.h"

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/flags.h"
#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

namespace egr {

using Tensor = paddle::Tensor;

static std::vector<std::vector<phi::DDim>> RunDefaultInferShapeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  std::vector<std::vector<phi::DDim>> result;
  if (inplace_map.empty()) {  // general case, assure single input and output
    PADDLE_ENFORCE_EQ(
        inputs.size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferShapeFn. "
            "At this time, the input shape will be directly set to "
            "the output shape.\n"
            "Please set the InferShapeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferShapeFn. "
            "At this time, the input shape will be directly set to "
            "the output shape.\n"
            "Please set the InferShapeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));

    VLOG(3) << "Custom Operator: Default InferShape - share ddim.";
    result.push_back({ctx.InputAt(0).dims()});
  } else {  // inplace case
    PADDLE_ENFORCE_EQ(
        inplace_map.size(),
        outputs.size(),
        phi::errors::Unavailable(
            "Your custom operator uses `SetInplaceMap` without setting the "
            "InferShapeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap` size = %d. Please check `SetInplaceMap` again or set "
            "the InferShapeFn of custom operator by "
            "`.SetInferShapeFn(PD_INFER_SHAPE(...)`)",
            outputs.size(),
            inplace_map.size()));
    for (size_t i = 0; i < ctx.InputRange().size(); ++i) {
      if (paddle::framework::detail::IsDuplicableVar(inputs[i])) {
        std::vector<phi::DDim> shapes;
        auto duplicable_input_pair = ctx.InputRangeAt(i);
        for (size_t j = duplicable_input_pair.first;
             j < duplicable_input_pair.second;
             j++) {
          shapes.push_back(ctx.InputAt(j).dims());
        }
        result.emplace_back(std::move(shapes));
      } else {
        auto duplicable_input_pair = ctx.InputRangeAt(i);
        result.push_back({ctx.InputAt(duplicable_input_pair.first).dims()});
      }
    }
  }
  return result;
}

static std::vector<std::vector<phi::DDim>> RunDefaultGradInferShapeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const std::vector<std::string>& grad_op_inputs,
    const std::vector<std::string>& grad_op_outputs,
    bool is_double_grad) {
  std::vector<std::vector<phi::DDim>> result;
  // 1. if forward input exists, gradient's shape is same with forward
  // input
  // default
  //    [Suitable for most situations]
  // 2. if forward input not exists, and only contains one grad input and
  // output,
  //    use grad input shape as grad output shape
  //    [Suitable for the situation that forward input is not used as
  //    backward input]
  for (auto& out_name : grad_op_outputs) {
    auto fwd_name = paddle::framework::detail::NoGrad(out_name, is_double_grad);
    if (paddle::framework::detail::IsDuplicableVar(fwd_name)) {
      // Duplicable forward var must as backward input
      auto iter =
          std::find(grad_op_inputs.begin(), grad_op_inputs.end(), fwd_name);
      PADDLE_ENFORCE_NE(
          iter,
          grad_op_inputs.end(),
          phi::errors::NotFound("Custom grad operator should have the forward "
                                "input(%s) as backward input",
                                fwd_name));
      auto pair = ctx.InputRangeAt(iter - grad_op_inputs.begin());
      std::vector<phi::DDim> tmp;
      for (size_t i = pair.first; i < pair.second; ++i) {
        tmp.emplace_back(ctx.InputAt(i).dims());
      }
      result.emplace_back(std::move(tmp));
    } else {
      if (grad_op_inputs.size() == grad_op_outputs.size()) {
        result.push_back({ctx.InputAt(0).dims()});
      } else {
        auto iter =
            std::find(grad_op_inputs.begin(), grad_op_inputs.end(), fwd_name);
        PADDLE_ENFORCE_NE(
            iter,
            grad_op_inputs.end(),
            phi::errors::NotFound("Custom grad operator should have the "
                                  "forward input(%s) as backward input",
                                  fwd_name));
        auto pair = ctx.InputRangeAt(iter - grad_op_inputs.begin());
        result.push_back({ctx.InputAt(pair.first).dims()});
      }
    }
  }
  return result;
}

static std::vector<std::vector<phi::DDim>> RunInferShapeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const paddle::InferShapeFunc& func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  std::vector<std::vector<phi::DDim>> result;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;

  VLOG(3) << "Custom Operator: InferShape - get input ddim.";
  for (size_t i = 0; i < ctx.InputRange().size(); ++i) {
    const auto& input_pair = ctx.InputRangeAt(i);
    if (input_pair.first == input_pair.second - 1) {
      input_shapes.emplace_back(
          std::move(ctx.InputAt(input_pair.first).shape()));
    } else {
      std::vector<std::vector<int64_t>> shapes;
      for (size_t j = input_pair.first; j < input_pair.second; j++) {
        shapes.push_back(std::move(ctx.InputAt(j).shape()));
      }
      vec_input_shapes.emplace_back(std::move(shapes));
    }
  }

  VLOG(3) << "Custom Operator: InferShape - calc output ddim.";
  auto output_shapes = func(input_shapes, vec_input_shapes, ctx.Attrs());
  if (inplace_map.empty()) {
    PADDLE_ENFORCE_EQ(outputs.size(),
                      output_shapes.size(),
                      phi::errors::InvalidArgument(
                          "Your custom operator has set the InferShapeFn. "
                          "However, `Outputs` size = %d does not match the "
                          "returned vector size of InferShapeFn = %d. Please "
                          "check InferShapeFn again.",
                          outputs.size(),
                          output_shapes.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        output_shapes.size() + inplace_map.size(),
        phi::errors::InvalidArgument(
            "Your custom operator uses `SetInplaceMap` and sets the "
            "InferShapeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap size + InferShapeFn output size` = %d. Please check "
            "InplaceMap and InferShapeFn again",
            outputs.size(),
            output_shapes.size() + inplace_map.size()));
  }

  VLOG(3)
      << "Custom Operator: InferShape - set output ddim: inplace_map.size() = "
      << inplace_map.size()
      << ", output_shapes.size() = " << output_shapes.size();
  size_t output_shape_idx = 0;
  auto inplace_reverse_map = ctx.GetInplaceReverseIndexMap();
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (paddle::framework::detail::IsDuplicableVar(outputs[i])) {
      PADDLE_ENFORCE(
          inplace_reverse_map.find(i) != inplace_reverse_map.end(),
          phi::errors::InvalidArgument(
              "Custom operator only supports `paddle::Vec(...)` inputs and "
              "cannot support `paddle::Vec(...)` output without setting "
              "InplaceMap. If you have to use `paddle::Vec(...)` output, "
              "please indicate it by setting InplaceMap manully."));
      std::vector<phi::DDim> shapes;
      auto duplicable_input_pair = ctx.InputRangeAt(inplace_reverse_map[i]);
      for (size_t j = duplicable_input_pair.first;
           j < duplicable_input_pair.second;
           j++) {
        shapes.push_back(ctx.InputAt(j).dims());
      }
      result.emplace_back(std::move(shapes));
    } else {
      if (inplace_reverse_map.find(i) != inplace_reverse_map.end()) {
        auto duplicable_input_pair = ctx.InputRangeAt(inplace_reverse_map[i]);
        result.push_back({ctx.InputAt(duplicable_input_pair.first).dims()});
      } else {
        result.push_back(
            {common::make_ddim(output_shapes[output_shape_idx++])});
      }
    }
  }
  return result;
}

static std::vector<std::vector<phi::DataType>> RunDefaultInferDtypeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  std::vector<std::vector<phi::DataType>> result;
  if (inplace_map.empty()) {  // general case, assure single input and output
    PADDLE_ENFORCE_EQ(
        inputs.size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));

    VLOG(3) << "Custom Operator: InferDtype - share dtype.";
    result.push_back({ctx.InputAt(0).dtype()});
  } else {  // inplace case
    PADDLE_ENFORCE_EQ(
        inplace_map.size(),
        outputs.size(),
        phi::errors::Unavailable(
            "Your custom operator uses `SetInplaceMap` without setting the "
            "InferDtypeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap` size = %d. Please check `SetInplaceMap` again or set "
            "the InferDtypeFn of custom operator by "
            "`.SetInferDtypeFn(PD_INFER_DTYPE(...))`",
            outputs.size(),
            inplace_map.size()));
    for (size_t i = 0; i < ctx.InputRange().size(); ++i) {
      if (paddle::framework::detail::IsDuplicableVar(inputs[i])) {
        std::vector<phi::DataType> shapes;
        auto duplicable_input_pair = ctx.InputRangeAt(i);
        for (size_t j = duplicable_input_pair.first;
             j < duplicable_input_pair.second;
             j++) {
          shapes.push_back(ctx.InputAt(j).dtype());
        }
        result.emplace_back(std::move(shapes));
      } else {
        auto duplicable_input_pair = ctx.InputRangeAt(i);
        result.push_back({ctx.InputAt(duplicable_input_pair.first).dtype()});
      }
    }
  }
  return result;
}

static std::vector<std::vector<phi::DataType>> RunDefaultGradInferDtypeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const std::vector<std::string>& grad_op_inputs,
    const std::vector<std::string>& grad_op_outputs,
    bool is_double_grad) {
  std::vector<std::vector<phi::DataType>> result;
  for (auto& out_name : grad_op_outputs) {
    auto fwd_name = paddle::framework::detail::NoGrad(out_name, is_double_grad);
    if (paddle::framework::detail::IsDuplicableVar(fwd_name)) {
      // Duplicable forward var must as backward input
      auto iter =
          std::find(grad_op_inputs.begin(), grad_op_inputs.end(), fwd_name);
      PADDLE_ENFORCE_NE(
          iter,
          grad_op_inputs.end(),
          phi::errors::NotFound("Custom grad operator should have the forward "
                                "input(%s) as backward input",
                                fwd_name));
      auto pair = ctx.InputRangeAt(iter - grad_op_inputs.begin());
      std::vector<phi::DataType> tmp;
      for (size_t i = pair.first; i < pair.second; ++i) {
        tmp.emplace_back(ctx.InputAt(i).dtype());
      }
      result.emplace_back(std::move(tmp));
    } else {
      if (grad_op_inputs.size() == grad_op_outputs.size()) {
        result.push_back({ctx.InputAt(0).dtype()});
      } else {
        auto iter =
            std::find(grad_op_inputs.begin(), grad_op_inputs.end(), fwd_name);
        PADDLE_ENFORCE_NE(
            iter,
            grad_op_inputs.end(),
            phi::errors::NotFound("Custom grad operator should have the "
                                  "forward input(%s) as backward input",
                                  fwd_name));
        auto pair = ctx.InputRangeAt(iter - grad_op_inputs.begin());
        result.push_back({ctx.InputAt(pair.first).dtype()});
      }
    }
  }
  return result;
}

static std::vector<std::vector<phi::DataType>> RunInferDtypeFunc(
    const paddle::CustomOpKernelContext& ctx,
    const paddle::InferDtypeFunc& func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  std::vector<std::vector<phi::DataType>> result;
  std::vector<phi::DataType> input_dtypes;
  std::vector<std::vector<phi::DataType>> vec_input_dtypes;

  VLOG(3) << "Custom Operator: InferDtype - get input dtype.";
  for (size_t i = 0; i < ctx.InputRange().size(); ++i) {
    const auto& input_pair = ctx.InputRangeAt(i);
    if (input_pair.first == input_pair.second - 1) {
      input_dtypes.emplace_back(
          std::move(ctx.InputAt(input_pair.first).dtype()));
    } else {
      std::vector<phi::DataType> dtypes;
      for (size_t j = input_pair.first; j < input_pair.second; j++) {
        dtypes.emplace_back(ctx.InputAt(j).dtype());
      }
      vec_input_dtypes.emplace_back(std::move(dtypes));
    }
  }

  VLOG(3) << "Custom Operator: InferDtype - infer output dtype.";
  auto output_dtypes = func(input_dtypes, vec_input_dtypes, ctx.Attrs());
  if (inplace_map.empty()) {
    PADDLE_ENFORCE_EQ(outputs.size(),
                      output_dtypes.size(),
                      phi::errors::InvalidArgument(
                          "Your custom operator has set the InferDtypeFn. "
                          "However, `Outputs` size = %d does not match the "
                          "returned vector size of InferDtypeFn = %d. Please "
                          "check InferDtypeFn again.",
                          outputs.size(),
                          output_dtypes.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        output_dtypes.size() + inplace_map.size(),
        phi::errors::InvalidArgument(
            "Your custom operator uses `SetInplaceMap` and sets the "
            "InferDtypeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap size + InferDtypeFn output size` = %d. Please check "
            "InplaceMap and InferDtypeFn again",
            outputs.size(),
            output_dtypes.size() + inplace_map.size()));
  }

  VLOG(3)
      << "Custom Operator: InferDtype - set output dtype: inplace_map.size() = "
      << inplace_map.size()
      << ", output_dtypes.size() = " << output_dtypes.size();
  size_t output_dtype_idx = 0;
  auto inplace_reverse_map = ctx.GetInplaceReverseIndexMap();
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (paddle::framework::detail::IsDuplicableVar(outputs[i])) {
      PADDLE_ENFORCE(
          inplace_reverse_map.find(i) != inplace_reverse_map.end(),
          phi::errors::InvalidArgument(
              "Custom operator only supports `paddle::Vec(...)` inputs and "
              "cannot support `paddle::Vec(...)` output without setting "
              "InplaceMap. If you have to use `paddle::Vec(...)` output, "
              "please indicate it by setting InplaceMap manully."));
      std::vector<phi::DataType> dtypes;
      auto duplicable_input_pair = ctx.InputRangeAt(inplace_reverse_map[i]);
      for (size_t j = duplicable_input_pair.first;
           j < duplicable_input_pair.second;
           j++) {
        dtypes.push_back(ctx.InputAt(j).dtype());
      }
      result.emplace_back(std::move(dtypes));
    } else {
      if (inplace_reverse_map.find(i) != inplace_reverse_map.end()) {
        auto duplicable_input_pair = ctx.InputRangeAt(inplace_reverse_map[i]);
        result.push_back({ctx.InputAt(duplicable_input_pair.first).dtype()});
      } else {
        result.push_back({output_dtypes[output_dtype_idx++]});
      }
    }
  }
  return result;
}

#ifdef PADDLE_WITH_DISTRIBUTE
paddle::Tensor BuildEmptyDistPaddleTensor(
    const phi::distributed::ProcessMesh& process_mesh,
    const phi::DDim& dims,
    phi::DataType dtype) {
  paddle::Tensor empty_tensor;
  phi::DenseTensorMeta meta;
  meta.dims = dims;
  meta.dtype = dtype;

  auto dist_attr = phi::distributed::TensorDistAttr(common::vectorize(dims));
  dist_attr.set_process_mesh(process_mesh);

  auto dist_t = std::make_shared<phi::distributed::DistTensor>(
      std::make_shared<phi::DenseTensor>(
          std::make_shared<phi::Allocation>(
              nullptr, 0, phi::distributed::GetDefaultPlace()),
          meta),
      dist_attr);
  empty_tensor.set_impl(dist_t);
  empty_tensor.set_autograd_meta(std::make_shared<egr::AutogradMeta>());
  return empty_tensor;
}
#endif

#ifdef PADDLE_WITH_DISTRIBUTE

phi::distributed::SpmdInfo RunInferSpmdFn(
    const paddle::OpMetaInfo& op_info,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    paddle::CustomOpKernelContext& ctx) {  // NOLINT
  auto& infer_spmd_func = paddle::OpMetaInfoHelper::GetInferSpmdFn(op_info);
  if (infer_spmd_func == nullptr) {
    // default rule
    std::vector<phi::distributed::DistMetaTensor> meta_dist_inputs;
    auto all_inputs = ctx.AllMutableInput();
    for (auto& t : *all_inputs) {
      phi::distributed::DistMetaTensor meta_dist_input;
      if (t.impl().get() && t.is_initialized()) {
        meta_dist_input =
            paddle::experimental::MakeDistMetaTensor(*(t.impl().get()));
      }
      meta_dist_inputs.push_back(meta_dist_input);
    }
    auto spmd_info_tmp =
        phi::distributed::VariadicReplicatedInferSpmdDynamic(meta_dist_inputs);
    // flatten input
    phi::distributed::SpmdInfo spmd_info;
    auto dist_attrs = PADDLE_GET(std::vector<phi::distributed::TensorDistAttr>,
                                 spmd_info_tmp.first[0]);
    for (auto& e : dist_attrs) {
      spmd_info.first.push_back(std::move(e));
    }
    return spmd_info;
  }
  std::vector<paddle::CustomSpmdInferTensorArg> tensor_inputs;
  size_t input_size = inputs.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto& in_name = inputs[i];
    if (paddle::framework::detail::IsDuplicableVar(in_name)) {
      std::vector<phi::distributed::DistMetaTensor> meta_tensors;
      auto& range = ctx.InputRangeAt(i);
      for (size_t j = range.first; j < range.second; ++j) {
        auto& t = ctx.InputAt(j);
        phi::distributed::DistMetaTensor meta_tensor;
        if (t.impl().get() && t.is_initialized()) {
          meta_tensor =
              paddle::experimental::MakeDistMetaTensor(*(t.impl().get()));
        }
        meta_tensors.emplace_back(std::move(meta_tensor));
      }
      tensor_inputs.emplace_back(std::move(meta_tensors));
    } else {
      auto& range = ctx.InputRangeAt(i);
      auto& t = ctx.InputAt(range.first);
      phi::distributed::DistMetaTensor meta_tensor;
      if (t.impl().get() && t.is_initialized()) {
        meta_tensor =
            paddle::experimental::MakeDistMetaTensor(*(t.impl().get()));
      }
      tensor_inputs.emplace_back(std::move(meta_tensor));
    }
  }
  const std::vector<paddle::CustomSpmdInferAttrArg>& attrs = ctx.Attrs();
  auto spmd_info_tmp = infer_spmd_func(tensor_inputs, attrs);
  // flatten input
  phi::distributed::SpmdInfo spmd_info;
  for (auto& e : spmd_info_tmp.first) {
    if (paddle::holds_alternative<phi::distributed::TensorDistAttr>(e)) {
      spmd_info.first.push_back(
          std::move(PADDLE_GET(phi::distributed::TensorDistAttr, e)));
    } else {
      for (auto& ee :
           PADDLE_GET(std::vector<phi::distributed::TensorDistAttr>, e)) {
        spmd_info.first.push_back(std::move(ee));
      }
    }
  }

  // flatten output
  for (auto& e : spmd_info_tmp.second) {
    if (paddle::holds_alternative<phi::distributed::TensorDistAttr>(e)) {
      spmd_info.second.push_back(
          std::move(PADDLE_GET(phi::distributed::TensorDistAttr, e)));
    } else {
      for (auto& ee :
           PADDLE_GET(std::vector<phi::distributed::TensorDistAttr>, e)) {
        spmd_info.second.push_back(std::move(ee));
      }
    }
  }

  return spmd_info;
}

std::vector<std::vector<phi::DDim>> RunInferShapeFn(
    const paddle::OpMetaInfo& op_info,
    bool is_forward,
    bool is_double_grad,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map,
    paddle::CustomOpKernelContext& ctx) {  // NOLINT
  auto& infer_shape_func = paddle::OpMetaInfoHelper::GetInferShapeFn(op_info);

  std::vector<std::vector<phi::DDim>> out_dims;
  if (infer_shape_func) {
    out_dims =
        RunInferShapeFunc(ctx, infer_shape_func, inputs, outputs, inplace_map);
  } else {
    if (is_forward) {
      out_dims = RunDefaultInferShapeFunc(ctx, inputs, outputs, inplace_map);
    } else {
      out_dims =
          RunDefaultGradInferShapeFunc(ctx, inputs, outputs, is_double_grad);
    }
  }

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      ctx.OutputRange().size(),
      phi::errors::InvalidArgument(
          "Custome op infer_shape return size should be %d, but got %d.",
          ctx.OutputRange().size(),
          out_dims.size()));

  return out_dims;
}

std::vector<std::vector<phi::DataType>> RunInferDtypeFn(
    const paddle::OpMetaInfo& op_info,
    bool is_forward,
    bool is_double_grad,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map,
    paddle::CustomOpKernelContext& ctx) {  // NOLINT

  auto& infer_dtype_func = paddle::OpMetaInfoHelper::GetInferDtypeFn(op_info);
  std::vector<std::vector<phi::DataType>> out_dtypes;
  if (infer_dtype_func) {
    out_dtypes =
        RunInferDtypeFunc(ctx, infer_dtype_func, inputs, outputs, inplace_map);
  } else {
    if (is_forward) {
      out_dtypes = RunDefaultInferDtypeFunc(ctx, inputs, outputs, inplace_map);
    } else {
      out_dtypes =
          RunDefaultGradInferDtypeFunc(ctx, inputs, outputs, is_double_grad);
    }
  }
  PADDLE_ENFORCE_EQ(
      out_dtypes.size(),
      ctx.OutputRange().size(),
      phi::errors::InvalidArgument(
          "Custome op infer_dtype return size should be %d, but got %d.",
          ctx.OutputRange().size(),
          out_dtypes.size()));
  return out_dtypes;
}

std::
    tuple<bool, bool, phi::distributed::ProcessMesh, phi::distributed::SpmdInfo>
    PrepareCtxForAutoParallel(
        const paddle::OpMetaInfo& op_info,
        bool is_forward,
        bool is_double_grad,
        paddle::CustomOpKernelContext& ctx,  // NOLINT
        std::vector<std::shared_ptr<phi::distributed::DistTensor>>&
            dist_inputs,                        // NOLINT
        std::vector<phi::DDim>& output_dims) {  // NOLINT
  bool run_auto_parallel = false;
  bool rank_is_in_current_mesh = true;
  phi::distributed::ProcessMesh current_process_mesh;
  phi::distributed::SpmdInfo spmd_info;

  const auto& inputs = paddle::OpMetaInfoHelper::GetInputs(op_info);
  const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(op_info);
  const auto& inplace_map = paddle::OpMetaInfoHelper::GetInplaceMap(op_info);

  std::vector<Tensor>* all_inputs = ctx.AllMutableInput();
  std::vector<Tensor> x;
  for (auto& t : *all_inputs) {
    if (t.impl().get()) {
      x.emplace_back(t);
    }
  }

  run_auto_parallel = paddle::experimental::AllInputsAreDistTensor(x);
  rank_is_in_current_mesh = true;
  if (run_auto_parallel) {
    auto mesh =
        std::static_pointer_cast<phi::distributed::DistTensor>(x.at(0).impl())
            ->dist_attr()
            .process_mesh();
    rank_is_in_current_mesh = phi::distributed::IsCurRankInMesh(mesh);

    spmd_info = RunInferSpmdFn(op_info, inputs, outputs, ctx);

    current_process_mesh =
        paddle::holds_alternative<phi::distributed::TensorDistAttr>(
            spmd_info.first[0])
            ? paddle::get<0>(spmd_info.first[0]).process_mesh()
            : paddle::get<1>(spmd_info.first[0]).at(0).process_mesh();

    std::vector<std::vector<phi::DDim>> out_dims = RunInferShapeFn(
        op_info, is_forward, is_double_grad, inputs, outputs, inplace_map, ctx);

    std::vector<std::vector<phi::DataType>> out_dtypes = RunInferDtypeFn(
        op_info, is_forward, is_double_grad, inputs, outputs, inplace_map, ctx);

    if (rank_is_in_current_mesh) {
      auto* dev_ctx = phi::DeviceContextPool::Instance().Get(x.at(0).place());
      for (size_t i = 0; i < x.size(); ++i) {
        auto dist_input_i = paddle::experimental::ReshardApiInputToKernelInput(
            dev_ctx, x[i], spmd_info.first[i]);
        all_inputs->at(i).set_impl(
            std::make_shared<phi::DenseTensor>(dist_input_i->value()));
        dist_inputs.emplace_back(dist_input_i);
      }
    }

    for (size_t i = 0; i < out_dims.size(); ++i) {
      const auto& out_dim = out_dims.at(i);
      const auto& out_dtype = out_dtypes.at(i);
      const auto& pair = ctx.OutputRangeAt(i);
      PADDLE_ENFORCE_EQ(
          out_dim.size(),
          pair.second - pair.first,
          phi::errors::InvalidArgument("custome op infer_shape result[%d]'s "
                                       "size should be %d, but got %d.",
                                       i,
                                       pair.second - pair.first,
                                       out_dim.size()));
      PADDLE_ENFORCE_EQ(
          out_dtype.size(),
          pair.second - pair.first,
          phi::errors::InvalidArgument("custome op infer_shape result[%d]'s "
                                       "size should be %d, but got %d.",
                                       i,
                                       pair.second - pair.first,
                                       out_dtype.size()));

      if (out_dim.size() == 1) {
        output_dims.emplace_back(out_dim[0]);
        if (!rank_is_in_current_mesh) {
          *(ctx.MutableOutputAt(pair.first)) = BuildEmptyDistPaddleTensor(
              current_process_mesh, out_dim[0], out_dtype[0]);
        }
      } else {
        for (size_t j = pair.first; j < pair.second; j++) {
          output_dims.emplace_back(out_dim[j]);
          if (!rank_is_in_current_mesh) {
            *(ctx.MutableOutputAt(j)) = BuildEmptyDistPaddleTensor(
                current_process_mesh, out_dim[j], out_dtype[j]);
          }
        }
      }
    }
  }
  return {run_auto_parallel,
          rank_is_in_current_mesh,
          current_process_mesh,
          spmd_info};
}
#endif

#ifdef PADDLE_WITH_DISTRIBUTE
void TransCtxTensorsToDistTensors(
    paddle::CustomOpKernelContext& ctx,  // NOLINT
    bool run_auto_parallel,
    const phi::distributed::ProcessMesh& current_process_mesh,
    const phi::distributed::SpmdInfo& spmd_info,
    std::vector<std::shared_ptr<phi::distributed::DistTensor>>&
        dist_inputs,                        // NOLINT
    std::vector<phi::DDim>& output_dims) {  // NOLINT
  if (run_auto_parallel) {
    std::vector<Tensor>* output_all = ctx.AllMutableOutput();
    for (size_t i = 0; i < output_all->size(); ++i) {
      auto& tensor = output_all->at(i);
      phi::distributed::TensorDistAttr dist_attr;
      if (!spmd_info.second.empty()) {
        dist_attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr,
                                     spmd_info.second[i]);
      } else {
        std::vector<int64_t> shape = common::vectorize(output_dims[i]);
        dist_attr.set_default_dims_mapping(shape);
        dist_attr.set_process_mesh(current_process_mesh);
      }
      auto dist_t = std::make_shared<phi::distributed::DistTensor>(
          std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()),
          output_dims[i],
          dist_attr);
      tensor.set_impl(dist_t);
    }
    std::vector<Tensor>* input_all = ctx.AllMutableInput();
    for (size_t i = 0; i < input_all->size(); ++i) {
      auto& tensor = input_all->at(i);
      phi::distributed::TensorDistAttr dist_attr;
      phi::DDim global_dims;

      if (i < dist_inputs.size()) {
        auto& dist_input = dist_inputs.at(i);
        global_dims = dist_input->dims();
        dist_attr = dist_input->dist_attr();
      } else {
        dist_attr = PADDLE_GET_CONST(phi::distributed::TensorDistAttr,
                                     spmd_info.first[i]);
        global_dims = tensor.dims();
      }
      auto dist_t = std::make_shared<phi::distributed::DistTensor>(
          std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()),
          global_dims,
          dist_attr);
      tensor.set_impl(dist_t);
    }
  }
}
#endif

void run_custom_op_impl(const paddle::OpMetaInfo& op_info,
                        bool is_forward,
                        bool is_double_grad,
                        paddle::CustomOpKernelContext& ctx) {  // NOLINT
  const auto& inputs = paddle::OpMetaInfoHelper::GetInputs(op_info);
  const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(op_info);
  const auto& inplace_map = paddle::OpMetaInfoHelper::GetInplaceMap(op_info);
  ctx.ConstructInplaceIndex(inputs, outputs, inplace_map);

#ifdef PADDLE_WITH_DISTRIBUTE
  // for output
  std::vector<std::shared_ptr<phi::distributed::DistTensor>> dist_inputs;
  std::vector<phi::DDim> output_dims;
  auto result = PrepareCtxForAutoParallel(
      op_info, is_forward, is_double_grad, ctx, dist_inputs, output_dims);
  bool run_auto_parallel = std::get<0>(result);
  bool rank_is_in_current_mesh = std::get<1>(result);
  phi::distributed::ProcessMesh current_process_mesh = std::get<2>(result);
  auto& spmd_info = std::get<3>(result);
  if (!rank_is_in_current_mesh) {
    return;
  }
#endif

  std::vector<Tensor>* all_inputs = ctx.AllMutableInput();
  for (size_t i = 0; i < all_inputs->size(); ++i) {
    auto& tensor = all_inputs->at(i);
    if (tensor.initialized() && tensor.is_dense_tensor() &&
        !std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
             ->meta()
             .is_contiguous()) {
      tensor.set_impl(std::make_shared<phi::DenseTensor>(
          std::move(paddle::experimental::Trans2Contiguous(
              *(std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()))))));
    }
  }

  // handle inplace map
  ctx.UpdatePlainOutputs(inputs, outputs, inplace_map);
  VLOG(7) << "Begin run Kernel of Custom Op";
  (*paddle::OpMetaInfoHelper::GetKernelFn(op_info))(&ctx);
  ctx.AssignInplaceOutputs();

#ifdef PADDLE_WITH_DISTRIBUTE
  TransCtxTensorsToDistTensors(ctx,
                               run_auto_parallel,
                               current_process_mesh,
                               spmd_info,
                               dist_inputs,
                               output_dims);
#endif
}

}  // namespace egr
