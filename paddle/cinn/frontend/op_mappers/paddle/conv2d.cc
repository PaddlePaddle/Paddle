// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void Conv2dOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Input").size(),
      1UL,
      phi::errors::InvalidArgument("Input size of conv2d op should be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Filter").size(),
      1UL,
      phi::errors::InvalidArgument("Filter size of conv2d op should be 1."));
  auto y_name = op_desc.Input("Filter").front();

  PADDLE_ENFORCE_EQ(
      op_desc.Output("Output").size(),
      1UL,
      phi::errors::InvalidArgument("Output size of conv2d op should be 1."));
  auto out_name = op_desc.Output("Output").front();

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");
  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  PADDLE_ENFORCE_EQ(
      x->shape.size(),
      4UL,
      phi::errors::InvalidArgument(
          "CINN conv2d operator's x only support 4-D tensor now."));
  PADDLE_ENFORCE_EQ(
      y->shape.size(),
      4,
      phi::errors::InvalidArgument(
          "CINN conv2d operator's y only support 4-D tensor now."));
  if (data_format == "NHWC") {
    // the weight in paddle always be NCHW, but cudnn need the same as input,
    // transpose before
    y = ctx.Builder()->Transpose(y, {0, 2, 3, 1});
  }
  auto out = ctx.Builder()->Conv2d(x,
                                   y,
                                   strides,
                                   paddings,
                                   dilations,
                                   groups,
                                   data_format,
                                   padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DepthwiseConv2dOpMapperImpl(common::UnknownArch,
                                 const paddle::cpp::OpDesc& op_desc,
                                 const OpMapperContext& ctx) {
  LOG(FATAL) << "NotImplemented.";
}

void DepthwiseConv2dOpMapperImpl(common::X86Arch,
                                 const paddle::cpp::OpDesc& op_desc,
                                 const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Input size of depthwise_conv2d op should be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Filter size of depthwise_conv2d op should be 1."));

  auto y_name = op_desc.Input("Filter").front();

  PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Output size of depthwise_conv2d op should be 1."));
  auto out_name = op_desc.Output("Output").front();

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");
  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  Variable out;
  out = ctx.Builder()->Conv2d(x,
                              y,
                              strides,
                              paddings,
                              dilations,
                              groups,
                              data_format,
                              padding_algorithm);
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DepthwiseConv2dOpMapperImpl(common::ARMArch,
                                 const paddle::cpp::OpDesc& op_desc,
                                 const OpMapperContext& ctx) {
  LOG(FATAL) << "NotImplemented.";
}

void DepthwiseConv2dOpMapperImpl(common::NVGPUArch,
                                 const paddle::cpp::OpDesc& op_desc,
                                 const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Input size of depthwise_conv2d op should be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Filter size of depthwise_conv2d op should be 1."));
  auto y_name = op_desc.Input("Filter").front();

  PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Output size of depthwise_conv2d op should be 1."));
  auto out_name = op_desc.Output("Output").front();

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");
  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  Variable out;
  out = ctx.Builder()->DepthwiseConv2d(x,
                                       y,
                                       strides,
                                       paddings,
                                       dilations,
                                       groups,
                                       data_format,
                                       padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DepthwiseConv2dOpMapperByArch(common::Arch arch,
                                   const paddle::cpp::OpDesc& op_desc,
                                   const OpMapperContext& ctx) {
  return std::visit(
      [&](const auto& impl) {
        return DepthwiseConv2dOpMapperImpl(impl, op_desc, ctx);
      },
      arch.variant());
}

void DepthwiseConv2dOpMapper(const paddle::cpp::OpDesc& op_desc,
                             const OpMapperContext& ctx) {
  return DepthwiseConv2dOpMapperByArch(ctx.Target().arch, op_desc, ctx);
}

void Conv2dGradOpMapper(const paddle::cpp::OpDesc& op_desc,
                        const OpMapperContext& ctx) {
  // get dy
  PADDLE_ENFORCE_EQ(op_desc.Input(paddle::GradVarName("Output")).size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Input size of conv2d_grad op should be 1."));
  auto dy_name = op_desc.Input(paddle::GradVarName("Output")).front();

  // get intput input,filter
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Input size of conv2d_grad op should be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Filter size of conv2d_grad op should be 1."));
  auto w_name = op_desc.Input("Filter").front();

  // get d_x
  std::string dx_name;
  bool has_dx = !op_desc.Output(paddle::GradVarName("Input")).empty();
  if (has_dx) {
    PADDLE_ENFORCE_EQ(op_desc.Output(paddle::GradVarName("Input")).size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "Output size of conv2d_grad op should be 1."));
    dx_name = op_desc.Output(paddle::GradVarName("Input")).front();
  }
  // get d_filter
  PADDLE_ENFORCE_EQ(op_desc.Output(paddle::GradVarName("Filter")).size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Output size of conv2d_grad op should be 1."));
  auto dw_name = op_desc.Output(paddle::GradVarName("Filter")).front();

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");

  auto dy = ctx.GetVar(dy_name);
  auto x = ctx.GetVar(x_name);
  auto weight = ctx.GetVar(w_name);

  PADDLE_ENFORCE_EQ(
      x->shape.size(),
      4UL,
      phi::errors::InvalidArgument(
          "CINN conv2d_grad operator's x only support 4-D tensor now."));
  PADDLE_ENFORCE_EQ(
      dy->shape.size(),
      4UL,
      phi::errors::InvalidArgument(
          "CINN conv2d_grad operator's dy only support 4-D tensor now."));
  PADDLE_ENFORCE_EQ(
      weight->shape.size(),
      4UL,
      phi::errors::InvalidArgument(
          "CINN conv2d_grad operator's weight only support 4-D tensor now."));

  if (data_format == "NHWC") {
    // the weight in paddle always be NCHW, but cudnn need the same as input,
    // transpose before
    weight = ctx.Builder()->Transpose(weight, {0, 2, 3, 1});
  }

  if (has_dx) {
    // create backward data
    auto dx = ctx.Builder()->Conv(weight,
                                  dy,
                                  strides,
                                  paddings,
                                  dilations,
                                  groups,
                                  "backward_data",
                                  data_format,
                                  padding_algorithm,
                                  x->shape);

    ctx.AddVar(dx_name, dx);
    ctx.AddVarModelToProgram(dx_name, dx->id);
  }

  // create backward filter
  auto dw = ctx.Builder()->Conv(x,
                                dy,
                                strides,
                                paddings,
                                dilations,
                                groups,
                                "backward_filter",
                                data_format,
                                padding_algorithm,
                                weight->shape);

  if (data_format == "NHWC") {
    // the weight in paddle always be NCHW, but cudnn need the same as input,
    // transpose back
    dw = ctx.Builder()->Transpose(dw, {0, 3, 1, 2});
  }

  ctx.AddVar(dw_name, dw);
  ctx.AddVarModelToProgram(dw_name, dw->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_conv2d) {
  CINN_REGISTER_OP_MAPPER(conv2d,
                          cinn::frontend::paddle_mappers::Conv2dOpMapper)
  CINN_REGISTER_OP_MAPPER(
      depthwise_conv2d, cinn::frontend::paddle_mappers::DepthwiseConv2dOpMapper)

#ifdef CINN_WITH_CUDNN
  CINN_REGISTER_OP_MAPPER(conv2d_grad,
                          cinn::frontend::paddle_mappers::Conv2dGradOpMapper)
#endif
  return true;
}
