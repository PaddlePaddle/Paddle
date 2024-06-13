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

#include "paddle/cinn/frontend/paddle_model_to_program.h"

#include <algorithm>

#include "paddle/cinn/frontend/paddle/framework.pb.h"
#include "paddle/cinn/frontend/paddle/model_parser.h"
#include "paddle/cinn/frontend/paddle/pb/program_desc.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/common/enforce.h"

PD_DECLARE_double(cinn_infer_model_version);

namespace cinn {
namespace frontend {
using utils::Join;
using utils::TransValidVarName;

void MoveData(float* data, int i, int M, int N) {
  float temp = data[i];
  int cur = i;  // current data index
  int pre = (cur % M) * N + cur / M;
  while (pre != i) {
    data[cur] = data[pre];
    cur = pre;
    pre = (cur % M) * N + cur / M;
  }
  data[cur] = temp;
}

void TransposeData(float* data, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    int next = (i % N) * M + i / N;
    while (next > i)  // next < 1 implies duplicate
      next = (next % N) * M + next / N;
    if (next == i)  // process current ring
      MoveData(data, i, M, N);
  }
}

void ReverseHWData(float* data, std::vector<int> shape) {
  PADDLE_ENFORCE_EQ(shape.size(),
                    4UL,
                    phi::errors::InvalidArgument(
                        "The shape size of the data is not equal to 4! Please "
                        "check."));
  for (int i = 0; i < shape[0] * shape[1]; i++) {
    int num = shape[2] * shape[3];
    std::reverse(data + (i * num), data + (i * num + num));
  }
}

void PaddleModelToProgram::AddOpMapper_feed() {
  op_mappers_["feed"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto outs = op_desc.Output("Out");
    PADDLE_ENFORCE_EQ(outs.size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the feed op is not equal to 1! "
                          "Please check."));
    VLOG(2) << "Model get feed [" << outs[0] << "]";
    CHECK(input_shape_map_.count(outs[0]));
    auto input_shape = input_shape_map_[outs[0]];
    auto input = net_builder_->CreateInput(Float(32), input_shape, outs[0]);
    AddVar(outs[0], input);
  };
}

void PaddleModelToProgram::AddOpMapper_fetch() {
  op_mappers_["fetch"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the fetch op is not equal to 1! "
                          "Please check."));
    auto output_names = op_desc.Input("X");
    for (auto& output_name : output_names) {
      VLOG(2) << "fetch model output: [" << output_name << "]";
      fetch_names_.insert(utils::TransValidVarName(output_name));
    }
  };
}

void PaddleModelToProgram::AddOpMapper_scale() {
  op_mappers_["scale"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the scale op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    float scale{};
    float bias{};
    if (op_desc.HasAttr("scale")) {  // the old model format
      scale = op_desc.GetAttr<float>("scale");
    } else {  // the newly refactored format
      // load scale tensor
      PADDLE_ENFORCE_EQ(op_desc.Input("ScaleTensor").size(),
                        1UL,
                        phi::errors::InvalidArgument(
                            "The input size of the ScaleTensor is not equal to "
                            "1! Please check."));
      auto* scale_tensor_var =
          scope_->FindVar(op_desc.Input("ScaleTensor").front());
      CHECK(scale_tensor_var) << "No scale tensor found in the scope";
      auto& scale_tensor =
          absl::get<hlir::framework::Tensor>(*scale_tensor_var);
      scale = scale_tensor->mutable_data<float>(
          cinn::common::DefaultHostTarget())[0];
    }
    if (op_desc.HasAttr("bias")) {  // the old model format
      bias = op_desc.GetAttr<float>("bias");
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Didn't find [bias] attr in Scale operator!!"));
    }
    absl::flat_hash_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    auto out = net_builder_->Scale(x, scale, bias);
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the scale op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_mul() {
  op_mappers_["mul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the mul op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the mul op is not equal to 1! "
                          "Please check."));
    auto y_name = op_desc.Input("Y").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    TransposeVar(TransValidVarName(y_name));
    auto y = GetVar(utils::TransValidVarName(y_name));
    int x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
    int y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");

    VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
    VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    VLOG(4) << "y shape: " << utils::Join(y->shape, ",");

    const auto& out =
        net_builder_->Mul(x, y, x_num_col_dims, y_num_col_dims, true);

    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the mul op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_matmul() {
  op_mappers_["matmul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the matmul op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the matmul op is not equal to 1! "
                          "Please check."));
    auto y_name = op_desc.Input("Y").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    auto y = GetVar(utils::TransValidVarName(y_name));
    bool trans_a = op_desc.GetAttr<bool>("transpose_X");
    bool trans_b = op_desc.GetAttr<bool>("transpose_Y");
    float alpha = op_desc.GetAttr<float>("alpha");
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    VLOG(4) << "y shape: " << utils::Join(y->shape, ",");
    auto out = net_builder_->Matmul(x, y, trans_a, trans_b, alpha);
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the matmul op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_reshape2() {
  op_mappers_["reshape2"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("X").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The input size of the reshape2 op is not equal to 1! "
            "Please check."));
    auto x_name = op_desc.Input("X").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    std::vector<int> shape = op_desc.GetAttr<std::vector<int>>("shape");
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    auto out = net_builder_->Reshape(x, shape);
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The output size of the reshape2 op is not equal to 1! "
            "Please check."));
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_concat() {
  op_mappers_["concat"] = [&](const paddle::cpp::OpDesc& op_desc) {
    int input_size = op_desc.Input("X").size();
    PADDLE_ENFORCE_GE(input_size,
                      2UL,
                      phi::errors::InvalidArgument(
                          "The input size of the concat op is less than 2! "
                          "Please check."));
    std::vector<Variable> input_vars;
    for (int i = 0; i < input_size; i++) {
      auto name = op_desc.Input("X")[i];
      input_vars.push_back(GetVar(utils::TransValidVarName(name)));
    }
    int axis = op_desc.GetAttr<int>("axis");
    VLOG(4) << "axis in op concat is : " << axis;
    auto out = net_builder_->Concat(input_vars, axis);
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the concat op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_assign() {
  op_mappers_["assign"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the assign op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the assign op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Identity(x);
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_fill_constant() {
  op_mappers_["fill_constant"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The output size of the fill_constant op is not equal "
            "to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();

    CHECK(op_desc.HasAttr("shape"));
    auto shape = op_desc.GetAttr<std::vector<int64_t>>("shape");
    std::vector<int> shapes;
    for (size_t i = 0; i < shape.size(); i++) {
      PADDLE_ENFORCE_LE(shape[i],
                        std::numeric_limits<int32_t>::max(),
                        phi::errors::InvalidArgument(
                            "The shape size of the data is too large! Please "
                            "check."));
      shapes.push_back(static_cast<int>(shape[i]));
    }
    CHECK(op_desc.HasAttr("dtype"));
    auto dtype = op_desc.GetAttr<int>("dtype");
    CHECK(op_desc.HasAttr("value"));
    auto value = op_desc.GetAttr<float>("value");
    CHECK(op_desc.HasAttr("str_value"));
    auto str_value = op_desc.GetAttr<std::string>("str_value");
    CHECK(op_desc.HasAttr("force_cpu"));
    auto force_cpu = op_desc.GetAttr<bool>("force_cpu");

    Variable out;
    switch (dtype) {
#define DO(desc, type)                                                         \
  case ::cinn::frontend::paddle::proto::VarType::Type::VarType_Type_##desc:    \
    out =                                                                      \
        net_builder_->FillConstant<type>(shapes, value, str_value, force_cpu); \
    break;
      DO(BOOL, bool);
      DO(FP32, float);
      DO(INT32, int);
#undef DO
      default:
        std::stringstream ss;
        ss << "unknown data type " << dtype;
        PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
    }
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_transpose2() {
  op_mappers_["transpose2"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the transpose2 op is not equal to "
                          "1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The output size of the transpose2 op is not equal to "
            "1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    CHECK(op_desc.HasAttr("axis"));
    auto axis = op_desc.GetAttr<std::vector<int>>("axis");

    auto out = net_builder_->Transpose(x, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_exp() {
  op_mappers_["exp"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the exp op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the exp op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));

    auto out = net_builder_->Exp(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu() {
  op_mappers_["relu"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the relu op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the relu op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Relu(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_softmax() {
  op_mappers_["softmax"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the softmax op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The output size of the softmax op is not equal to 1! "
            "Please check."));
    auto out_name = op_desc.Output("Out").front();

    int axis = 0;
    if (op_desc.HasAttr("axis")) {
      axis = op_desc.GetAttr<int>("axis");
    } else {
      axis = static_cast<int>(-1);
    }
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Softmax(x, {axis});
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_add() {
  op_mappers_["elementwise_add"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_add op is not "
                          "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_add op is not "
                          "equal to 1! Please check."));
    auto y_name = op_desc.Input("Y").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the elementwise_add op is not "
                          "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    int axis = op_desc.GetAttr<int>("axis");

    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto out = net_builder_->Add(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_mul() {
  op_mappers_["elementwise_mul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_mul op is not "
                          "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_mul op is not "
                          "equal to 1! Please check."));
    auto y_name = op_desc.Input("Y").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the elementwise_mul op is not "
                          "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    int axis = op_desc.GetAttr<int>("axis");

    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto out = net_builder_->Multiply(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_div() {
  op_mappers_["elementwise_div"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_div op is not "
                          "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_div op is not "
                          "equal to 1! Please check."));
    auto y_name = op_desc.Input("Y").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the elementwise_div op is not "
                          "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    CHECK(op_desc.HasAttr("axis"));
    int axis = op_desc.GetAttr<int>("axis");

    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto out = net_builder_->Divide(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_elementwise_sub() {
  op_mappers_["elementwise_sub"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_sub op is not "
                          "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the elementwise_sub op is not "
                          "equal to 1! Please check."));
    auto y_name = op_desc.Input("Y").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the elementwise_sub op is not "
                          "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    CHECK(op_desc.HasAttr("axis"));
    int axis = op_desc.GetAttr<int>("axis");

    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto out = net_builder_->Subtract(x, y, axis);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu6() {
  op_mappers_["relu6"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the relu6 op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the relu6 op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();

    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Relu6(x);
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

template <typename T>
Variable AddOpMapperDepthwiseConv2dImpl(common::UnknownArch,
                                        T* net_builder,
                                        const paddle::cpp::OpDesc& op_desc,
                                        const Variable& x,
                                        const Variable& y) {
  LOG(FATAL) << "NotImplemented.";
}

template <typename T>
Variable AddOpMapperDepthwiseConv2dImpl(common::X86Arch,
                                        T* net_builder,
                                        const paddle::cpp::OpDesc& op_desc,
                                        const Variable& x,
                                        const Variable& y) {
  CHECK(op_desc.HasAttr("paddings"));
  auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  CHECK(op_desc.HasAttr("strides"));
  auto strides = op_desc.GetAttr<std::vector<int>>("strides");
  CHECK(op_desc.HasAttr("dilations"));
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  CHECK(op_desc.HasAttr("groups"));
  auto groups = op_desc.GetAttr<int>("groups");
  CHECK(op_desc.HasAttr("data_format"));
  std::string data_format = op_desc.GetAttr<std::string>("data_format");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }
  return net_builder->Conv2d(
      x, y, strides, paddings, dilations, groups, data_format);
}

template <typename T>
Variable AddOpMapperDepthwiseConv2dImpl(common::ARMArch,
                                        T* net_builder,
                                        const paddle::cpp::OpDesc& op_desc,
                                        const Variable& x,
                                        const Variable& y) {
  LOG(FATAL) << "NotImplemented.";
}

template <typename T>
Variable AddOpMapperDepthwiseConv2dImpl(common::NVGPUArch,
                                        T* net_builder,
                                        const paddle::cpp::OpDesc& op_desc,
                                        const Variable& x,
                                        const Variable& y) {
  CHECK(op_desc.HasAttr("paddings"));
  auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  CHECK(op_desc.HasAttr("strides"));
  auto strides = op_desc.GetAttr<std::vector<int>>("strides");
  CHECK(op_desc.HasAttr("dilations"));
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  CHECK(op_desc.HasAttr("groups"));
  auto groups = op_desc.GetAttr<int>("groups");
  CHECK(op_desc.HasAttr("data_format"));
  std::string data_format = op_desc.GetAttr<std::string>("data_format");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }
  Variable out;
  return net_builder->DepthwiseConv2d(
      x, y, strides, paddings, dilations, groups, data_format);
}

template <typename T>
Variable AddOpMapperDepthwiseConv2dImpl(common::HygonDCUArchHIP,
                                        T* net_builder,
                                        const paddle::cpp::OpDesc& op_desc,
                                        const Variable& x,
                                        const Variable& y) {
  // old code
  CINN_NOT_IMPLEMENTED
}

template <typename T>
Variable AddOpMapperDepthwiseConv2d(common::Arch arch,
                                    T* net_builder,
                                    const paddle::cpp::OpDesc& op_desc,
                                    const Variable& x,
                                    const Variable& y) {
  return std::visit(
      [&](const auto& impl) {
        return AddOpMapperDepthwiseConv2dImpl(impl, net_builder, op_desc, x, y);
      },
      arch.variant());
}

void PaddleModelToProgram::AddOpMapper_depthwise_conv2d() {
  op_mappers_["depthwise_conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Input").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the depthwise_conv2d "
                                     "op is not equal to 1! Please check."));
    auto x_name = op_desc.Input("Input").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Filter").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the depthwise_conv2d "
                                     "op is not equal to 1! Please check."));
    auto y_name = op_desc.Input("Filter").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto* net_builder = net_builder_.get();
    Variable out =
        AddOpMapperDepthwiseConv2d(target_.arch, net_builder, op_desc, x, y);
    CHECK_EQ(op_desc.Output("Output").size(), 1UL);
    auto out_name = op_desc.Output("Output").front();
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_conv2d() {
  op_mappers_["conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Input").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the conv2d op is not "
                                     "equal to 1! Please check."));
    auto x_name = op_desc.Input("Input").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Filter").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the conv2d op is not "
                                     "equal to 1! Please check."));
    auto y_name = op_desc.Input("Filter").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Output").size(),
        1UL,
        phi::errors::InvalidArgument("The output size of the conv2d op is not "
                                     "equal to 1! Please check."));
    auto out_name = op_desc.Output("Output").front();

    CHECK(op_desc.HasAttr("paddings"));
    auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    CHECK(op_desc.HasAttr("strides"));
    auto strides = op_desc.GetAttr<std::vector<int>>("strides");
    CHECK(op_desc.HasAttr("dilations"));
    auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    CHECK(op_desc.HasAttr("groups"));
    auto groups = op_desc.GetAttr<int>("groups");
    CHECK(op_desc.HasAttr("data_format"));
    std::string data_format = op_desc.GetAttr<std::string>("data_format");
    if (data_format == "AnyLayout") {
      data_format = "NCHW";
    }
    auto x = GetVar(TransValidVarName(x_name));
    auto y = GetVar(TransValidVarName(y_name));
    auto out = net_builder_->Conv2d(
        x, y, strides, paddings, dilations, groups, data_format);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_pool2d() {
  op_mappers_["pool2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("X").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the pool2d op is not "
                                     "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument("The output size of the pool2d op is not "
                                     "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();

    CHECK(op_desc.HasAttr("pooling_type"));
    auto pool_type = op_desc.GetAttr<std::string>("pooling_type");
    CHECK(op_desc.HasAttr("ksize"));
    auto ksize = op_desc.GetAttr<std::vector<int>>("ksize");
    CHECK(op_desc.HasAttr("strides"));
    auto strides = op_desc.GetAttr<std::vector<int>>("strides");
    CHECK(op_desc.HasAttr("paddings"));
    auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    CHECK(op_desc.HasAttr("ceil_mode"));
    auto ceil_mode = op_desc.GetAttr<bool>("ceil_mode");
    CHECK(op_desc.HasAttr("exclusive"));
    auto exclusive = op_desc.GetAttr<bool>("exclusive");
    CHECK(op_desc.HasAttr("data_format"));
    auto data_format = op_desc.GetAttr<std::string>("data_format");
    CHECK(op_desc.HasAttr("global_pooling"));
    auto global_pooling = op_desc.GetAttr<bool>("global_pooling");
    CHECK(op_desc.HasAttr("adaptive"));
    auto adaptive = op_desc.GetAttr<bool>("adaptive");

    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Pool2d(x,
                                    pool_type,
                                    ksize,
                                    strides,
                                    paddings,
                                    ceil_mode,
                                    exclusive,
                                    global_pooling,
                                    data_format,
                                    adaptive);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_batchnorm() {
  op_mappers_["batch_norm"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("X").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the batch_norm op is "
                                     "not equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Scale").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the batch_norm op is "
                                     "not equal to 1! Please check."));
    auto scale_name = op_desc.Input("Scale").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Bias").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the batch_norm op is "
                                     "not equal to 1! Please check."));
    auto bias_name = op_desc.Input("Bias").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Mean").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the batch_norm op is "
                                     "not equal to 1! Please check."));
    auto mean_name = op_desc.Input("Mean").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Input("Variance").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the batch_norm op is "
                                     "not equal to 1! Please check."));
    auto variance_name = op_desc.Input("Variance").front();
    CHECK(!op_desc.Output("Y").empty());
    auto out_name = op_desc.Output("Y").front();

    auto x = GetVar(TransValidVarName(x_name));
    auto scale = GetVar(TransValidVarName(scale_name));
    auto bias = GetVar(TransValidVarName(bias_name));
    auto mean = GetVar(TransValidVarName(mean_name));
    auto variance = GetVar(TransValidVarName(variance_name));
    CHECK(op_desc.HasAttr("epsilon"));
    auto epsilon = op_desc.GetAttr<float>("epsilon");
    CHECK(op_desc.HasAttr("momentum"));
    auto momentum = op_desc.GetAttr<float>("momentum");
    // CHECK(op_desc.HasAttr("data_format"));
    // auto data_format = op_desc.GetAttr<std::string>("data_format");
    std::string data_format = "NCHW";

    auto out = net_builder_->BatchNorm(
        x, scale, bias, mean, variance, epsilon, momentum, data_format, true);

    AddVar(TransValidVarName(out_name), out[0]);
    var_model_to_program_map_[out_name] = out[0]->id;
  };
}

void PaddleModelToProgram::AddOpMapper_sigmoid() {
  op_mappers_["sigmoid"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(
        op_desc.Input("X").size(),
        1UL,
        phi::errors::InvalidArgument("The input size of the sigmoid op is not "
                                     "equal to 1! Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument("The output size of the sigmoid op is not "
                                     "equal to 1! Please check."));
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Sigmoid(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_slice() {
  op_mappers_["slice"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the slice op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("Input").front();
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The output size of the slice op is not equal to 1! "
                          "Please check."));
    auto out_name = op_desc.Output("Out").front();

    absl::flat_hash_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("starts"));
    auto starts = op_desc.GetAttr<std::vector<int>>("starts");
    CHECK(op_desc.HasAttr("ends"));
    auto end = op_desc.GetAttr<std::vector<int>>("ends");
    CHECK(op_desc.HasAttr("axes"));
    auto axes = op_desc.GetAttr<std::vector<int>>("axes");
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Slice(x, axes, starts, end);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_dropout_infer() {
  op_mappers_["dropout"] = [&](const paddle::cpp::OpDesc& op_desc) {
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The input size of the dropout op is not equal to 1! "
                          "Please check."));
    auto x_name = op_desc.Input("X").front();
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument(
            "The output size of the dropout op is not equal to 1! "
            "Please check."));
    auto out_name = op_desc.Output("Out").front();

    absl::flat_hash_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    CHECK(op_desc.HasAttr("dropout_prob"));
    auto dropout_prob = op_desc.GetAttr<float>("dropout_prob");
    CHECK(op_desc.HasAttr("dropout_implementation"));
    auto dropout_implementation =
        op_desc.GetAttr<std::string>("dropout_implementation");
    auto x = GetVar(TransValidVarName(x_name));
    auto out =
        net_builder_->DropoutInfer(x, dropout_prob, dropout_implementation);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOp(const paddle::cpp::OpDesc& op_desc) {
  const auto& op_type = op_desc.Type();
  auto it = op_mappers_.find(op_type);
  if (it != op_mappers_.end()) {
    it->second(op_desc);
    return;
  }
  // feed op's output is a input of the model
  std::stringstream ss;
  ss << "Not supported op [" << op_desc.Type() << "] found";
  PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
}

void PaddleModelToProgram::TransposeVar(const std::string& name) {
  CheckVarNameValid(name);
  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (std::holds_alternative<common::X86Arch>(target_.arch)) {
      float* data = tensor->mutable_data<float>(target_);
      CHECK(tensor->shape().size() == 2)
          << "The y data's shape size of op [mul] is not equal to 2! Please "
             "check.";
      TransposeData(data, tensor->shape().data()[0], tensor->shape().data()[1]);
    } else if (std::holds_alternative<common::NVGPUArch>(target_.arch)) {
#ifdef CINN_WITH_CUDA
      // To use cublas mul api, there is no need to transpose data.
#ifndef CINN_WITH_CUDNN
      std::vector<float> data(tensor->shape().numel());
      CUDA_CALL(cudaMemcpy(
          data.data(),
          reinterpret_cast<void*>(tensor->mutable_data<float>(target_)),
          tensor->shape().numel() * sizeof(float),
          cudaMemcpyDeviceToHost));
      CHECK(tensor->shape().size() == 2)
          << "The y data's shape size of op [mul] is not equal to 2! Please "
             "check.";
      TransposeData(
          data.data(), tensor->shape().data()[0], tensor->shape().data()[1]);
      CUDA_CALL(cudaMemcpy(
          reinterpret_cast<void*>(tensor->mutable_data<float>(target_)),
          data.data(),
          tensor->shape().numel() * sizeof(float),
          cudaMemcpyHostToDevice));
#endif
#else
      PADDLE_THROW(phi::errors::Fatal(
          "To use CUDA backends, you need to set WITH_CUDA ON!"));
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }

    Variable var;
    var.set_id(name);
    std::vector<int> reverse_shape = tensor->shape().data();
    std::reverse(reverse_shape.begin(), reverse_shape.end());
    tensor->shape().SetData(reverse_shape);
    var->shape = tensor->shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    AddVar(name, var, true);
  } else {
    std::stringstream ss;
    ss << "No var called [" << name << "] exists";
    PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  }
}

void PaddleModelToProgram::ReverseHWVar(const std::string& name) {
  CheckVarNameValid(name);
  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (std::holds_alternative<common::X86Arch>(target_.arch)) {
      float* data = tensor->mutable_data<float>(target_);
      CHECK(tensor->shape().size() == 4)
          << "The y data's shape size of op [conv2d] is not equal to 4! Please "
             "check.";
      ReverseHWData(data, tensor->shape().data());
    } else if (std::holds_alternative<common::NVGPUArch>(target_.arch)) {
#ifdef CINN_WITH_CUDA
      std::vector<float> data(tensor->shape().numel());
      CUDA_CALL(cudaMemcpy(
          data.data(),
          reinterpret_cast<void*>(tensor->mutable_data<float>(target_)),
          tensor->shape().numel() * sizeof(float),
          cudaMemcpyDeviceToHost));
      CHECK(tensor->shape().size() == 4)
          << "The y data's shape size of op [conv2d] is not equal to 4! Please "
             "check.";
      ReverseHWData(data.data(), tensor->shape().data());
      CUDA_CALL(cudaMemcpy(
          reinterpret_cast<void*>(tensor->mutable_data<float>(target_)),
          data.data(),
          tensor->shape().numel() * sizeof(float),
          cudaMemcpyHostToDevice));
#else
      PADDLE_THROW(phi::errors::Fatal(
          "To use CUDA backends, you need to set WITH_CUDA ON!"));
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    std::stringstream ss;
    ss << "No var called [" << name << "] exists";
    PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  }
}

Variable PaddleModelToProgram::GetVar(const std::string& name) {
  CheckVarNameValid(name);

  auto it = var_map_.find(name);
  if (it != var_map_.end()) return it->second;

  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    Variable var;
    var.set_id(name);
    var->shape = tensor->shape().data();
    // TODO(Superjomn) Make this determined by model.
    var->type = Float(32);
    var.set_const(true);
    AddVar(name, var);
    return var;
  }

  std::stringstream ss;
  ss << "No var called [" << name << "] exists";
  PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  return Variable();
}

std::unique_ptr<Program> PaddleModelToProgram::operator()(
    const std::string& model_dir, bool is_combined) {
  paddle::cpp::ProgramDesc program_desc;
  if (FLAGS_cinn_infer_model_version < 2.0) {
    paddle::LoadModelPb(model_dir,
                        "/__model__",
                        "/params",
                        scope_,
                        &program_desc,
                        is_combined,
                        false,
                        target_);
  } else {
    paddle::LoadModelPb(model_dir,
                        ".pdmodel",
                        ".pdiparams",
                        scope_,
                        &program_desc,
                        is_combined,
                        false,
                        target_);
  }
  PADDLE_ENFORCE_EQ(program_desc.BlocksSize(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "CINN can only support the model with a single block"));
  auto* block_desc = program_desc.GetBlock<paddle::cpp::BlockDesc>(0);

  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    AddOp(*op_desc);
  }
  return std::unique_ptr<Program>(new Program(net_builder_->Build()));
}

void PaddleModelToProgram::AddVar(const std::string& name,
                                  const Variable& var,
                                  bool replace) {
  CheckVarNameValid(name);
  if (replace == false) {
    CHECK(!var_map_.count(name)) << "Duplicate variable [" << name << "] found";
  }
  var_map_[name] = var;
}

}  // namespace frontend
}  // namespace cinn
