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
  CHECK_EQ(shape.size(), 4UL);
  for (int i = 0; i < shape[0] * shape[1]; i++) {
    int num = shape[2] * shape[3];
    std::reverse(data + (i * num), data + (i * num + num));
  }
}

void PaddleModelToProgram::AddOpMapper_feed() {
  op_mappers_["feed"] = [&](const paddle::cpp::OpDesc& op_desc) {
    auto outs = op_desc.Output("Out");
    CHECK_EQ(outs.size(), 1UL);
    VLOG(2) << "Model get feed [" << outs[0] << "]";
    CHECK(input_shape_map_.count(outs[0]));
    auto input_shape = input_shape_map_[outs[0]];
    auto input = net_builder_->CreateInput(Float(32), input_shape, outs[0]);
    AddVar(outs[0], input);
  };
}

void PaddleModelToProgram::AddOpMapper_fetch() {
  op_mappers_["fetch"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto output_names = op_desc.Input("X");
    for (auto& output_name : output_names) {
      VLOG(2) << "fetch model output: [" << output_name << "]";
      fetch_names_.insert(utils::TransValidVarName(output_name));
    }
  };
}

void PaddleModelToProgram::AddOpMapper_scale() {
  op_mappers_["scale"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    float scale{};
    float bias{};
    if (op_desc.HasAttr("scale")) {  // the old model format
      scale = op_desc.GetAttr<float>("scale");
    } else {  // the newly refactored format
      // load scale tensor
      CHECK_EQ(op_desc.Input("ScaleTensor").size(), 1UL);
      auto* scale_tensor_var =
          scope_->FindVar(op_desc.Input("ScaleTensor").front());
      CHECK(scale_tensor_var) << "No scale tensor found in the scope";
      auto& scale_tensor =
          absl::get<hlir::framework::Tensor>(*scale_tensor_var);
      scale = scale_tensor->mutable_data<float>(common::DefaultHostTarget())[0];
    }
    if (op_desc.HasAttr("bias")) {  // the old model format
      bias = op_desc.GetAttr<float>("bias");
    } else {
      LOG(FATAL) << "Didn't find [bias] attr in Scale operator!!";
    }
    absl::flat_hash_map<std::string, hlir::framework::NodeAttr::attr_t> attrs;
    auto out = net_builder_->Scale(x, scale, bias);
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_mul() {
  op_mappers_["mul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
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

    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_matmul() {
  op_mappers_["matmul"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
    auto y_name = op_desc.Input("Y").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    auto y = GetVar(utils::TransValidVarName(y_name));
    bool trans_a = op_desc.GetAttr<bool>("transpose_X");
    bool trans_b = op_desc.GetAttr<bool>("transpose_Y");
    float alpha = op_desc.GetAttr<float>("alpha");
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    VLOG(4) << "y shape: " << utils::Join(y->shape, ",");
    auto out = net_builder_->Matmul(x, y, trans_a, trans_b, alpha);
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_reshape2() {
  op_mappers_["reshape2"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    auto x = GetVar(utils::TransValidVarName(x_name));
    std::vector<int> shape = op_desc.GetAttr<std::vector<int>>("shape");
    VLOG(4) << "x shape: " << utils::Join(x->shape, ",");
    auto out = net_builder_->Reshape(x, shape);
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_concat() {
  op_mappers_["concat"] = [&](const paddle::cpp::OpDesc& op_desc) {
    int input_size = op_desc.Input("X").size();
    CHECK_GE(input_size, 2UL);
    std::vector<Variable> input_vars;
    for (int i = 0; i < input_size; i++) {
      auto name = op_desc.Input("X")[i];
      input_vars.push_back(GetVar(utils::TransValidVarName(name)));
    }
    int axis = op_desc.GetAttr<int>("axis");
    VLOG(4) << "axis in op concat is : " << axis;
    auto out = net_builder_->Concat(input_vars, axis);
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    AddVar(utils::TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_assign() {
  op_mappers_["assign"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Identity(x);
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_fill_constant() {
  op_mappers_["fill_constant"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();

    CHECK(op_desc.HasAttr("shape"));
    auto shape = op_desc.GetAttr<std::vector<int64_t>>("shape");
    std::vector<int> shapes;
    for (size_t i = 0; i < shape.size(); i++) {
      CHECK_LE(shape[i], std::numeric_limits<int32_t>::max());
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
        LOG(FATAL) << "unknown data type " << dtype;
    }
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_transpose2() {
  op_mappers_["transpose2"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));

    auto out = net_builder_->Exp(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_relu() {
  op_mappers_["relu"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Relu(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_softmax() {
  op_mappers_["softmax"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
    auto y_name = op_desc.Input("Y").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
    auto y_name = op_desc.Input("Y").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
    auto y_name = op_desc.Input("Y").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);
    auto y_name = op_desc.Input("Y").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();

    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Relu6(x);
    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}
void PaddleModelToProgram::AddOpMapper_depthwise_conv2d() {
  op_mappers_["depthwise_conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("Input").size(), 1UL);
    auto x_name = op_desc.Input("Input").front();
    CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
    auto y_name = op_desc.Input("Filter").front();
    CHECK_EQ(op_desc.Output("Output").size(), 1UL);
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
    Variable out;
    if (target_.arch == Target::Arch::X86) {
      out = net_builder_->Conv2d(
          x, y, strides, paddings, dilations, groups, data_format);
    } else {
      out = net_builder_->DepthwiseConv2d(
          x, y, strides, paddings, dilations, groups, data_format);
    }

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_conv2d() {
  op_mappers_["conv2d"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("Input").size(), 1UL);
    auto x_name = op_desc.Input("Input").front();
    CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
    auto y_name = op_desc.Input("Filter").front();
    CHECK_EQ(op_desc.Output("Output").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Input("Scale").size(), 1UL);
    auto scale_name = op_desc.Input("Scale").front();
    CHECK_EQ(op_desc.Input("Bias").size(), 1UL);
    auto bias_name = op_desc.Input("Bias").front();
    CHECK_EQ(op_desc.Input("Mean").size(), 1UL);
    auto mean_name = op_desc.Input("Mean").front();
    CHECK_EQ(op_desc.Input("Variance").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    auto out_name = op_desc.Output("Out").front();
    auto x = GetVar(TransValidVarName(x_name));
    auto out = net_builder_->Sigmoid(x);

    AddVar(TransValidVarName(out_name), out);
    var_model_to_program_map_[out_name] = out->id;
  };
}

void PaddleModelToProgram::AddOpMapper_slice() {
  op_mappers_["slice"] = [&](const paddle::cpp::OpDesc& op_desc) {
    CHECK_EQ(op_desc.Input("Input").size(), 1UL);
    auto x_name = op_desc.Input("Input").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
    CHECK_EQ(op_desc.Input("X").size(), 1UL);
    auto x_name = op_desc.Input("X").front();
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
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
  LOG(FATAL) << "Not supported op [" << op_desc.Type() << "] found";
}

void PaddleModelToProgram::TransposeVar(const std::string& name) {
  CheckVarNameValid(name);
  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (target_.arch == Target::Arch::X86) {
      float* data = tensor->mutable_data<float>(target_);
      CHECK(tensor->shape().size() == 2)
          << "The y data's shape size of op [mul] is not equal to 2! Please "
             "check.";
      TransposeData(data, tensor->shape().data()[0], tensor->shape().data()[1]);
    } else if (target_.arch == Target::Arch::NVGPU) {
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
      LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
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
    LOG(FATAL) << "No var called [" << name << "] exists";
  }
}

void PaddleModelToProgram::ReverseHWVar(const std::string& name) {
  CheckVarNameValid(name);
  auto* var = scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (target_.arch == Target::Arch::X86) {
      float* data = tensor->mutable_data<float>(target_);
      CHECK(tensor->shape().size() == 4)
          << "The y data's shape size of op [conv2d] is not equal to 4! Please "
             "check.";
      ReverseHWData(data, tensor->shape().data());
    } else if (target_.arch == Target::Arch::NVGPU) {
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
      LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    LOG(FATAL) << "No var called [" << name << "] exists";
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

  LOG(FATAL) << "No var called [" << name << "] exists";
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
  CHECK_EQ(program_desc.BlocksSize(), 1)
      << "CINN can only support the model with a single block";
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
