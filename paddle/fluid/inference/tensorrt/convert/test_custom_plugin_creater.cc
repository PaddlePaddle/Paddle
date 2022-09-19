/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>  // NOLINT

#include "paddle/extension.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/test_custom_op_plugin.h"

PD_BUILD_OP(custom_op)
    .Inputs({"Input"})
    .Outputs({"Output"})
    .Attrs({
        "float_attr",
        "int_attr",
        "bool_attr",
        "string_attr",
        "ints_attr",
        "floats_attr",
        "bools_attr",
    });

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(CustomPluginCreater, StaticShapePlugin) {
  framework::ProgramDesc prog;
  auto *block = prog.MutableBlock(0);
  auto *op = block->AppendOp();
  framework::proto::OpDesc *op_desc = op->Proto();

  op_desc->set_type("custom_op");
  auto *input_var = op_desc->add_inputs();
  input_var->set_parameter("Input");
  *input_var->add_arguments() = "X";

  auto *output_var = op_desc->add_outputs();
  output_var->set_parameter("Output");
  *output_var->add_arguments() = "Out";

  auto *attr = op_desc->add_attrs();
  attr->set_name("float_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(1.0);

  attr = op_desc->add_attrs();
  attr->set_name("int_attr");
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(1);

  attr = op_desc->add_attrs();
  attr->set_name("bool_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);
  attr->set_b(true);

  attr = op_desc->add_attrs();
  attr->set_name("string_attr");
  attr->set_type(paddle::framework::proto::AttrType::STRING);
  attr->set_s("test_string_attr");

  attr = op_desc->add_attrs();
  attr->set_name("ints_attr");
  attr->set_type(paddle::framework::proto::AttrType::INTS);
  attr->add_ints(1);
  attr->add_ints(2);
  attr->add_ints(3);

  attr = op_desc->add_attrs();
  attr->set_name("floats_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOATS);
  attr->add_floats(1.0);
  attr->add_floats(2.0);
  attr->add_floats(3.0);

  attr = op_desc->add_attrs();
  attr->set_name("bools_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEANS);
  attr->add_bools(true);
  attr->add_bools(false);
  attr->add_bools(true);

  // init trt engine
  std::unique_ptr<TensorRTEngine> engine_;
  engine_.reset(new TensorRTEngine(5, 1 << 15));
  engine_->InitNetwork();

  engine_->DeclareInput(
      "X", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3(2, 5, 5));

  framework::Scope scope;

  tensorrt::plugin::TrtPluginRegistry::Global()->RegistToTrt();

  auto &custom_plugin_tell = OpTeller::Global().GetCustomPluginTeller();

  framework::OpDesc custom_op(*op_desc, nullptr);
  CHECK_EQ((*custom_plugin_tell)(custom_op, false, false), true);

  OpTeller::Global().SetOpConverterType("custom_op",
                                        OpConverterType::CustomPluginCreater);

  OpConverter converter;
  converter.ConvertBlock(
      *block->Proto(), {}, scope, engine_.get() /*TensorRTEngine*/);
}

TEST(CustomPluginCreater, DynamicShapePlugin) {
  framework::ProgramDesc prog;
  auto *block = prog.MutableBlock(0);
  auto *op = block->AppendOp();
  framework::proto::OpDesc *op_desc = op->Proto();

  op_desc->set_type("custom_op");
  auto *input_var = op_desc->add_inputs();
  input_var->set_parameter("Input");
  *input_var->add_arguments() = "X";

  auto *output_var = op_desc->add_outputs();
  output_var->set_parameter("Output");
  *output_var->add_arguments() = "Out";

  auto *attr = op_desc->add_attrs();
  attr->set_name("float_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);

  attr = op_desc->add_attrs();
  attr->set_name("int_attr");
  attr->set_type(paddle::framework::proto::AttrType::INT);

  attr = op_desc->add_attrs();
  attr->set_name("bool_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEAN);

  attr = op_desc->add_attrs();
  attr->set_name("string_attr");
  attr->set_type(paddle::framework::proto::AttrType::STRING);

  attr = op_desc->add_attrs();
  attr->set_name("ints_attr");
  attr->set_type(paddle::framework::proto::AttrType::INTS);

  attr = op_desc->add_attrs();
  attr->set_name("floats_attr");
  attr->set_type(paddle::framework::proto::AttrType::FLOATS);

  attr = op_desc->add_attrs();
  attr->set_name("bools_attr");
  attr->set_type(paddle::framework::proto::AttrType::BOOLEANS);

  // init trt engine
  std::unique_ptr<TensorRTEngine> engine_;

  std::map<std::string, std::vector<int>> min_input_shape = {
      {"x", {1, 2, 5, 5}}};

  std::map<std::string, std::vector<int>> max_input_shape = {
      {"x", {1, 2, 5, 5}}};

  std::map<std::string, std::vector<int>> optim_input_shape = {
      {"x", {1, 2, 5, 5}}};

  engine_.reset(new TensorRTEngine(5,
                                   1 << 15,
                                   AnalysisConfig::Precision::kFloat32,
                                   nullptr,
                                   0,
                                   min_input_shape,
                                   max_input_shape,
                                   optim_input_shape));
  engine_->InitNetwork();

  LOG(INFO) << "with_dynamic_shape " << engine_->with_dynamic_shape();
  engine_->DeclareInput(
      "X", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, 2, 5, 5));

  framework::Scope scope;

  tensorrt::plugin::TrtPluginRegistry::Global()->RegistToTrt();

  auto &custom_plugin_tell = OpTeller::Global().GetCustomPluginTeller();

  framework::OpDesc custom_op(*op_desc, nullptr);
  CHECK_EQ((*custom_plugin_tell)(custom_op, false, true), true);

  OpTeller::Global().SetOpConverterType("custom_op",
                                        OpConverterType::CustomPluginCreater);

  OpConverter converter;
  converter.ConvertBlock(
      *block->Proto(), {}, scope, engine_.get() /*TensorRTEngine*/);
}
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_TRT_CONVERTER(custom_plugin_creater)
