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

#include <NvInfer.h>
#include <string>
#include "glog/logging.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class UnaryOpConverter : public OpConverter {
 public:
  UnaryOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert a fluid unary op to tensorrt unary layer whose "
               "type is "
            << op_type_;
    nvinfer1::ITensor* input_tensor =
        engine_->GetITensor(op_desc.Input("X")[0]);
    auto op_pair = ops.find(op_type_);
    nvinfer1::IUnaryLayer* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Unary, *input_tensor, op_pair->second);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, op_type_, {output_name}, test_mode);
  }

 protected:
  std::string op_type_;
  static const std::unordered_map<std::string, nvinfer1::UnaryOperation> ops;
};

const std::unordered_map<std::string, nvinfer1::UnaryOperation>
    UnaryOpConverter::ops = {
        {"exp", nvinfer1::UnaryOperation::kEXP},
        {"log", nvinfer1::UnaryOperation::kLOG},
};

class ExpOpConverter : public UnaryOpConverter {
 public:
  ExpOpConverter() { op_type_ = "exp"; }
};

class LogOpConverter : public UnaryOpConverter {
 public:
  LogOpConverter() { op_type_ = "log"; }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(exp, ExpOpConverter);
REGISTER_TRT_OP_CONVERTER(log, LogOpConverter);
