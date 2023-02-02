/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

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

/*
 * ReshapeOp
 */
class ReshapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    std::cout << "log 开始-----------------------------------------------------"
              << std::endl;
    std::cout << op_desc.Input("X")[0] << std::endl;
    std::vector<int> compiled_shape = PADDLE_GET_CONST(
        std::vector<int>, op_desc.GetAttr("compiled_time_shape"));
    std::vector<int> shape =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("shape"));
    int nbDims_num = shape.size();
    nvinfer1::Dims reshape_dim;
    nvinfer1::ITensor* real_shape_tensor = nullptr;
    std::vector<nvinfer1::ITensor*> concat_inputs;
    bool one_input = false;
    if (engine_->with_dynamic_shape()) {
      if (op_desc.Inputs().find("ShapeTensor") != op_desc.Inputs().end() &&
          op_desc.Input("ShapeTensor").size() > 0) {
        int output_rank = op_desc.Input("ShapeTensor").size();
        for (int i = 0; i < output_rank; i++) {
          if (engine_->GetITensor(op_desc.Input("ShapeTensor")[i])->getType() ==
              nvinfer1::DataType::kINT32) {
            std::cout << "是int32" << std::endl;
          } else {
            std::cout << "不是int32" << std::endl;
          }
          if (compiled_shape[i] > 0 && i >= 0 && 0) {
            concat_inputs.push_back(Add1DConstantLayer(compiled_shape[i]));
          } else {
            concat_inputs.push_back(
                engine_->GetITensor(op_desc.Input("ShapeTensor")[i]));
          }
        }
        real_shape_tensor = Concat(concat_inputs);
      } else if (op_desc.Inputs().find("Shape") != op_desc.Inputs().end() &&
                 op_desc.Input("Shape").size() > 0) {
        real_shape_tensor = engine_->GetITensor(op_desc.Input("Shape")[0]);
      } else {
        reshape_dim.nbDims = nbDims_num;
        for (int i = 0; i < nbDims_num; ++i) {
          reshape_dim.d[i] = shape[i];
        }
        one_input = true;
      }
    } else {  // running the TRT Static Shape mode
      reshape_dim.nbDims = nbDims_num - 1;
      for (int i = 0; i < nbDims_num - 1; ++i) {
        reshape_dim.d[i] = shape[i + 1];
      }
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    std::cout << "输入的rank：" << input->getDimensions().nbDims << std::endl;

    if (!engine_->with_dynamic_shape() || one_input || !real_shape_tensor)
      layer->setReshapeDimensions(reshape_dim);
    else
      layer->setInput(1, *real_shape_tensor);

    std::cout << "log 结束-----------------------------------------------------"
              << std::endl;

    PADDLE_ENFORCE_GE(
        layer->getOutput(0)->getDimensions().nbDims,
        0,
        platform::errors::InvalidArgument(
            "Errors occures in Paddle-TRT reshape2 op, try to use C++ Api "
            "config.Exp_DisableTensorRtOPs({\"reshape2\"})\n; or Python Api "
            "config.exp_disable_tensorrt_ops([\"reshape2\"]) to forbid "
            "reshape2 op into "
            "Paddle-TRT."));
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "reshape", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(reshape, ReshapeOpConverter);
REGISTER_TRT_OP_CONVERTER(reshape2, ReshapeOpConverter);
