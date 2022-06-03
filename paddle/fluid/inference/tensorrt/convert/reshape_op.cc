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
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    std::vector<int> shape =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("shape"));
    int nbDims_num = shape.size();
    nvinfer1::Dims reshape_dim;
    nvinfer1::ITensor* real_shape_tensor;
    std::vector<nvinfer1::ITensor*> concat_inputs;

    if (engine_->with_dynamic_shape()) {
      if (op_desc.Inputs().find("ShapeTensor") != op_desc.Inputs().end() &&
          op_desc.Input("ShapeTensor").size() > 0) {
        for (size_t i = 0; i < op_desc.Input("ShapeTensor").size(); i++)
          concat_inputs.push_back(
              engine_->GetITensor(op_desc.Input("ShapeTensor")[i]));
        real_shape_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs.data(),
                                 concat_inputs.size())
                ->getOutput(0);
      } else {
        std::vector<int> shape =
            BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("shape"));
        std::vector<int> del_batch_shape(shape.begin() + 1, shape.end());
        auto input_shape_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
        std::vector<int32_t> gather_indices(1, 0);
        std::string name = "_add_reshape_op_";
        std::cout << name << std::endl;
        auto gather_indices_tensor =
            Add1DConstantLayer(gather_indices, name + "gather_indices");
        auto batch_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Gather, *input_shape_tensor,
                                 *gather_indices_tensor, 0)
                ->getOutput(0);
        auto del_batch_shape_tensor =
            Add1DConstantLayer(del_batch_shape, name + "del_batch_shape");
        concat_inputs.push_back(batch_tensor);
        concat_inputs.push_back(del_batch_shape_tensor);
        real_shape_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs.data(),
                                 concat_inputs.size())
                ->getOutput(0);
      }
    } else {  // running the TRT Static Shape mode
      reshape_dim.nbDims = nbDims_num - 1;
      for (int i = 0; i < nbDims_num - 1; ++i) {
        reshape_dim.d[i] = shape[i + 1];
      }
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    if (!engine_->with_dynamic_shape())
      layer->setReshapeDimensions(reshape_dim);
    else
      layer->setInput(1, *real_shape_tensor);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "reshape", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(reshape, ReshapeOpConverter);
