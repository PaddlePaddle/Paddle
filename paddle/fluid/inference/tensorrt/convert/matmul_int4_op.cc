/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int4_plugin.h"

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
class MatMulInt4OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a int4 matmul op to cutlass int4 plugin";
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X").front());
    auto* input2 = engine_->GetITensor(op_desc.Input("Y").front());

    nvinfer1::Dims dims_x = input1->getDimensions();
    nvinfer1::Dims dims_y = input2->getDimensions();

    bool transpose_X = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_x"));
    bool transpose_Y = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_y"));

    auto output_name = op_desc.Output("Out").front();

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    if (transpose_X) {
      nvinfer1::Permutation permutation;
      for (int i = 0; i < dims_x.nbDims - 2; ++i) {
        permutation.order[i] = i;
      }
      permutation.order[dims_x.nbDims - 2] = dims_x.nbDims - 1;
      permutation.order[dims_x.nbDims - 1] = dims_x.nbDims - 2;
      auto* transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      transpose_layer->setFirstTranspose(permutation);
      transpose_layer->setName(
          ("matmul_int4_op_transpose_x: Shuffle (Output:" + output_name + ")")
              .c_str());
      plugin_inputs.push_back(transpose_layer->getOutput(0));
      dims_x = plugin_inputs.back()->getDimensions();
    } else {
      plugin_inputs.push_back(input1);
    }
    if (!transpose_Y) {
      // cutlass int4 gemm need y in column major,so the action on y is opposite
      nvinfer1::Permutation permutation;
      for (int i = 0; i < dims_y.nbDims - 2; ++i) {
        permutation.order[i] = i;
      }
      permutation.order[dims_y.nbDims - 2] = dims_y.nbDims - 1;
      permutation.order[dims_y.nbDims - 1] = dims_y.nbDims - 2;
      auto* transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input2);
      transpose_layer->setFirstTranspose(permutation);
      transpose_layer->setName(
          ("matmul_int4_op_transpose_y: Shuffle (Output:" + output_name + ")")
              .c_str());
      plugin_inputs.push_back(transpose_layer->getOutput(0));
    } else {
      plugin_inputs.push_back(input2);
      dims_y = plugin_inputs.back()->getDimensions();
      std::swap(dims_y.d[dims_y.nbDims - 1], dims_y.d[dims_y.nbDims - 2]);
    }

    // nvinfer1::Dims dims_x_ = plugin_inputs[0]->getDimensions();
    // nvinfer1::Dims dims_y_ = plugin_inputs[1]->getDimensions();

    std::vector<nvinfer1::PluginField> fields;
    fields.emplace_back("dims_x", &dims_x, nvinfer1::PluginFieldType::kDIMS, 1);
    fields.emplace_back("dims_y", &dims_y, nvinfer1::PluginFieldType::kDIMS, 1);

    nvinfer1::PluginFieldCollection* plugin_ptr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_ptr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_ptr->nbFields = fields.size();
    plugin_ptr->fields = fields.data();

    auto creator =
        GetPluginRegistry()->getPluginCreator("MatmulInt4PluginCreator", "1");
    auto plugin_obj = creator->createPlugin("MatmulInt4Plugin", plugin_ptr);
    auto plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

    plugin_layer->setName(
        ("matmul_int4: (Output: " + output_name + ")").c_str());
    engine_->SetITensor(output_name, plugin_layer->getOutput(0));
    free(plugin_ptr);
  }
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul_int4, MatMulInt4OpConverter);
