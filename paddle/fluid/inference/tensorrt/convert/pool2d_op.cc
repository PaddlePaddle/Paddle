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
namespace inference {
namespace tensorrt {

/*
 * Pool2dOp, IPoolingLayer in TRT. This Layer doesn't has weights.
 */
class Pool2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4)
        << "convert a fluid pool2d op to tensorrt pool2d layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);

    bool global_pooling = boost::get<bool>(op_desc.GetAttr("global_pooling"));
    std::string pool_type =
        boost::get<std::string>(op_desc.GetAttr("pooling_type"));
    std::vector<int> ksize =
        boost::get<std::vector<int>>(op_desc.GetAttr("ksize"));
    std::vector<int> strides =
        boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
    std::vector<int> paddings =
        boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));

    nvinfer1::DimsHW nv_ksize(ksize[0], ksize[1]);
    if (global_pooling == true) {
      nvinfer1::Dims input_shape = input1->getDimensions();
      int nbDims = input_shape.nbDims;
      nv_ksize.d[0] = input_shape.d[nbDims - 2];
      nv_ksize.d[1] = input_shape.d[nbDims - 1];
    }
    const nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
    const nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

    PADDLE_ENFORCE_EQ(input1->getDimensions().nbDims, 3UL);

    nvinfer1::PoolingType nv_pool_type = nvinfer1::PoolingType::kMAX;
    if (pool_type == "max") {
      nv_pool_type = nvinfer1::PoolingType::kMAX;
    } else if (pool_type == "avg") {
      nv_pool_type = nvinfer1::PoolingType::kAVERAGE;
    } else {
      PADDLE_THROW("TensorRT unsupported pooling type!");
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Pooling,
                                       *const_cast<nvinfer1::ITensor*>(input1),
                                       nv_pool_type, nv_ksize);
    PADDLE_ENFORCE_NOT_NULL(layer, "pool layer could not be created.");
    layer->setStride(nv_strides);
    layer->setPadding(nv_paddings);

    auto output_name = op_desc.Output("Out")[0];
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(pool2d);
REGISTER_TRT_OP_CONVERTER(pool2d, Pool2dOpConverter);
