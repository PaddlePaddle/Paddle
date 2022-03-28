/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/plugin/pool3d_op_plugin.h"

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

inline void DealCeilMode(const nvinfer1::Dims &input_shape,
                         std::vector<int> ksize, std::vector<int> strides,
                         std::vector<int> paddings, nvinfer1::Dims3 *pre_pad,
                         nvinfer1::Dims3 *post_pad, int input_dims) {
  int input_depth = input_shape.d[input_dims - 3];
  int input_height = input_shape.d[input_dims - 2];
  int input_width = input_shape.d[input_dims - 1];

  int floor_d_output_size =
      (input_depth - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
  int ceil_d_output_size =
      (input_depth - ksize[0] + 2 * paddings[0] + strides[0] - 1) / strides[0] +
      1;

  int floor_h_output_size =
      (input_height - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
  int ceil_h_output_size =
      (input_height - ksize[1] + 2 * paddings[1] + strides[1] - 1) /
          strides[1] +
      1;

  int floor_w_output_size =
      (input_width - ksize[2] + 2 * paddings[2]) / strides[2] + 1;
  int ceil_w_output_size =
      (input_width - ksize[2] + 2 * paddings[2] + strides[2] - 1) / strides[2] +
      1;

  if (floor_d_output_size != ceil_d_output_size) {
    post_pad->d[0] = strides[0] - 1;
  }

  if (floor_h_output_size != ceil_h_output_size) {
    post_pad->d[1] = strides[1] - 1;
  }

  if (floor_w_output_size != ceil_w_output_size) {
    post_pad->d[2] = strides[2] - 1;
  }
}

class Pool3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope, bool test_mode) override {
    VLOG(4)
        << "convert a fluid pool3d op to tensorrt pool3d layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    auto *input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims input_shape = input1->getDimensions();
    int input_dims = input_shape.nbDims;

    bool global_pooling =
        BOOST_GET_CONST(bool, op_desc.GetAttr("global_pooling"));
    std::string pool_type =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("pooling_type"));
    std::vector<int> ksize =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("ksize"));
    std::vector<int> strides =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    std::vector<int> paddings =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
    bool exclusive = op_desc.HasAttr("exclusive")
                         ? BOOST_GET_CONST(bool, op_desc.GetAttr("exclusive"))
                         : true;
    bool ceil_mode = BOOST_GET_CONST(bool, op_desc.GetAttr("ceil_mode"));
    bool adaptive = false;
    if (op_desc.HasAttr("adaptive"))
      adaptive = BOOST_GET_CONST(bool, op_desc.GetAttr("adaptive"));
    std::string padding_algorithm = "EXPLICIT";
    if (op_desc.HasAttr("padding_algorithm"))
      padding_algorithm =
          BOOST_GET_CONST(std::string, op_desc.GetAttr("padding_algorithm"));
    if (padding_algorithm == "VALID" || padding_algorithm == "SAME") {
      std::fill(paddings.begin(), paddings.end(), 0);
    }

    nvinfer1::PoolingType nv_pool_type = nvinfer1::PoolingType::kMAX;
    nvinfer1::ReduceOperation reduce_operation =
        nvinfer1::ReduceOperation::kMAX;
    plugin::Pool3DPlugin::Pool3DType plugin_pool_type =
        plugin::Pool3DPlugin::Pool3DType::max;
    if (pool_type == "max") {
      nv_pool_type = nvinfer1::PoolingType::kMAX;
      reduce_operation = nvinfer1::ReduceOperation::kMAX;
      plugin_pool_type = plugin::Pool3DPlugin::Pool3DType::max;
    } else if (pool_type == "avg") {
      nv_pool_type = nvinfer1::PoolingType::kAVERAGE;
      reduce_operation = nvinfer1::ReduceOperation::kAVG;
      plugin_pool_type = plugin::Pool3DPlugin::Pool3DType::avg;
    }
    nvinfer1::Dims3 nv_ksize(ksize[0], ksize[1], ksize[2]);
    nvinfer1::Dims3 nv_strides(strides[0], strides[1], strides[2]);
    nvinfer1::Dims3 nv_paddings(paddings[0], paddings[1], paddings[2]);
    nvinfer1::ILayer *layer = nullptr;
    if (op_desc.HasAttr("enable_int8")) {
      CHECK(op_desc.HasAttr("X_scale"));
      float input_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X_scale"));
      engine_->SetTensorDynamicRange(input1, input_scale);
    }

    if (engine_->with_dynamic_shape()) {
      if (!adaptive && !global_pooling && !ceil_mode) {
        auto *pool_layer = TRT_ENGINE_ADD_LAYER(engine_, PoolingNd, *input1,
                                                nv_pool_type, nv_ksize);
        pool_layer->setStrideNd(nv_strides);
        pool_layer->setPaddingNd(nv_paddings);
        pool_layer->setAverageCountExcludesPadding(exclusive);
        layer = pool_layer;
      } else if (global_pooling) {
        auto *reduce_layer = TRT_ENGINE_ADD_LAYER(engine_, Reduce, *input1,
                                                  reduce_operation, 28, true);
        layer = reduce_layer;
      } else {
        plugin::Pool3DPluginDynamic *plugin = new plugin::Pool3DPluginDynamic(
            ceil_mode, pool_type, adaptive, ksize, strides, paddings,
            global_pooling);
        layer = engine_->AddDynamicPlugin(&input1, 1, plugin);
      }
      auto output_name = op_desc.Output("Out")[0];
      layer->setName(("pool3d (Output: " + output_name + ")").c_str());
      layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(0));
      if (test_mode) {
        engine_->DeclareOutput(output_name);
      }
      return;
    }

    if (global_pooling == true) {
      auto *reduce_layer = TRT_ENGINE_ADD_LAYER(engine_, Reduce, *input1,
                                                reduce_operation, 14, true);
      layer = reduce_layer;
      auto output_name = op_desc.Output("Out")[0];
      layer->setName(("pool3d (Output: " + output_name + ")").c_str());
      layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(0));
      if (test_mode) {
        engine_->DeclareOutput(output_name);
      }
      return;
    }

    if (!adaptive) {
      if (!ceil_mode) {
        auto *pool_layer = TRT_ENGINE_ADD_LAYER(engine_, PoolingNd, *input1,
                                                nv_pool_type, nv_ksize);
        PADDLE_ENFORCE_NOT_NULL(
            pool_layer,
            platform::errors::Fatal(
                "trt pool layer in converter could not be created."));
        pool_layer->setStrideNd(nv_strides);
        pool_layer->setPaddingNd(nv_paddings);
        pool_layer->setAverageCountExcludesPadding(exclusive);
        layer = pool_layer;
      } else {
        std::vector<int> input_shape_v;
        for (int i = 0; i < input_dims; i++) {
          input_shape_v.push_back(input_shape.d[i]);
        }
        plugin::Pool3DPlugin *plugin =
            new plugin::Pool3DPlugin(ceil_mode, plugin_pool_type, adaptive,
                                     ksize, strides, paddings, input_shape_v);
        auto *pool_layer = engine_->AddPluginV2Ext(&input1, 1, plugin);
        PADDLE_ENFORCE_NOT_NULL(
            pool_layer,
            platform::errors::Fatal(
                "trt pool3d plugin layer in converter could not be created."));
        layer = pool_layer;
      }
    } else {
      // Average pooling needs to exclude the padding pixels from the average
      // mean.
      // It is not supported well by TRT, we use a plugin here.
      std::vector<int> input_shape_v;
      for (int i = 0; i < input_dims; i++) {
        input_shape_v.push_back(input_shape.d[i]);
      }
      plugin::Pool3DPlugin *plugin =
          new plugin::Pool3DPlugin(ceil_mode, plugin_pool_type, adaptive, ksize,
                                   strides, paddings, input_shape_v);
      auto *pool_layer = engine_->AddPluginV2Ext(&input1, 1, plugin);
      PADDLE_ENFORCE_NOT_NULL(
          pool_layer,
          platform::errors::Fatal(
              "trt pool3d plugin layer in converter could not be created."));
      layer = pool_layer;
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "pool3d", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP_ITSELF(pool3d);
REGISTER_TRT_OP_CONVERTER(pool3d, Pool3dOpConverter);
