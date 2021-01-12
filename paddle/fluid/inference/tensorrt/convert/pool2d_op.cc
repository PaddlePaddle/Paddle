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
#include "paddle/fluid/inference/tensorrt/plugin/pool_op_plugin.h"

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
                         std::vector<int> paddings, nvinfer1::DimsHW *pre_pad,
                         nvinfer1::DimsHW *post_pad, int input_dims) {
  int input_height = input_shape.d[input_dims - 2];
  int input_width = input_shape.d[input_dims - 1];
  int floor_h_output_size =
      (input_height - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
  int ceil_h_output_size =
      (input_height - ksize[0] + 2 * paddings[0] + strides[0] - 1) /
          strides[0] +
      1;

  int floor_w_output_size =
      (input_width - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
  int ceil_w_output_size =
      (input_width - ksize[1] + 2 * paddings[1] + strides[1] - 1) / strides[1] +
      1;
  if (floor_h_output_size != ceil_h_output_size) {
    post_pad->h() = strides[0] - 1;
  }

  if (floor_w_output_size != ceil_w_output_size) {
    post_pad->w() = strides[1] - 1;
  }
}

/*
 * Pool2dOp, IPoolingLayer in TRT. This Layer doesn't has weights.
 */
class Pool2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc &op,
                  const framework::Scope &scope, bool test_mode) override {
    VLOG(4)
        << "convert a fluid pool2d op to tensorrt pool2d layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "TRT Pool2d expect 1 input, but got %d input.",
                          op_desc.Input("X").size()));
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "TRT Pool2d expect 1 Output, but got %d output.",
                          op_desc.Output("Out").size()));

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

    nvinfer1::PoolingType nv_pool_type = nvinfer1::PoolingType::kMAX;
    nvinfer1::ReduceOperation reduce_operation =
        nvinfer1::ReduceOperation::kMAX;
    plugin::PoolPlugin::PoolType plugin_pool_type =
        plugin::PoolPlugin::PoolType::max;
    if (pool_type == "max") {
      nv_pool_type = nvinfer1::PoolingType::kMAX;
      reduce_operation = nvinfer1::ReduceOperation::kMAX;
      plugin_pool_type = plugin::PoolPlugin::PoolType::max;
    } else if (pool_type == "avg") {
      nv_pool_type = nvinfer1::PoolingType::kAVERAGE;
      reduce_operation = nvinfer1::ReduceOperation::kAVG;
      plugin_pool_type = plugin::PoolPlugin::PoolType::avg;
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "Wrong pool op type, the trt do not support the %s pool type.",
          pool_type));
    }

    nvinfer1::DimsHW nv_ksize(ksize[0], ksize[1]);
    nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
    nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

    nvinfer1::ILayer *layer = nullptr;

    if (op_desc.HasAttr("enable_int8")) {
#if IS_TRT_VERSION_GE(5000)
      CHECK(op_desc.HasAttr("X_scale"));
      float input_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X_scale"));
      engine_->SetTensorDynamicRange(input1, input_scale);
#endif
    }

    if (engine_->with_dynamic_shape()) {
      if (!adaptive && !global_pooling && !ceil_mode) {
        auto *pool_layer = TRT_ENGINE_ADD_LAYER(engine_, Pooling, *input1,
                                                nv_pool_type, nv_ksize);
        pool_layer->setStride(nv_strides);
        pool_layer->setPadding(nv_paddings);
        pool_layer->setAverageCountExcludesPadding(exclusive);
        layer = pool_layer;
      } else if (global_pooling) {
        auto *reduce_layer = TRT_ENGINE_ADD_LAYER(engine_, Reduce, *input1,
                                                  reduce_operation, 12, true);
        layer = reduce_layer;
      } else {
#if IS_TRT_VERSION_GE(6000)
        plugin::PoolPluginDynamic *plugin =
            new plugin::PoolPluginDynamic(ceil_mode, pool_type, adaptive, ksize,
                                          strides, paddings, global_pooling);
        layer = engine_->AddPluginV2(&input1, 1, plugin);
#endif
      }
      auto output_name = op_desc.Output("Out")[0];
      layer->setName(("pool2d (Output: " + output_name + ")").c_str());
      layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(0));
      if (test_mode) {
        engine_->DeclareOutput(output_name);
      }
      return;
    }

    if (global_pooling == true) {
      nv_ksize.d[0] = input_shape.d[input_dims - 2];
      nv_ksize.d[1] = input_shape.d[input_dims - 1];
      auto *pool_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Pooling, *const_cast<nvinfer1::ITensor *>(input1),
          nv_pool_type, nv_ksize);
      PADDLE_ENFORCE_NOT_NULL(
          pool_layer, platform::errors::Fatal(
                          "trt pool layer in converter could not be created."));
      auto output_name = op_desc.Output("Out")[0];
      pool_layer->setStride(nv_strides);
      pool_layer->setPadding(nv_paddings);
      pool_layer->setAverageCountExcludesPadding(exclusive);
      pool_layer->setName(("pool2d (Output: " + output_name + ")").c_str());
      pool_layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, pool_layer->getOutput(0));
      layer = pool_layer;
      if (test_mode) {
        engine_->DeclareOutput(output_name);
      }
      return;
    }

    if (!adaptive) {
      // Under ceil mode, the pre_pad and post_pad are used to
      // record the the padding size. In some ceil mode cases,
      // we do not need padding, so we initialize the two vars to 0.

      nvinfer1::DimsHW pre_pad(0, 0);
      nvinfer1::DimsHW post_pad(0, 0);
      if (ceil_mode) {
        // If ceil mode is true, we will pad the appropriate size to the input.
        DealCeilMode(input_shape, ksize, strides, paddings, &pre_pad, &post_pad,
                     input_dims);
        auto *pad_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Padding, *const_cast<nvinfer1::ITensor *>(input1), pre_pad,
            post_pad);
        PADDLE_ENFORCE_NOT_NULL(
            pad_layer, platform::errors::Fatal(
                           "Pad layer in poolOp converter could not be "
                           "created. The pointer to pad layer is `NULL`."));
        input1 = pad_layer->getOutput(0);
      }
      auto *pool_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Pooling, *const_cast<nvinfer1::ITensor *>(input1),
          nv_pool_type, nv_ksize);
      PADDLE_ENFORCE_NOT_NULL(
          pool_layer, platform::errors::Fatal(
                          "trt pool layer in converter could not be created."));
      pool_layer->setStride(nv_strides);
      pool_layer->setPadding(nv_paddings);
      pool_layer->setAverageCountExcludesPadding(exclusive);
      layer = pool_layer;
    } else {
      // Average pooling needs to exclude the padding pixels from the average
      // mean.
      // It is not supported well by TRT, we use a plugin here.
      std::vector<int> input_shape_v;
      for (int i = 0; i < input_dims; i++) {
        input_shape_v.push_back(input_shape.d[i]);
      }
      plugin::PoolPlugin *plugin =
          new plugin::PoolPlugin(ceil_mode, plugin_pool_type, adaptive, ksize,
                                 strides, paddings, input_shape_v);
      auto *pool_layer = engine_->AddPlugin(&input1, 1, plugin);
      PADDLE_ENFORCE_NOT_NULL(
          pool_layer,
          platform::errors::Fatal(
              "trt pool plugin layer in converter could not be created."));
      layer = pool_layer;
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "pool2d", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(pool2d);
REGISTER_TRT_OP_CONVERTER(pool2d, Pool2dOpConverter);
