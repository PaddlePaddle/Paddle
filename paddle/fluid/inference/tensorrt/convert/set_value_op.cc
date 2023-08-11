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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

#define GET_ATTR_FROM_VECTOR(attr_name__)                                   \
  do {                                                                      \
    std::vector<int64_t> vec_##attr_name__;                                 \
    if (op_desc.HasAttr(#attr_name__)) {                                    \
      vec_##attr_name__ = PADDLE_GET_CONST(std::vector<int64_t>,            \
                                           op_desc.GetAttr(#attr_name__));  \
      if (vec_##attr_name__.size() > 0) {attr_name__ = vec_##attr_name__[0]; \
      PADDLE_ENFORCE_EQ(vec_##attr_name__.size(), 1UL, platform::errors::InvalidArgument("attr axes/starst/ends/steps 's size in set_value must be one, but got %d", vec_##attr_name__.size()));  }      \
    }                                                                       \
  } while (0)

namespace paddle {
namespace inference {
namespace tensorrt {
// we use tensorrt ScatterElement to generate set value
// For example, if indices has dimensions [N,C,H,W] and axis is 2, then the
// updates happen as: for n in [0,n)
//     for c in [0,n)
//         for h in [0,n)
//             for w in [0,n)
//                 output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]]

class SetValueConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a set value op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::ITensor* updates;
    if (op_desc.Input("ValueTensor").size() > 0) {
      updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);
    } else {
      PADDLE_ENFORCE_EQ(PADDLE_GET_CONST(int, op_desc.GetAttr("dtype")), 5);
      float value = PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>, op_desc.GetAttr("values"))[0].to<int>();
      //std::cout << "属性values的值：" << value << std::endl;
      nvinfer1::Dims tmp_dim;
      tmp_dim.nbDims = inputs->getDimensions().nbDims;
      for (int i = 0; i < tmp_dim.nbDims; i++)
      tmp_dim.d[i] = 1;
      updates = AddConstantLayer(&value, tmp_dim);
    }

    nvinfer1::Dims tmp_dims = inputs->getDimensions();
    for (int i = 0; i < tmp_dims.nbDims; i++) {
      PADDLE_ENFORCE_GT(tmp_dims.d[i], 0);
      std::cout << "输入dims值：" << tmp_dims.d[i] << std::endl;
    }

    const auto decrease_axes = PADDLE_GET_CONST(
        std::vector<int64_t>, op_desc.GetAttr("decrease_axes"));
    std::vector<int32_t> decr_axes{decrease_axes.begin(), decrease_axes.end()};
    auto value_rank = updates->getDimensions().nbDims;
    auto input_rank = inputs->getDimensions().nbDims;

    std::cout << "输入名字：" << op_desc.Input("Input")[0] << std::endl;
    std::cout << "value_rank: "  << value_rank << std::endl;
    std::cout << "input_rank: " << input_rank  << std::endl;

    for (auto i : decrease_axes) {
      std::cout << "decrease_axes :" <<  i << std::endl;
    }

    if (decrease_axes.size() > 0 && value_rank != input_rank) {
      updates = Unsqueeze(updates, decr_axes);
    }
    
    // if still < input_rank, means we need broadcast!
    // value_rank = updates->getDimensions().nbDims;
    // if (value_rank < input_rank) {
    //   std::vector<int> axis (input_rank - value_rank, 0);
    //   std::iota(axis.begin(), axis.end(), 0);
    //   updates = Unsqueeze(updates, axis);
    // }

    

    tmp_dims = updates->getDimensions();
    for (int i = 0; i < tmp_dims.nbDims; i++) {
      PADDLE_ENFORCE_GT(tmp_dims.d[i], 0);
      std::cout << "updates dims值：" << tmp_dims.d[i] << std::endl;
    }

    int64_t axes = 0;
    int64_t starts = 0;
    int64_t steps = 1;
    int64_t ends = 0;

    GET_ATTR_FROM_VECTOR(axes);
    GET_ATTR_FROM_VECTOR(starts);
    GET_ATTR_FROM_VECTOR(steps);
    GET_ATTR_FROM_VECTOR(ends);

    std::cout << "axes" <<  axes  << std::endl;
    std::cout << "starts" <<  starts  << std::endl;
    std::cout << "steps" <<  steps<< std::endl;
    std::cout << "ends" << ends  << std::endl;

    // calculate dims
    auto input_dims = inputs->getDimensions();
    auto update_dims = updates->getDimensions();

    // check params and refill
    if (axes < 0) {
      axes += input_dims.nbDims;
    }

    if (ends < 0) {
      ends += input_dims.d[axes];
    }
    if (ends >= input_dims.d[axes]) {
      ends = input_dims.d[axes];
    }

    if (axes >= input_dims.nbDims) {
      platform::errors::InvalidArgument(
          "The axes %d is larger than total axes %d", axes, input_dims.nbDims);
    }
    if (starts >= input_dims.d[axes]) {
      platform::errors::InvalidArgument(
          "The start %d of dim %d is larger than origin shape %d",
          starts,
          axes,
          input_dims.d[axes]);
    }
    if (update_dims.d[axes] != (input_dims.d[axes] - starts) / steps) {
      platform::errors::InvalidArgument("The update dim error, should be %d",
                                        (input_dims.d[axes] - starts) / steps);
    }

    for(int i = 0; i < input_dims.nbDims; i++) {
      if (i != axes) {
        PADDLE_ENFORCE_EQ(input_dims.d[i], update_dims.d[i]);
      } 
    }

    if (engine_->with_dynamic_shape()) {
      // generate indice
      int post_size = 1;
      for (int j = axes + 1; j < update_dims.nbDims; ++j) {
        post_size = post_size * update_dims.d[j];
      }

      int pre_size = 1;
      for (int i = 0; i < axes; ++i) {
        pre_size *= update_dims.d[i];
      }

      std::vector<int> indices;
      for (int i = 0; i < pre_size; i++) {
        for (int j = starts; j < ends; j += steps) {
          for (int k = 0; k < post_size; k++) {
            indices.push_back(j);
          }
        }
      }


      const auto const_layer = AddConstantLayer(
          indices.data(), update_dims, "set_value_index_" + output_name);

      auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         Scatter,
                                         *inputs,
                                         *const_layer,
                                         *updates,
                                         nvinfer1::ScatterMode::kELEMENT);

      layer->setAxis(axes);

      RreplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
     // std::cout << output_name << std::endl;
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "static shape mode not supported in set value yet"));
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(set_value, SetValueConverter);
