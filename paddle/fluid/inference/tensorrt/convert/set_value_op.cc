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

#define GET_ATTR_FROM_VECTOR(attr_name__)                                  \
  do {                                                                     \
    std::vector<int64_t> vec_##attr_name__;                                \
    if (op_desc.HasAttr(#attr_name__)) {                                   \
      vec_##attr_name__ = PADDLE_GET_CONST(std::vector<int64_t>,           \
                                           op_desc.GetAttr(#attr_name__)); \
      if (vec_##attr_name__.size() > 0) {                                  \
        attr_name__ = vec_##attr_name__[0];                                \
        PADDLE_ENFORCE_EQ(vec_##attr_name__.size(),                        \
                          1UL,                                             \
                          platform::errors::InvalidArgument(               \
                              "attr axes/starst/ends/steps 's size in "    \
                              "set_value must be one, but got %d",         \
                              vec_##attr_name__.size()));                  \
      }                                                                    \
    }                                                                      \
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
      float value = PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>,
                                     op_desc.GetAttr("values"))[0]
                        .to<int>();
      VLOG(3) << "the attribute value is: " << value;
      nvinfer1::Dims tmp_dim;
      tmp_dim.nbDims = inputs->getDimensions().nbDims;
      for (int i = 0; i < tmp_dim.nbDims; i++) tmp_dim.d[i] = 1;
      updates = AddConstantLayer(&value, tmp_dim);
    }

    // for debug
    {
      nvinfer1::Dims tmp_dims = inputs->getDimensions();
      std::vector<int> tmp_vec;
      for (int i = 0; i < tmp_dims.nbDims; i++)
        tmp_vec.push_back(tmp_dims.d[i]);
      VLOG(3) << "Input(Name:" << op_desc.Input("Input")[0] << ")"
              << "'s dimension is :[" << string::join_strings(tmp_vec, ',')
              << "]";

      tmp_vec.clear();
      tmp_dims = updates->getDimensions();
      for (int i = 0; i < tmp_dims.nbDims; i++)
        tmp_vec.push_back(tmp_dims.d[i]);
      VLOG(3) << "updates tensor"
              << "'s dimension is :[" << string::join_strings(tmp_vec, ',')
              << "]";
    }

    const auto decrease_axes = PADDLE_GET_CONST(
        std::vector<int64_t>, op_desc.GetAttr("decrease_axes"));
    std::vector<int32_t> decr_axes{decrease_axes.begin(), decrease_axes.end()};
    auto value_rank = updates->getDimensions().nbDims;
    auto input_rank = inputs->getDimensions().nbDims;
    VLOG(3) << "decrease_axes is: [" << string::join_strings(decrease_axes, ',')
            << "]";

    if (decrease_axes.size() > 0 && value_rank != input_rank) {
      updates = Unsqueeze(updates, decr_axes);
    }

    PADDLE_ENFORCE_EQ(
        updates->getDimensions().nbDims,
        input_rank,
        platform::errors::InvalidArgument(
            "ValueTensor‘s rank not equal to Input's rank, "
            "you should try use C++ API "
            "config.exp_disable_tensorrt_ops({\"%s\"}) to forbind this op "
            "enter into TRT, "
            "please find the %s's real name from .pdmodel or shape.txt",
            output_name,
            output_name));

    // if still < input_rank, means we need broadcast!
    // value_rank = updates->getDimensions().nbDims;
    // if (value_rank < input_rank) {
    //   std::vector<int> axis (input_rank - value_rank, 0);
    //   std::iota(axis.begin(), axis.end(), 0);
    //   updates = Unsqueeze(updates, axis);
    // }

    // for debug
    {
      auto tmp_dims = updates->getDimensions();
      std::vector<int> tmp_vec;
      tmp_vec.clear();
      tmp_dims = updates->getDimensions();
      for (int i = 0; i < tmp_dims.nbDims; i++)
        tmp_vec.push_back(tmp_dims.d[i]);
      VLOG(3) << "updates tensor"
              << "'s dimension is :[" << string::join_strings(tmp_vec, ',')
              << "]";
    }

    int64_t axes = 0;
    int64_t starts = 0;
    int64_t steps = 1;
    int64_t ends = 0;

    GET_ATTR_FROM_VECTOR(axes);
    GET_ATTR_FROM_VECTOR(starts);
    GET_ATTR_FROM_VECTOR(steps);
    GET_ATTR_FROM_VECTOR(ends);

    VLOG(3) << "axes is: " << axes;
    VLOG(3) << "starts is: " << starts;
    VLOG(3) << "steps is: " << steps;
    VLOG(3) << "ends is: " << ends;

    // calculate dims
    auto input_dims = inputs->getDimensions();
    auto update_dims = updates->getDimensions();

    PADDLE_ENFORCE_GT(
        input_dims.d[axes],
        0,
        platform::errors::InvalidArgument(
            "the input_dims.d[%d] must be greater than 0, but received %d",
            axes,
            input_dims.d[axes]));

    PADDLE_ENFORCE_GT(
        update_dims.d[axes],
        0,
        platform::errors::InvalidArgument(
            "the update_dims.d[%d] must be greater than 0, but received %d",
            axes,
            update_dims.d[axes]));

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

    PADDLE_ENFORCE_LE(axes,
                      input_dims.nbDims,
                      platform::errors::InvalidArgument(
                          "The axes %d is larger than total axes %d",
                          axes,
                          input_dims.nbDims));

    PADDLE_ENFORCE_LE(
        starts,
        input_dims.d[axes],
        platform::errors::InvalidArgument(
            "The start %d of dim %d is larger than origin shape %d",
            starts,
            axes,
            input_dims.d[axes]));

    PADDLE_ENFORCE_EQ(
        update_dims.d[axes],
        (ends - starts) / steps,
        platform::errors::InvalidArgument(
            "the %dth axis of update dim error, should be %d, but we got %d",
            axes,
            (ends - starts) / steps,
            update_dims.d[axes]));

    for (int i = 0; i < input_dims.nbDims; i++) {
      if (i != axes) {
        // PADDLE_ENFORCE_EQ(input_dims.d[i], update_dims.d[i]);
      }
    }

    if (engine_->with_dynamic_shape()) {
      nvinfer1::Dims shape_0;
      shape_0.nbDims = update_dims.nbDims;
      for (int i = 0; i < shape_0.nbDims; ++i) {
        shape_0.d[i] = 1;
      }
      std::vector<float> tmp_0(1, 0);
      auto zero_tensor = AddConstantLayer(tmp_0.data(), shape_0);
      auto indice_tensor = Prod(zero_tensor, updates);
      auto cast_layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *indice_tensor);
      cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
      indice_tensor = cast_layer->getOutput(0);

      nvinfer1::Dims shape_1;
      shape_1.nbDims = update_dims.nbDims;
      for (int i = 0; i < update_dims.nbDims; ++i) {
        shape_1.d[i] = 1;
      }
      shape_1.d[axes] = update_dims.d[axes];
      std::vector<int> tmp_1;
      for (int i = starts; i < ends; i += steps) {
        tmp_1.push_back(i);
      }
      auto one_tensor = AddConstantLayer(tmp_1.data(), shape_1);
      indice_tensor = Sum(indice_tensor, one_tensor);

      auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         Scatter,
                                         *inputs,
                                         *indice_tensor,
                                         *updates,
                                         nvinfer1::ScatterMode::kELEMENT);

      layer->setAxis(axes);

      RreplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
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
