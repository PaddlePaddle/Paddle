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
                          common::errors::InvalidArgument(                 \
                              "attr axes/starts/ends/steps 's size in "    \
                              "set_value must be one, but got %d",         \
                              vec_##attr_name__.size()));                  \
      }                                                                    \
    }                                                                      \
  } while (0)

namespace paddle::inference::tensorrt {
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

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);

    auto input_dims = inputs->getDimensions();

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

    VLOG(3) << "after standardization" << axes;
    VLOG(3) << "axes is: " << axes;
    VLOG(3) << "starts is: " << starts;
    VLOG(3) << "steps is: " << steps;
    VLOG(3) << "ends is: " << ends;

    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::ITensor* updates;
    if (op_desc.HasInput("ValueTensor") &&
        op_desc.Input("ValueTensor").size() > 0) {
      updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);
    } else {
      int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
      PADDLE_ENFORCE_EQ(
          dtype,
          5,
          common::errors::InvalidArgument("set_value OP dtype must be float"));
      float value = PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>,
                                     op_desc.GetAttr("values"))[0]
                        .to<float>();
      VLOG(3) << "the attribute value is: " << value;

      nvinfer1::ITensor* input_shape_tensor = Shape(inputs);
      std::vector<nvinfer1::ITensor*> vec_tensor;
      for (int32_t i = 0; i < input_dims.nbDims; ++i) {
        vec_tensor.push_back(GetEleTensorOfShape(input_shape_tensor, i));
      }
      std::vector<int32_t> axes_vec(1, (ends - 1 - starts) / steps + 1);
      vec_tensor[axes] = Add1DConstantLayer(axes_vec, "axes_vec", false);
      nvinfer1::ITensor* output_shape_tensor = Concat(vec_tensor, 0);
      updates = FillConstantLayer(
          output_shape_tensor, inputs->getDimensions().nbDims, value);
    }

    // for log
    {
      std::vector<int> tmp_vec;
      for (int i = 0; i < input_dims.nbDims; i++)
        tmp_vec.push_back(input_dims.d[i]);
      VLOG(3) << "Input(Name:" << op_desc.Input("Input")[0] << ")"
              << "'s dimension is :[" << string::join_strings(tmp_vec, ',')
              << "]";

      tmp_vec.clear();
      nvinfer1::Dims tmp_dims = updates->getDimensions();
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
    // GLOG_vmodule=op_teller=6
    VLOG(3) << "decrease_axes is: [" << string::join_strings(decrease_axes, ',')
            << "]";

    if (decrease_axes.size() > 0 && value_rank != input_rank) {
      updates = Unsqueeze(updates, decr_axes);
    }

    PADDLE_ENFORCE_EQ(
        updates->getDimensions().nbDims,
        input_rank,
        common::errors::InvalidArgument(
            "ValueTensor‘s rank not equal to Input's rank, "
            "you should try use C++ API "
            "config.exp_disable_tensorrt_ops({\"%s\"}) to forbid this op "
            "enter into TRT, "
            "please find the %s's real name from .pdmodel or shape.txt",
            output_name,
            output_name));

    // for log
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

    // calculate dims
    auto update_dims = updates->getDimensions();

    PADDLE_ENFORCE_GT(
        input_dims.d[axes],
        0,
        common::errors::InvalidArgument(
            "the input_dims.d[%d] must be greater than 0, but received %d",
            axes,
            input_dims.d[axes]));

    PADDLE_ENFORCE_GT(
        update_dims.d[axes],
        0,
        common::errors::InvalidArgument(
            "the update_dims.d[%d] must be greater than 0, but received %d",
            axes,
            update_dims.d[axes]));

    PADDLE_ENFORCE_LE(axes,
                      input_dims.nbDims,
                      common::errors::InvalidArgument(
                          "The axes %d is larger than total axes %d",
                          axes,
                          input_dims.nbDims));

    PADDLE_ENFORCE_LE(
        starts,
        input_dims.d[axes],
        common::errors::InvalidArgument(
            "The start %d of dim %d is larger than origin shape %d",
            starts,
            axes,
            input_dims.d[axes]));

    PADDLE_ENFORCE_EQ(
        update_dims.d[axes],
        (ends - 1 - starts) / steps + 1,
        common::errors::InvalidArgument(
            "the %dth axis of update dim error, should be %d, but we got %d",
            axes,
            (ends - 1 - starts) / steps + 1,
            update_dims.d[axes]));

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

      ReplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "static shape mode not supported in set value yet"));
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(set_value, SetValueConverter);
