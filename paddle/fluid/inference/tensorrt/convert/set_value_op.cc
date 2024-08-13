/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#define USE_INPUT(input_name_)                                     \
  op_desc.Inputs().find(#input_name_) != op_desc.Inputs().end() && \
      !op_desc.Input(#input_name_).empty()

#define GENERAL_GET_ATTR(input_name_, attr_name_)                             \
  std::vector<nvinfer1::ITensor*> itensors_##attr_name_;                      \
  std::unordered_map<size_t, int> w_##attr_name_;                             \
  std::vector<int64_t> vec_##attr_name_;                                      \
  if (USE_INPUT(input_name_)) {                                               \
    auto input_names = op_desc.Input(#input_name_);                           \
    PADDLE_ENFORCE_EQ(input_names.size(),                                     \
                      num_axes,                                               \
                      phi::errors::InvalidArgument(                           \
                          "size of %s[%d] must to size of axes[%d]",          \
                          #input_name_,                                       \
                          itensors_##attr_name_.size(),                       \
                          num_axes));                                         \
    for (size_t i = 0; i < num_axes; ++i) {                                   \
      itensors_##attr_name_.push_back(engine_->GetITensor(input_names[i]));   \
      auto* var = scope.FindVar(input_names[i]);                              \
      if (var == nullptr) continue;                                           \
      auto* dense_tensor = var->GetMutable<phi::DenseTensor>();               \
      w_##attr_name_[i] =                                                     \
          *dense_tensor->data<int>() < 0                                      \
              ? *dense_tensor->data<int>() + input_dims.d[axes[i]]            \
              : *dense_tensor->data<int>();                                   \
      w_##attr_name_[i] = std::min(w_##attr_name_[i], input_dims.d[axes[i]]); \
    }                                                                         \
  } else {                                                                    \
    vec_##attr_name_ =                                                        \
        PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr(#attr_name_)); \
    PADDLE_ENFORCE_EQ(vec_##attr_name_.size(),                                \
                      num_axes,                                               \
                      phi::errors::InvalidArgument(                           \
                          "size of %s[%d] must equal to size of axes[%d]",    \
                          #attr_name_,                                        \
                          vec_##attr_name_.size(),                            \
                          num_axes));                                         \
    for (size_t i = 0; i < num_axes; ++i) {                                   \
      vec_##attr_name_[i] = vec_##attr_name_[i] < 0                           \
                                ? vec_##attr_name_[i] + input_dims.d[axes[i]] \
                                : vec_##attr_name_[i];                        \
      vec_##attr_name_[i] = std::min(static_cast<int>(vec_##attr_name_[i]),   \
                                     input_dims.d[axes[i]]);                  \
    }                                                                         \
  }

#define GET_VALUE(input_name_, attr_name_, default_, i_)           \
  nvinfer1::ITensor* itensor_##attr_name_ = nullptr;               \
  int val_##attr_name_ = default_;                                 \
  if (std::find(axes.begin(), axes.end(), i_) != axes.end()) {     \
    int j_ = i_ - axes[0];                                         \
    if (USE_INPUT(input_name_)) {                                  \
      if (w_##attr_name_.find(j_) != w_##attr_name_.end()) {       \
        val_##attr_name_ = w_##attr_name_[j_];                     \
      } else {                                                     \
        itensor_##attr_name_ = itensors_##attr_name_[j_];          \
      }                                                            \
    } else {                                                       \
      val_##attr_name_ = static_cast<int>(vec_##attr_name_[j_]);   \
    }                                                              \
  }                                                                \
  if (val_##attr_name_ < 0) {                                      \
    itensor_##attr_name_ = GetEleTensorOfShape(Shape(inputs), i_); \
  }

namespace paddle {
namespace inference {
namespace tensorrt {

class SetValueConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a set value op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);
    auto axes = PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("axes"));
    auto output_name = op_desc.Output("Out")[0];

    if (axes.empty()) {
      auto* value = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);
      auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *value);
      ReplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
    } else {
      auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
      auto input_dims = inputs->getDimensions();
      for (auto& axis : axes) {
        if (axis < 0) axis += input_dims.nbDims;
      }
      std::sort(axes.begin(), axes.end());
      size_t num_axes = axes.size();
      GENERAL_GET_ATTR(StartsTensorList, starts);
      GENERAL_GET_ATTR(EndsTensorList, ends);
      GENERAL_GET_ATTR(StepsTensorList, steps);

      std::vector<nvinfer1::ITensor*> indices_vec;
      std::vector<nvinfer1::ITensor*> indices_length;
      size_t num_input_dims = input_dims.nbDims;
      for (size_t i = 0; i < num_input_dims; ++i) {
        GET_VALUE(StartsTensorList, starts, 0, i);
        GET_VALUE(EndsTensorList, ends, input_dims.d[i], i);
        GET_VALUE(StepsTensorList, steps, 1, i);

        PADDLE_ENFORCE_NE(val_steps, 0, "set_value_op step can not be zero");

        if (itensor_starts || itensor_ends || itensor_steps) {
          // TODO(ming1753): StartsTensorList or EndsTensorList may be less than
          // 0 or greater than dims.d[i], not support now.
          if (itensor_starts == nullptr) {
            itensor_starts =
                Add1DConstantLayer<int>(val_starts, output_name + "_starts");
          }
          if (itensor_ends == nullptr) {
            val_ends = std::min(static_cast<int>(input_dims.d[i]), val_ends);
            itensor_ends =
                Add1DConstantLayer<int>(val_ends, output_name + "_ends");
          }
          if (itensor_steps == nullptr) {
            itensor_steps =
                Add1DConstantLayer<int>(val_steps, output_name + "_steps");
          }
          auto* sub_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *itensor_ends,
                                   *itensor_starts,
                                   nvinfer1::ElementWiseOperation::kSUB);
          auto* floor_div_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *(sub_layer->getOutput(0)),
                                   *itensor_steps,
                                   nvinfer1::ElementWiseOperation::kFLOOR_DIV);

          indices_length.push_back(floor_div_layer->getOutput(0));

          auto* fill_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Fill,
                                   nvinfer1::Dims{},
                                   nvinfer1::FillOperation::kLINSPACE);
          fill_layer->setInput(0, *floor_div_layer->getOutput(0));
          fill_layer->setInput(
              1,
              *Add1DConstantLayer<int>(
                  val_starts, output_name + "_starts_scalar", true));
          fill_layer->setInput(2, *itensor_steps);

          auto* unsqueeze_layer = TRT_ENGINE_ADD_LAYER(
              engine_, Shuffle, *(fill_layer->getOutput(0)));
          nvinfer1::Dims tmp_indices_dims;
          tmp_indices_dims.nbDims = static_cast<int>(num_input_dims + 1);
          for (int j = 0; j < tmp_indices_dims.nbDims; ++j) {
            if (j == static_cast<int>(i)) {
              tmp_indices_dims.d[j] = -1;
            } else {
              tmp_indices_dims.d[j] = 1;
            }
          }
          unsqueeze_layer->setReshapeDimensions(tmp_indices_dims);
          indices_vec.push_back(unsqueeze_layer->getOutput(0));
        } else {
          std::vector<int> sequence;
          for (int j = val_starts; j < val_ends; j += val_steps) {
            sequence.push_back(j);
          }
          PADDLE_ENFORCE_GE(
              sequence.size(),
              0,
              "The length of the part to be set value must be greater than 0.");
          indices_length.push_back(
              Add1DConstantLayer<int>(static_cast<int>(sequence.size())));
          nvinfer1::Dims tmp_indices_dims;
          tmp_indices_dims.nbDims = static_cast<int>(num_input_dims + 1);
          for (int j = 0; j < tmp_indices_dims.nbDims; ++j) {
            if (j == static_cast<int>(i)) {
              tmp_indices_dims.d[j] = static_cast<int>(sequence.size());
            } else {
              tmp_indices_dims.d[j] = 1;
            }
          }
          auto tmp_indices = AddConstantLayer(sequence.data(),
                                              tmp_indices_dims,
                                              "indices_" + std::to_string(i));
          indices_vec.push_back(tmp_indices);
        }
      }

      std::vector<nvinfer1::ITensor*> indices_before_stack;
      auto one_tensor = Add1DConstantLayer<int>(1, output_name + "_one_tensor");
      for (size_t i = 0; i < num_input_dims; ++i) {
        std::vector<nvinfer1::ITensor*> tmp_indices_length = indices_length;
        tmp_indices_length[i] = one_tensor;
        tmp_indices_length.push_back(one_tensor);
        auto* concat_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                  Concatenation,
                                                  tmp_indices_length.data(),
                                                  tmp_indices_length.size());

        auto* fill_layer =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 Fill,
                                 nvinfer1::Dims{},
                                 nvinfer1::FillOperation::kLINSPACE);

        fill_layer->setInput(0, *(concat_layer->getOutput(0)));
        fill_layer->setInput(1, *Add1DConstantLayer<int>(0, "", true));
        fill_layer->setInput(
            2, *Add1DConstantLayer(std::vector<int>(num_input_dims + 1, 0)));

        auto* add_layer =
            TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *indices_vec[i],
                                 *fill_layer->getOutput(0),
                                 nvinfer1::ElementWiseOperation::kSUM);
        indices_before_stack.push_back(add_layer->getOutput(0));
      }

      auto* stack_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                               Concatenation,
                                               indices_before_stack.data(),
                                               indices_before_stack.size());
      stack_layer->setAxis(static_cast<int>(num_input_dims));

      nvinfer1::ITensor* indices = stack_layer->getOutput(0);

      nvinfer1::Dims indices_shape;
      indices_shape.nbDims = 2;
      indices_shape.d[0] = -1;
      indices_shape.d[1] = static_cast<int>(num_input_dims);
      indices = Reshape(indices, indices_shape);

      nvinfer1::ITensor* updates = nullptr;
      int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
      if (USE_INPUT(ValueTensor)) {
        updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);
        nvinfer1::Dims updates_shape;
        updates_shape.nbDims = 1;
        updates_shape.d[0] = -1;
        updates = Reshape(updates, updates_shape);
      } else {
        nvinfer1::ITensor* indices_shape_tensor = Shape(indices);
        auto* indices_count_tensor =
            GetEleTensorOfShape(indices_shape_tensor, 0);
        if (dtype == 2) {
          int value =
              PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>,
                               op_desc.GetAttr("values"))[0]
                  .to<int>();
          auto fill_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Fill,
                                   nvinfer1::Dims{},
                                   nvinfer1::FillOperation::kLINSPACE);
          fill_layer->setInput(0, *indices_count_tensor);
          fill_layer->setInput(1, *Add1DConstantLayer(value, "", true));
          fill_layer->setInput(2, *Add1DConstantLayer(std::vector<int>(1, 0)));
          updates = fill_layer->getOutput(0);
        } else {
          float value;
          if (dtype == 0) {
            value = static_cast<float>(
                PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>,
                                 op_desc.GetAttr("values"))[0]
                    .to<bool>());
          } else if (dtype == 5) {
            value = PADDLE_GET_CONST(std::vector<paddle::experimental::Scalar>,
                                     op_desc.GetAttr("values"))[0]
                        .to<float>();
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "set_value only supports 'bool', 'int32', and 'float"));
          }
          auto fill_layer =
              TRT_ENGINE_ADD_LAYER(engine_,
                                   Fill,
                                   nvinfer1::Dims{},
                                   nvinfer1::FillOperation::kLINSPACE);
          fill_layer->setInput(0, *indices_count_tensor);
          fill_layer->setInput(1, *Add1DConstantLayer(value, "", true));
          fill_layer->setInput(2,
                               *Add1DConstantLayer(std::vector<float>(1, 0.)));
          updates = fill_layer->getOutput(0);
        }
      }

      if (dtype == 0) {
        inputs = Cast(inputs, nvinfer1::DataType::kFLOAT);
      }

      auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                        Scatter,
                                        *inputs,
                                        *indices,
                                        *updates,
                                        nvinfer1::ScatterMode::kND);

      if (dtype == 0) {
        auto* cast_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Identity, *layer->getOutput(0));
        cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);
        cast_layer->getOutput(0)->setType(nvinfer1::DataType::kBOOL);
        ReplenishLayerAndOutput(
            cast_layer, "where_index", {output_name}, test_mode);
      } else {
        ReplenishLayerAndOutput(layer, "where_index", {output_name}, test_mode);
      }
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(set_value, SetValueConverter);
