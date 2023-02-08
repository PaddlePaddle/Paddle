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

class SetValueConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a cast op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);

    // auto out_dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("out_dtype"));

    // get attributes
    std::vector<int64_t> v_axis;
    std::vector<int64_t> v_starts;
    std::vector<int64_t> v_steps;
    std::vector<int64_t> v_ends;
    int64_t axis = 0;
    int64_t starts = 0;
    int64_t steps = 1;
    int64_t ends = 0;
    if (op_desc.HasAttr("axes")) {
      v_axis = PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("axes"));
      if (v_axis.size() > 0) axis = v_axis[0];
    }
    if (op_desc.HasAttr("starts")) {
      v_starts =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("starts"));
      if (v_starts.size() > 0) starts = v_starts[0];
    }
    if (op_desc.HasAttr("ends")) {
      v_ends = PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("ends"));
      if (v_ends.size() > 0) ends = v_ends[0];
    }
    if (op_desc.HasAttr("steps")) {
      v_steps =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("steps"));
      if (v_steps.size() > 0) steps = v_steps[0];
    }

    // calculate dims
    auto input_dims = inputs->getDimensions();
    auto update_dims = updates->getDimensions();

    // check params and refill
    if (axis == -1) {
      axis = input_dims.nbDims - 1;
    }

    if (axis >= input_dims.nbDims) {
      platform::errors::InvalidArgument(
          "The axis %d is larger than total axis %d", axis, input_dims.nbDims);
    }
    if (starts >= input_dims.d[axis]) {
      platform::errors::InvalidArgument(
          "The start %d of dim %d is larger than origin shape %d",
          starts,
          axis,
          input_dims.d[axis]);
    }
    if (update_dims.d[axis] != (input_dims.d[axis] - starts) / steps) {
      platform::errors::InvalidArgument("The update dim error, should be %d",
                                        (input_dims.d[axis] - starts) / steps);
    }
    // generate indice
    // calculate generated index location
    int post_size = 1;
    for (int j = axis + 1; j < update_dims.nbDims; ++j) {
      post_size = post_size * update_dims.d[j];
    }
    std::vector<int> axis_index;
    for (int i = starts; i < ends; i += steps) {
      for (int j = 0; j < post_size; ++j) {
        axis_index.emplace_back(i);
      }
    }
    int pre_size = 1;
    for (int i = 0; i < axis; ++i) {
      pre_size *= update_dims.d[i];
    }
    std::vector<int> index;
    for (int i = 0; i < pre_size; ++i) {
      index.insert(index.end(), axis_index.begin(), axis_index.end());
    }

    nvinfer1::Dims indice_dims = update_dims;
    std::vector<int> vec_tmp_dims;
    for (int i = 0; i < update_dims.nbDims; i++) {
      vec_tmp_dims.push_back(update_dims.d[i]);
    }

    auto* dev_ctx = static_cast<phi::CPUContext*>(
        platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));
    auto tmp_dims = phi::make_ddim(vec_tmp_dims);
    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(tmp_dims);
    auto* weight_data = dev_ctx->template HostAlloc<int>(&tmp_tensor);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kINT32,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(indice_dims.nbDims)};

    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, indice_dims, weight.get());

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       Scatter,
                                       *inputs,
                                       *const_layer->getOutput(0),
                                       *updates,
                                       nvinfer1::ScatterMode::kELEMENT);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "set_value", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(set_value, SetValueConverter);
