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

class StridedSliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert strided_slice op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto output_name = op_desc.Output("Out")[0];

    // phi only allow axes[i] >= 0 && <rank, so we need not deal with minus
    // axes[i]
    std::vector<int> axes =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    std::vector<int> strides =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    std::vector<int> decrease_axises =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("decrease_axis"));

    auto input_dims = input->getDimensions();
    if (!engine_->with_dynamic_shape()) {
      // notice that input shape is [CHW] without batch axis when input has
      // static shape
      for (size_t i = input_dims.nbDims; i > 0; i--) {
        input_dims.d[i] = input_dims.d[i - 1];
      }
      input_dims.d[0] = 1;  // fake batchsize, not useful here
      for (size_t i = 0; i < axes.size(); i++) {
        if (starts[i] < 0) {
          starts[i] = std::max(starts[i] + input_dims.d[axes[i]], 0);
        }
        if (ends[i] < 0) {
          ends[i] = std::max(ends[i] + input_dims.d[axes[i]], 0);
        }
        ends[i] = std::min(ends[i], input_dims.d[axes[i]]);
        PADDLE_ENFORCE_GT(
            ends[i],
            starts[i],
            platform::errors::InvalidArgument(
                "Attr(ends) should be greater than attr(starts) in "
                "slice op. But received ends = %d, starts = %d.",
                ends[i],
                starts[i]));
      }
    }

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      auto nchw_input_dims = input->getDimensions();
      nvinfer1::Dims trt_start_dims;
      trt_start_dims.nbDims = nchw_input_dims.nbDims;
      memset(trt_start_dims.d, 0, sizeof(int32_t) * nchw_input_dims.nbDims);
      nvinfer1::Dims trt_size_dims = trt_start_dims;
      nvinfer1::Dims trt_end_dims = trt_start_dims;
      nvinfer1::Dims trt_step_dims = trt_start_dims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;
      // input : [N,C,H,W]
      bool has_neg_indices = false;
      for (size_t i = 0; i < axes.size(); i++) {
        int trt_axis = axes[i];
        trt_start_dims.d[trt_axis] = starts[i];
        trt_end_dims.d[trt_axis] = ends[i];
        trt_step_dims.d[axes[i]] = strides[i];
        if (starts[i] < 0 || ends[i] < 0) has_neg_indices = true;
      }
      auto* shape_tensor = Shape(input);
      auto* start_tensor = Add1DConstantLayer(trt_start_dims);
      if (has_neg_indices) {
        start_tensor = FixNegIndices(shape_tensor, start_tensor);
      }

      std::vector<nvinfer1::ITensor*> end_vec_tensor;
      for (int i = 0; i < trt_end_dims.nbDims; i++) {
        end_vec_tensor.push_back(GetEleTensorOfShape(shape_tensor, i));
      }

      for (size_t i = 0; i < axes.size(); i++) {
        int trt_axis = axes[i];
        if (ends[i] >= 0) {
          end_vec_tensor[trt_axis] = Add1DConstantLayer(ends[i]);
        } else {
          end_vec_tensor[trt_axis] =
              Sum(end_vec_tensor[trt_axis], Add1DConstantLayer(ends[i]));
        }
      }

      auto* size_tensor =
          Sub(start_tensor, Min(Concat(end_vec_tensor), shape_tensor));
      std::vector<int> tmp_zero_vec(size_tensor->getDimensions().nbDims, 0);
      auto zero_t = Add1DConstantLayer(tmp_zero_vec);
      auto step_tensor = Add1DConstantLayer(trt_step_dims);

      size_tensor = Sub(zero_t, FloorDiv(size_tensor, step_tensor));

      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *input, trt_start_dims, trt_size_dims, trt_step_dims);
      layer->setInput(1, *start_tensor);
      layer->setInput(2, *size_tensor);
      layer->setInput(3, *step_tensor);

      if (decrease_axises.size() > 0) {
        std::vector<int32_t> gather_indices;
        for (int i = 0; i < trt_size_dims.nbDims; i++) {
          if (decrease_axises.end() !=
              std::find(decrease_axises.begin(), decrease_axises.end(), i))
            continue;
          gather_indices.push_back(i);
        }
        if (gather_indices.empty())
          gather_indices.push_back(decrease_axises[0]);
        auto real_size_tensor = Gather(size_tensor, gather_indices);
        layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
        layer->setInput(1, *real_size_tensor);
      }
    } else {
      auto chw_input_dims = input->getDimensions();
      nvinfer1::Dims trt_start_dims;
      trt_start_dims.nbDims = chw_input_dims.nbDims;
      memset(trt_start_dims.d, 0, sizeof(int32_t) * chw_input_dims.nbDims);
      nvinfer1::Dims trt_size_dims = chw_input_dims;
      nvinfer1::Dims trt_step_dims;
      trt_step_dims.nbDims = chw_input_dims.nbDims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

      // input : [C,H,W]
      for (size_t i = 0; i < axes.size(); i++) {
        int trt_axis = axes[i] - 1;
        trt_start_dims.d[trt_axis] = starts[i];
        trt_size_dims.d[trt_axis] =
            (ends[i] - starts[i] + strides[i] - 1) / strides[i];
        trt_step_dims.d[trt_axis] = strides[i];
      }
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *input, trt_start_dims, trt_size_dims, trt_step_dims);
      nvinfer1::Dims real_trt_size_dims;
      real_trt_size_dims.nbDims = 0;

      if (decrease_axises.size() > 0) {
        for (size_t i = 0; i < decrease_axises.size(); i++) {
          decrease_axises[i]--;
        }
        for (int i = 0; i < trt_size_dims.nbDims; i++) {
          if (decrease_axises.end() !=
              std::find(decrease_axises.begin(), decrease_axises.end(), i))
            continue;
          real_trt_size_dims.d[real_trt_size_dims.nbDims] = trt_size_dims.d[i];
          real_trt_size_dims.nbDims++;
        }
        if (real_trt_size_dims.nbDims == 0) {
          real_trt_size_dims.nbDims = 1;
          real_trt_size_dims.d[0] = 1;
        }
        auto reshape_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
        reshape_layer->setReshapeDimensions(real_trt_size_dims);
        layer = static_cast<nvinfer1::ILayer*>(reshape_layer);
      }
    }
    RreplenishLayerAndOutput(layer, "strided_slice", {output_name}, test_mode);

    std::cout << layer->getOutput(0)->getDimensions().d[0] << std::endl;
    std::cout << layer->getOutput(0)->getDimensions().d[1] << std::endl;
    std::cout << layer->getOutput(0)->getDimensions().nbDims << std::endl;
    std::cout << output_name << std::endl;
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(strided_slice, StridedSliceOpConverter);
