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
      if (vec_##attr_name__.size() > 0) attr_name__ = vec_##attr_name__[0]; \
    }                                                                       \
  } while (0)

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
// we use tensorrt ScatterElement to generate set value
// set value operator is like:{
//  outputs = inputs
//  outputs[:,:,0:1] = updates
// }
// in particular, output = ScatterElement(inputs, indice, updates)
// in this op, indice location is calculatd using axes, start, end, step
// here is an example
// inputs shape is (2, 3, 4)
// inputs  = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
//          [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
// axes = 2
// start = 0
// end = 1
// step = 1
// the updates should be (2, 3, 1)
// updates = [[[2], [2], [2]],
//           [[2], [2], [2]]]
// the generated index shape should be the same as updates shape, which is (2,
// 3, 1) indices = [[[0], [0], [0]],
//           [[0], [0], [0]]]
// output = [[[2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]],
//          [[2, 1, 1, 1], [2, 1, 1, 1]], [2, 1, 1, 1]]

// For example, if indices has dimensions [N,C,H,W] and axis is 2, then the
// updates happen as: for n in [0,n)
//     for c in [0,n)
//         for h in [0,n)
//             for w in [0,n)
//                 output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]]
//
class SetValueConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a set value op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* updates = engine_->GetITensor(op_desc.Input("ValueTensor")[0]);

    int64_t axes = 0;
    int64_t starts = 0;
    int64_t steps = 1;
    int64_t ends = 0;

    GET_ATTR_FROM_VECTOR(axes);
    GET_ATTR_FROM_VECTOR(starts);
    GET_ATTR_FROM_VECTOR(steps);
    GET_ATTR_FROM_VECTOR(ends);

    // calculate dims
    auto input_dims = inputs->getDimensions();
    auto update_dims = updates->getDimensions();

    // check params and refill
    if (axes == -1) {
      axes = input_dims.nbDims - 1;
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

    // generate indice
    int post_size = 1;
    for (int j = axes + 1; j < update_dims.nbDims; ++j) {
      post_size = post_size * update_dims.d[j];
    }
    std::vector<int> axes_index;
    for (int i = starts; i < ends; i += steps) {
      for (int j = 0; j < post_size; ++j) {
        axes_index.emplace_back(i);
      }
    }
    int pre_size = 1;
    for (int i = 0; i < axes; ++i) {
      pre_size *= update_dims.d[i];
    }
    std::vector<int> indices;
    for (int i = 0; i < pre_size; ++i) {
      indices.insert(indices.end(), axes_index.begin(), axes_index.end());
    }

    nvinfer1::Dims indice_dims = update_dims;

    // create a tensor to store data
    std::vector<int> indice_dim_vec;
    for (int i = 0; i < update_dims.nbDims; i++) {
      indice_dim_vec.emplace_back(update_dims.d[i]);
    }
    auto indice_tensor_dims = phi::make_ddim(indice_dim_vec);
    std::unique_ptr<phi::DenseTensor> indice_tensor(new phi::DenseTensor());
    indice_tensor->Resize(indice_tensor_dims);
    auto* weight_data = indice_tensor->mutable_data<int>(platform::CPUPlace());

    memcpy(weight_data, indices.data(), sizeof(int) * indice_tensor->numel());
    TensorRTEngine::Weight weight{nvinfer1::DataType::kINT32,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(indice_tensor->numel())};
    engine_->SetWeights("set_value_index_op", std::move(indice_tensor));

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
