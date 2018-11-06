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

bool to_skip_merging_optimize(TensorRTEngine* engine_,
                              const std::vector<int>& filters,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              std::string input_name) {
  if (engine_->itensor_quote_num[input_name] > 0) {
    return true;
  }
  if (filters[0] == 1 && filters[1] == 1 && strides[0] == 1 &&
      strides[1] == 1 && paddings[0] == 0 && paddings[1] == 0)
    engine_->itensor_quote_num[input_name] += 1;

  return false;
}

class Conv2dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    LOG(INFO)
        << "convert a fluid conv2d op to tensorrt conv layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Filter").size(), 1);  // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Output").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("Input").front());

    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input("Filter").front());
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> weight_tensor(
        new framework::LoDTensor());
    weight_tensor->Resize(Y_t->dims());
    TensorCopySync((*Y_t), cpu_place, weight_tensor.get());

    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());

    PADDLE_ENFORCE_EQ(weight_tensor->dims().size(), 4UL);
    const int n_output = weight_tensor->dims()[0];
    const int filter_h = weight_tensor->dims()[2];
    const int filter_w = weight_tensor->dims()[3];

    const int groups = boost::get<int>(op_desc.GetAttr("groups"));
    const std::vector<int> dilations =
        boost::get<std::vector<int>>(op_desc.GetAttr("dilations"));
    const std::vector<int> strides =
        boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
    const std::vector<int> paddings =
        boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));

    nvinfer1::DimsHW nv_ksize(filter_h, filter_w);
    nvinfer1::DimsHW nv_dilations(dilations[0], dilations[1]);
    nvinfer1::DimsHW nv_strides(strides[0], strides[1]);
    nvinfer1::DimsHW nv_paddings(paddings[0], paddings[1]);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  weight_tensor->memory_size() / sizeof(float)};

    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Convolution, *const_cast<nvinfer1::ITensor*>(X), n_output,
        nv_ksize, weight.get(), bias.get());
    PADDLE_ENFORCE(layer != nullptr);
    layer->setStride(nv_strides);
    layer->setPadding(nv_paddings);
    layer->setDilation(nv_dilations);
    layer->setNbGroups(groups);

    auto output_name = op_desc.Output("Output").front();
    layer->setName(("conv2d (Output: " + output_name + ")").c_str());
    engine_->weight_map[op_desc.Input("Filter").front()] =
        std::move(weight_tensor);
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->SetITensor(output_name, layer->getOutput(0));

    if (test_mode ||
        to_skip_merging_optimize(engine_, {filter_h, filter_w}, strides,
                                 paddings, op_desc.Input("Input").front())) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(conv2d, Conv2dOpConverter);
