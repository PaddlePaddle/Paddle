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

#include <vector>
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

class MultiClassNMSOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid multiclassNMS op to tensorrt plugin";

    // for now, only work for static shape and regular tensor
    framework::OpDesc op_desc(op, nullptr);

    std::string bboxes = op_desc.Input("BBoxes").front();
    std::string scores = op_desc.Input("Scores").front();
    std::string output_name = op_desc.Output("Out").front();

    auto* bboxes_tensor = engine_->GetITensor(bboxes);
    auto* scores_tensor = engine_->GetITensor(scores);

    int background_label =
        BOOST_GET_CONST(int, op_desc.GetAttr("background_label"));
    float score_threshold =
        BOOST_GET_CONST(float, op_desc.GetAttr("score_threshold"));
    int nms_top_k = BOOST_GET_CONST(int, op_desc.GetAttr("nms_top_k"));
    float nms_threshold =
        BOOST_GET_CONST(float, op_desc.GetAttr("nms_threshold"));
    int keep_top_k = BOOST_GET_CONST(int, op_desc.GetAttr("keep_top_k"));
    bool normalized = BOOST_GET_CONST(bool, op_desc.GetAttr("normalized"));
    int num_classes = scores_tensor->getDimensions().d[0];

    auto bboxes_dims = bboxes_tensor->getDimensions();
    nvinfer1::Dims3 bboxes_expand_dims(bboxes_dims.d[0], 1, bboxes_dims.d[1]);
    auto* bboxes_expand_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *bboxes_tensor);
    bboxes_expand_layer->setReshapeDimensions(bboxes_expand_dims);

    nvinfer1::Permutation permutation{1, 0};
    auto* scores_transpose_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *scores_tensor);
    scores_transpose_layer->setFirstTranspose(permutation);

    std::vector<nvinfer1::ITensor*> batch_nms_inputs;
    batch_nms_inputs.push_back(bboxes_expand_layer->getOutput(0));
    batch_nms_inputs.push_back(scores_transpose_layer->getOutput(0));

    constexpr bool shareLocation = true;
    constexpr bool clip_boxes = false;

    const std::vector<nvinfer1::PluginField> fields{
        {"shareLocation", &shareLocation, nvinfer1::PluginFieldType::kINT32, 1},
        {"backgroundLabelId", &background_label,
         nvinfer1::PluginFieldType::kINT32, 1},
        {"numClasses", &num_classes, nvinfer1::PluginFieldType::kINT32, 1},
        {"topK", &nms_top_k, nvinfer1::PluginFieldType::kINT32, 1},
        {"keepTopK", &keep_top_k, nvinfer1::PluginFieldType::kINT32, 1},
        {"scoreThreshold", &score_threshold,
         nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"iouThreshold", &nms_threshold, nvinfer1::PluginFieldType::kFLOAT32,
         1},
        {"isNormalized", &normalized, nvinfer1::PluginFieldType::kINT32, 1},
        {"clipBoxes", &clip_boxes, nvinfer1::PluginFieldType::kINT32, 1},
    };

    nvinfer1::PluginFieldCollection* plugin_collections =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_collections) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_collections->nbFields = static_cast<int>(fields.size());
    plugin_collections->fields = fields.data();

    auto creator = GetPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
    auto batch_nms_plugin =
        creator->createPlugin("BatchNMSPlugin", plugin_collections);
    free(plugin_collections);

    auto batch_nms_layer = engine_->network()->addPluginV2(
        batch_nms_inputs.data(), batch_nms_inputs.size(), *batch_nms_plugin);
    auto nmsed_boxes = batch_nms_layer->getOutput(1);
    auto nmsed_scores = batch_nms_layer->getOutput(2);
    auto nmsed_classes = batch_nms_layer->getOutput(3);

    auto nmsed_scores_transpose_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *nmsed_scores);
    nmsed_scores_transpose_layer->setReshapeDimensions(
        nvinfer1::Dims2(keep_top_k, 1));
    auto nmsed_classes_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *nmsed_classes);
    nmsed_classes_reshape_layer->setReshapeDimensions(
        nvinfer1::Dims2(keep_top_k, 1));

    std::vector<nvinfer1::ITensor*> concat_inputs;
    concat_inputs.push_back(nmsed_classes_reshape_layer->getOutput(0));
    concat_inputs.push_back(nmsed_scores_transpose_layer->getOutput(0));
    concat_inputs.push_back(nmsed_boxes);

    auto nms_concat_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Concatenation, concat_inputs.data(), concat_inputs.size());
    nms_concat_layer->setAxis(1);

    RreplenishLayerAndOutput(nms_concat_layer, "multiclass_nms", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multiclass_nms, MultiClassNMSOpConverter);
