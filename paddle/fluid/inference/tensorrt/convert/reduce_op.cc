/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <NvInfer.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

class ReduceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "Convert " << op_type << " op to tensorrt reduce layer";
    framework::OpDesc op_desc(op, nullptr);
    auto reduce_type = ops_.find(op_type);
    auto* x = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::Dims input_shape = x->getDimensions();
    int input_dims = input_shape.nbDims;

    bool keep_dim = PADDLE_GET_CONST(bool, op_desc.GetAttr("keep_dim"));

    std::vector<int32_t> dim;
    if (op_desc.GetProtoAttr("dim").type() ==
        framework::proto::AttrType::INTS) {
      dim = PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("dim"));
    } else if (op_desc.GetProtoAttr("dim").type() ==
               framework::proto::AttrType::LONGS) {
      std::vector<int64_t> tem_dim =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("dim"));
      for (size_t i = 0; i < tem_dim.size(); i++) {
        dim.push_back(static_cast<int32_t>(tem_dim[i]));
      }
    }

    bool reduce_all = PADDLE_GET_CONST(bool, op_desc.GetAttr("reduce_all"));

    if (dim.size() == 0) {
      reduce_all = true;
    }

    nvinfer1::IReduceLayer* layer = nullptr;
    if (reduce_all) {
      uint32_t reduce_dim = 0;
      for (int i = 0; i < input_dims; ++i) {
        reduce_dim |= 1 << i;
      }
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   Reduce,
                                   *x,
                                   reduce_type->second.front(),
                                   reduce_dim,
                                   keep_dim);
    } else {
      auto CvtToBitMask = [&](const std::vector<int32_t>& dims) -> uint32_t {
        uint32_t res = 0;
        for (auto x : dims) {
          if (x < 0) {
            res |= 1 << (x + input_dims);
          } else {
            res |= 1 << x;
          }
        }
        return res;
      };
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   Reduce,
                                   *x,
                                   reduce_type->second.front(),
                                   CvtToBitMask(dim),
                                   keep_dim);
    }

    auto output_name = op_desc.Output("Out")[0];
    // Ensure that the output type and input type are consistent.
    layer->getOutput(0)->setType(layer->getInput(0)->getType());
    ReplenishLayerAndOutput(layer, op_type, {output_name}, test_mode);
  }

 protected:
  std::string op_type;
  static const std::unordered_map<std::string,
                                  std::vector<nvinfer1::ReduceOperation>>
      ops_;
};

const std::unordered_map<std::string, std::vector<nvinfer1::ReduceOperation>>
    ReduceOpConverter::ops_ = {
        {"reduce_mean", {nvinfer1::ReduceOperation::kAVG}},
        {"reduce_sum", {nvinfer1::ReduceOperation::kSUM}},
        {"reduce_max", {nvinfer1::ReduceOperation::kMAX}},
        {"reduce_min", {nvinfer1::ReduceOperation::kMIN}},
        {"reduce_prod", {nvinfer1::ReduceOperation::kPROD}},
        {"reduce_any", {nvinfer1::ReduceOperation::kMAX}},
        {"reduce_all", {nvinfer1::ReduceOperation::kMIN}},
};

class ReduceSumOpConverter : public ReduceOpConverter {
 public:
  ReduceSumOpConverter() { op_type = "reduce_sum"; }
};

class ReduceMeanOpConverter : public ReduceOpConverter {
 public:
  ReduceMeanOpConverter() { op_type = "reduce_mean"; }
};

class ReduceMaxOpConverter : public ReduceOpConverter {
 public:
  ReduceMaxOpConverter() { op_type = "reduce_max"; }
};

class ReduceMinOpConverter : public ReduceOpConverter {
 public:
  ReduceMinOpConverter() { op_type = "reduce_min"; }
};

class ReduceProdOpConverter : public ReduceOpConverter {
 public:
  ReduceProdOpConverter() { op_type = "reduce_prod"; }
};

class ReduceAnyOpConverter : public ReduceOpConverter {
 public:
  ReduceAnyOpConverter() { op_type = "reduce_any"; }
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a paddle " << op_type << " op to tensorrt reduce layer";
    framework::OpDesc op_desc(op, nullptr);
    auto reduce_type = ops_.find(op_type);
    auto* x = engine_->GetITensor(op_desc.Input("X").front());
    // Cast the DataType to float
    nvinfer1::IReduceLayer* reduce_layer = nullptr;
    auto* cast_layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *x);
    cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
    cast_layer->getOutput(0)->setType(nvinfer1::DataType::kINT32);

    nvinfer1::Dims input_shape = x->getDimensions();
    int input_dims = input_shape.nbDims;
    // Discriminate DataType between int and bool.
    bool keep_dim = PADDLE_GET_CONST(bool, op_desc.GetAttr("keep_dim"));

    std::vector<int32_t> dim;
    if (op_desc.GetProtoAttr("dim").type() ==
        framework::proto::AttrType::INTS) {
      dim = PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("dim"));
    } else if (op_desc.GetProtoAttr("dim").type() ==
               framework::proto::AttrType::LONGS) {
      std::vector<int64_t> tem_dim =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("dim"));
      for (size_t i = 0; i < tem_dim.size(); i++) {
        dim.push_back(static_cast<int32_t>(tem_dim[i]));
      }
    }
    bool reduce_all = PADDLE_GET_CONST(bool, op_desc.GetAttr("reduce_all"));

    if (reduce_all) {
      uint32_t reduce_dim = 0;
      for (int i = 0; i < input_dims; ++i) {
        reduce_dim |= 1 << i;
      }
      reduce_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                          Reduce,
                                          *cast_layer->getOutput(0),
                                          reduce_type->second.front(),
                                          reduce_dim,
                                          keep_dim);
    } else {
      auto CvtToBitMask = [&](const std::vector<int32_t>& dims) -> uint32_t {
        uint32_t res = 0;
        for (auto x : dims) {
          if (x < 0) {
            res |= 1 << (x + input_dims);
          } else {
            res |= 1 << x;
          }
        }
        return res;
      };
      reduce_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                          Reduce,
                                          *cast_layer->getOutput(0),
                                          reduce_type->second.front(),
                                          CvtToBitMask(dim),
                                          keep_dim);
    }

    auto output_name = op_desc.Output("Out")[0];

    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Identity, *reduce_layer->getOutput(0));
    layer->setOutputType(0, nvinfer1::DataType::kBOOL);
    layer->getOutput(0)->setType(nvinfer1::DataType::kBOOL);
    // Ensure that the output type and input type are consistent.
    layer->getOutput(0)->setType(cast_layer->getInput(0)->getType());

    ReplenishLayerAndOutput(layer, op_type, {output_name}, test_mode);
  };
};

class ReduceAllOpConverter : public ReduceAnyOpConverter {
 public:
  ReduceAllOpConverter() { op_type = "reduce_all"; }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(reduce_sum, ReduceSumOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_mean, ReduceMeanOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_max, ReduceMaxOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_min, ReduceMinOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_prod, ReduceProdOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_any, ReduceAnyOpConverter);
REGISTER_TRT_OP_CONVERTER(reduce_all, ReduceAllOpConverter);
