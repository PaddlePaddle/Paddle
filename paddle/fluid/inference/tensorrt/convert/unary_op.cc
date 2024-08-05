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

#include <NvInfer.h>

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::inference::tensorrt {

class UnaryOpConverter : public OpConverter {
 public:
  UnaryOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert a unary op to tensorrt unary layer whose "
               "type is "
            << op_type_;
    nvinfer1::ITensor* input_tensor =
        engine_->GetITensor(op_desc.Input("X")[0]);
    auto op_pair = ops.find(op_type_);
    nvinfer1::ILayer* layer = nullptr;

    nvinfer1::DataType org_type = input_tensor->getType();
    bool cast = org_type == nvinfer1::DataType::kINT8 ||
                org_type == nvinfer1::DataType::kINT32;
    if (cast) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input_tensor);
      if (engine_->precision() == phi::DataType::FLOAT32) {
        layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
      } else {
        layer->setOutputType(0, nvinfer1::DataType::kHALF);
      }
      input_tensor = layer->getOutput(0);
    }

    for (auto trt_op : op_pair->second) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Unary, *input_tensor, trt_op);
      input_tensor = layer->getOutput(0);
    }

    // type restore
    if (cast) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *input_tensor);
      layer->setOutputType(0, org_type);
      input_tensor = layer->getOutput(0);
    }

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, op_type_, {output_name}, test_mode);
  }

 protected:
  std::string op_type_;
  static const std::unordered_map<std::string,
                                  std::vector<nvinfer1::UnaryOperation>>
      ops;
};

const std::unordered_map<std::string, std::vector<nvinfer1::UnaryOperation>>
    UnaryOpConverter::ops = {
        {"exp", {nvinfer1::UnaryOperation::kEXP}},
        {"log", {nvinfer1::UnaryOperation::kLOG}},
        {"sqrt", {nvinfer1::UnaryOperation::kSQRT}},
        {"abs", {nvinfer1::UnaryOperation::kABS}},
        {"sin", {nvinfer1::UnaryOperation::kSIN}},
        {"cos", {nvinfer1::UnaryOperation::kCOS}},
        {"tan", {nvinfer1::UnaryOperation::kTAN}},
        {"sinh", {nvinfer1::UnaryOperation::kSINH}},
        {"cosh", {nvinfer1::UnaryOperation::kCOSH}},
        {"asin", {nvinfer1::UnaryOperation::kASIN}},
        {"acos", {nvinfer1::UnaryOperation::kACOS}},
        {"atan", {nvinfer1::UnaryOperation::kATAN}},
        {"asinh", {nvinfer1::UnaryOperation::kASINH}},
        {"acosh", {nvinfer1::UnaryOperation::kACOSH}},
        {"atanh", {nvinfer1::UnaryOperation::kATANH}},
        {"ceil", {nvinfer1::UnaryOperation::kCEIL}},
        {"floor", {nvinfer1::UnaryOperation::kFLOOR}},
        {"rsqrt",
         {nvinfer1::UnaryOperation::kSQRT, nvinfer1::UnaryOperation::kRECIP}},
        {"logical_not", {nvinfer1::UnaryOperation::kNOT}},
        {"reciprocal", {nvinfer1::UnaryOperation::kRECIP}},
#if IS_TRT_VERSION_GE(7000)
        {"erf", {nvinfer1::UnaryOperation::kERF}},
#endif
#if IS_TRT_VERSION_GE(8200)
        {"sign", {nvinfer1::UnaryOperation::kSIGN}},
        {"round", {nvinfer1::UnaryOperation::kROUND}},
#endif
};

class ExpOpConverter : public UnaryOpConverter {
 public:
  ExpOpConverter() { op_type_ = "exp"; }
};

class LogOpConverter : public UnaryOpConverter {
 public:
  LogOpConverter() { op_type_ = "log"; }
};

class SqrtOpConverter : public UnaryOpConverter {
 public:
  SqrtOpConverter() { op_type_ = "sqrt"; }
};
class AbsOpConverter : public UnaryOpConverter {
 public:
  AbsOpConverter() { op_type_ = "abs"; }
};
class SinOpConverter : public UnaryOpConverter {
 public:
  SinOpConverter() { op_type_ = "sin"; }
};
class CosOpConverter : public UnaryOpConverter {
 public:
  CosOpConverter() { op_type_ = "cos"; }
};
class TanOpConverter : public UnaryOpConverter {
 public:
  TanOpConverter() { op_type_ = "tan"; }
};
class SinhOpConverter : public UnaryOpConverter {
 public:
  SinhOpConverter() { op_type_ = "sinh"; }
};
class CoshOpConverter : public UnaryOpConverter {
 public:
  CoshOpConverter() { op_type_ = "cosh"; }
};
class AsinOpConverter : public UnaryOpConverter {
 public:
  AsinOpConverter() { op_type_ = "asin"; }
};
class AcosOpConverter : public UnaryOpConverter {
 public:
  AcosOpConverter() { op_type_ = "acos"; }
};
class AtanOpConverter : public UnaryOpConverter {
 public:
  AtanOpConverter() { op_type_ = "atan"; }
};
class AsinhOpConverter : public UnaryOpConverter {
 public:
  AsinhOpConverter() { op_type_ = "asinh"; }
};
class AcoshOpConverter : public UnaryOpConverter {
 public:
  AcoshOpConverter() { op_type_ = "acosh"; }
};
class AtanhOpConverter : public UnaryOpConverter {
 public:
  AtanhOpConverter() { op_type_ = "atanh"; }
};
class CeilOpConverter : public UnaryOpConverter {
 public:
  CeilOpConverter() { op_type_ = "ceil"; }
};
class FloorOpConverter : public UnaryOpConverter {
 public:
  FloorOpConverter() { op_type_ = "floor"; }
};

class RsqrtOpConverter : public UnaryOpConverter {
 public:
  RsqrtOpConverter() { op_type_ = "rsqrt"; }
};

class LogicalNotOpConverter : public UnaryOpConverter {
 public:
  LogicalNotOpConverter() { op_type_ = "logical_not"; }
};

class ReciprocalOpConverter : public UnaryOpConverter {
 public:
  ReciprocalOpConverter() { op_type_ = "reciprocal"; }
};

#if IS_TRT_VERSION_GE(8200)
class SignOpConverter : public UnaryOpConverter {
 public:
  SignOpConverter() { op_type_ = "sign"; }
};
#endif

#if IS_TRT_VERSION_GE(7000)
class ErfOpConverter : public UnaryOpConverter {
 public:
  ErfOpConverter() { op_type_ = "erf"; }
};
class RoundOpConverter : public UnaryOpConverter {
 public:
  RoundOpConverter() { op_type_ = "round"; }
};
#endif

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(exp, ExpOpConverter);
REGISTER_TRT_OP_CONVERTER(log, LogOpConverter);
REGISTER_TRT_OP_CONVERTER(sqrt, SqrtOpConverter);
REGISTER_TRT_OP_CONVERTER(abs, AbsOpConverter);
REGISTER_TRT_OP_CONVERTER(sin, SinOpConverter);
REGISTER_TRT_OP_CONVERTER(cos, CosOpConverter);
REGISTER_TRT_OP_CONVERTER(tan, TanOpConverter);
REGISTER_TRT_OP_CONVERTER(sinh, SinhOpConverter);
REGISTER_TRT_OP_CONVERTER(cosh, CoshOpConverter);
REGISTER_TRT_OP_CONVERTER(asin, AsinOpConverter);
REGISTER_TRT_OP_CONVERTER(acos, AcosOpConverter);
REGISTER_TRT_OP_CONVERTER(atan, AtanOpConverter);
REGISTER_TRT_OP_CONVERTER(asinh, AsinhOpConverter);
REGISTER_TRT_OP_CONVERTER(acosh, AcoshOpConverter);
REGISTER_TRT_OP_CONVERTER(atanh, AtanhOpConverter);
REGISTER_TRT_OP_CONVERTER(ceil, CeilOpConverter);
REGISTER_TRT_OP_CONVERTER(floor, FloorOpConverter);
REGISTER_TRT_OP_CONVERTER(rsqrt, RsqrtOpConverter);
REGISTER_TRT_OP_CONVERTER(logical_not, LogicalNotOpConverter);
REGISTER_TRT_OP_CONVERTER(reciprocal, ReciprocalOpConverter);
#if IS_TRT_VERSION_GE(7000)
REGISTER_TRT_OP_CONVERTER(erf, ErfOpConverter);
#endif
#if IS_TRT_VERSION_GE(8200)
REGISTER_TRT_OP_CONVERTER(sign, SignOpConverter);
REGISTER_TRT_OP_CONVERTER(round, RoundOpConverter);
#endif
