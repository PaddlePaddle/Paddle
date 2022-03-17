// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/infrt/kernel/tensorrt/trt_kernels.h"
#include <string>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "glog/logging.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "paddle/infrt/backends/tensorrt/trt_engine.h"
#include "paddle/infrt/backends/tensorrt/trt_options.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"
#include "paddle/infrt/host_context/symbol_table.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace tensorrt {

::infrt::backends::tensorrt::TrtEngine CreateTrtEngine(
    MlirOperationWithInfrtSymbol
        create_engine_op /*, input_tensors, output_tensors, weights*/) {
  // TODO(wilber): The device_id needs to get from mlir.
  int device_id = 0;
  backends::tensorrt::TrtEngine engine(device_id);

  auto* builder = engine.GetTrtBuilder();
  // TODO(wilber): How to process weights?
  backends::tensorrt::TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  // TODO(wilber): static_shape or dynamic_shape network? The code is just
  // static_shape test.
  network.reset(builder->createNetworkV2(0));

  // TODO(wilber): The build option shoule be fiiled from mlir info.
  backends::tensorrt::BuildOptions options;
  options.max_batch = 4;

  // Parse mlir Region which only has one block.
  mlir::Operation& operation = *create_engine_op.operation;
  auto* symbol_table = create_engine_op.symbol_table;
  CHECK_NOTNULL(symbol_table);

  unsigned int num_regions = operation.getNumRegions();
  CHECK_EQ(num_regions, 1U) << "only support one region case.";
  auto& region = operation.getRegion(0);
  auto& block = region.getBlocks().front();

  llvm::DenseMap<mlir::Value, nvinfer1::ITensor*> map_info;
  std::unordered_map<std::string, phi::DenseTensor*> trt_bind_inputs;

  for (auto index_operand : llvm::enumerate(operation.getOperands())) {
    mlir::Value operand = index_operand.value();
    size_t idx = index_operand.index();

    const std::string input_name = "input_" + std::to_string(idx);
    auto* v = symbol_table->GetValue(std::to_string(idx));
    CHECK_NOTNULL(v);
    auto* t = &v->get<phi::DenseTensor>();
    trt_bind_inputs[input_name] = t;
    // TODO(wilber): get input info from mlir.
    // TODO(wilber): input dims, now only support static_shape, and just remove
    // the first dimension.
    // TODO(wilber): now only suppot float input.
    nvinfer1::Dims dims;
    dims.nbDims = t->dims().size() - 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      dims.d[i] = t->dims()[i + 1];
    }
    auto* in =
        network->addInput(input_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
    map_info[operand] = in;
  }

  // TODO(wilber): Find a way to add layer.
  for (auto& inner_op : block.without_terminator()) {
    if (inner_op.getName().getStringRef() == "trt.Activation") {
      trt::ActivationOp act_op = llvm::dyn_cast<trt::ActivationOp>(inner_op);
      auto in_arg = act_op.getOperand();
      if (!map_info.count(in_arg)) {
        CHECK(false) << "map_info not has in_arg.";
      }
      nvinfer1::ActivationType act_type =
          static_cast<nvinfer1::ActivationType>(act_op.activation_type());
      auto* act_layer = network->addActivation(*map_info[in_arg], act_type);
      act_layer->setAlpha(act_op.alpha().convertToFloat());
      act_layer->setBeta(act_op.beta().convertToFloat());
      for (size_t i = 0; i < act_op->getNumResults(); ++i) {
        nvinfer1::ITensor* act_out_tensor = act_layer->getOutput(i);
        mlir::Value act_out = act_op->getResult(i);
        map_info[act_out] = act_out_tensor;
      }
    }

    // if (inner_op.getName().getStringRef() == "trt.Constant") {
    //   trt::ConstantOp op = llvm::dyn_cast<trt::ConstantOp>(inner_op);
    //   mlir::Value op_out = op.getResult();
    //   std::vector<float> weight_data{1};
    //   auto* layer = network->addConstant(nvinfer1::Dims2(1, 1),
    //   nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weight_data.data(), 1});
    //   auto* op_out_tenor = layer->getOutput(0);
    //   map_info[op_out] = op_out_tenor;
    // }
  }
  for (auto& inner_op : block.without_terminator()) {
    for (mlir::Value v : inner_op.getResults()) {
      for (mlir::Operation* user : v.getUsers()) {
        if (user->getName().getStringRef() == "infrt.return") {
          if (!map_info.count(v)) {
            CHECK(false) << "map_info not has value";
          }
          network->markOutput(*map_info[v]);
        }
      }
    }
  }
  // std::unordered_map<std::string, phi::DenseTensor*> trt_bind_outputs;
  mlir::Operation* ret = block.getTerminator();
  for (unsigned int i = 0; i < ret->getNumOperands(); ++i) {
    mlir::Value arg = ret->getOperand(i);
    CHECK(map_info.count(arg));
    map_info[arg]->setName(("output_" + std::to_string(i)).c_str());
  }
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    engine.PrepareOutputHandle(network->getOutput(i)->getName());
  }

  VLOG(3) << "trt engine build start.";
  engine.Build(std::move(network), options);
  VLOG(3) << "trt engine build done.";

  // TODO(wilber): get inference options from mlir.
  backends::tensorrt::InferenceOptions inference_options;
  inference_options.batch = 1;
  // TODO(wilber): bind trt input/output tensors.
  engine.SetUpInference(inference_options, trt_bind_inputs);
  return engine;
}

void PrintTrtLayer(backends::tensorrt::TrtEngine* engine) {
  engine->GetEngineInfo();
}

std::vector<phi::DenseTensor*> TrtEngineCompute(
    backends::tensorrt::TrtEngine* engine, const phi::GPUContext& context) {
  engine->Run(context);
  std::vector<phi::DenseTensor*> res;
  for (size_t i = 0; i < engine->GetOutputNum(); ++i) {
    res.push_back(engine->GetOutput("output_" + std::to_string(i)));
  }
  return res;
}

}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
