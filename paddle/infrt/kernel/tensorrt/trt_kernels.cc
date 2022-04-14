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
#include <unordered_set>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "glog/logging.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "paddle/infrt/kernel/tensorrt/trt_helper.h"
#include "paddle/infrt/kernel/tensorrt/trt_layers.h"

#include "paddle/infrt/backends/tensorrt/trt_engine.h"
#include "paddle/infrt/backends/tensorrt/trt_options.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"
#include "paddle/infrt/host_context/symbol_table.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace tensorrt {

::infrt::backends::tensorrt::TrtEngine CreateTrtEngine(
    MlirOperationWithInfrtSymbol create_engine_op) {
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
  options.workspace = 128;

  // Parse mlir Region which only has one block.
  mlir::Operation& operation = *create_engine_op.operation;
  auto* symbol_table = create_engine_op.symbol_table;
  CHECK_NOTNULL(symbol_table);

  unsigned int num_regions = operation.getNumRegions();
  CHECK_EQ(num_regions, 1U) << "only support one region case.";
  auto& region = operation.getRegion(0);
  auto& block = region.getBlocks().front();

  std::unordered_map<std::string, ::phi::DenseTensor*> trt_bind_inputs;
  ValueToITensorMap value_to_trt_tensor_map;
  ValueToTensorMap value_to_tensor_map;

  for (auto index_operand : llvm::enumerate(operation.getOperands())) {
    mlir::Value operand = index_operand.value();
    size_t idx = index_operand.index();

    const std::string input_name = "input_" + std::to_string(idx);
    auto* v = symbol_table->GetValue(std::to_string(idx));
    CHECK_NOTNULL(v);
    auto* t = &v->get<::phi::DenseTensor>();
    value_to_tensor_map[operand] = t;

    // TODO(wilber): get input info from mlir.

    // TODO(wilber): input dims, now only support static_shape, and just remove
    // the first dimension. If the first dim is not -1, maybe we can pass the
    // origin dims.

    // TODO(wilber): now only suppot float input.

    if (operand.isa<mlir::BlockArgument>()) {
      // TODO(wilber): A trick: the weights are CPU tensor and inputs are GPU
      // tensor, so we treat all GPU tensors as inputs to trt.
      if (t->place().GetType() == ::phi::AllocationType::GPU) {
        trt_bind_inputs[input_name] = t;
        nvinfer1::Dims dims;
        dims.nbDims = t->dims().size() - 1;
        for (int i = 0; i < dims.nbDims; ++i) {
          dims.d[i] = t->dims()[i + 1];
        }
        auto* in = network->addInput(
            input_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
        value_to_trt_tensor_map[operand] = in;
      }
    } else {
      // TODO(wilber): Replace with the op name that generates the weights.
      std::unordered_set<std::string> weight_flags{
          "phi_dt.tensor_map_get_tensor",
          "phi_dt.create_dense_tensor.cpu",
          "phi_dt.create_inited_dense_tensor.cpu.f32",
          "phi_dt.create_host_inited_dense_tensor.f32"};
      if (!weight_flags.count(
              operand.getDefiningOp()->getName().getStringRef().str())) {
        trt_bind_inputs[input_name] = t;
        nvinfer1::Dims dims;
        dims.nbDims = t->dims().size() - 1;
        for (int i = 0; i < dims.nbDims; ++i) {
          dims.d[i] = t->dims()[i + 1];
        }
        auto* in = network->addInput(
            input_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
        value_to_trt_tensor_map[operand] = in;
      }
    }
  }

  // TODO(wilber): Find a way to add layer.
  for (auto& operation : block.without_terminator()) {
    VLOG(1) << "process " << operation.getName().getStringRef().str() << " ...";
    if (trt::ActivationOp op = llvm::dyn_cast<trt::ActivationOp>(operation)) {
      ActivationFunc(
          op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::FullyConnectedOp op =
                   llvm::dyn_cast<trt::FullyConnectedOp>(operation)) {
      FcFunc(op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::ConvolutionOp op =
                   llvm::dyn_cast<trt::ConvolutionOp>(operation)) {
      ConvFunc(op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::PoolingOp op = llvm::dyn_cast<trt::PoolingOp>(operation)) {
      PoolFunc(op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::ShuffleOp op = llvm::dyn_cast<trt::ShuffleOp>(operation)) {
      ShuffleFunc(
          op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::ScaleNdOp op = llvm::dyn_cast<trt::ScaleNdOp>(operation)) {
      ScaleNdFunc(
          op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else if (trt::ElementWiseOp op =
                   llvm::dyn_cast<trt::ElementWiseOp>(operation)) {
      EltwiseFunc(
          op, network.get(), value_to_trt_tensor_map, value_to_tensor_map);
    } else {
      CHECK(false) << "not supported operation.";
    }
  }

  for (auto index_operand :
       llvm::enumerate(block.getTerminator()->getOperands())) {
    mlir::Value arg = index_operand.value();
    CHECK(value_to_trt_tensor_map.count(arg));
    // TODO(wilber): A trick that we name trt output tensor's name as output_0,
    // output_1, ...
    value_to_trt_tensor_map[arg]->setName(
        ("output_" + std::to_string(index_operand.index())).c_str());
    network->markOutput(*value_to_trt_tensor_map[arg]);
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

std::vector<::phi::DenseTensor*> TrtEngineCompute(
    backends::tensorrt::TrtEngine* engine, const ::phi::GPUContext& context) {
  engine->Run(context);
  std::vector<::phi::DenseTensor*> res;
  for (size_t i = 0; i < engine->GetOutputNum(); ++i) {
    res.push_back(engine->GetOutput("output_" + std::to_string(i)));
  }
  return res;
}

}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
