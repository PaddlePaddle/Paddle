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

#pragma once

#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <glog/logging.h>
#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

namespace dy = paddle::platform::dynload;

static size_t AccumDims(nvinfer1::Dims dims) {
  size_t num = dims.nbDims == 0 ? 0 : 1;
  for (int i = 0; i < dims.nbDims; i++) {
    PADDLE_ENFORCE_GT(dims.d[i], 0);
    num *= dims.d[i];
  }
  return num;
}

// TensorRT data type to size
const int kDataTypeSize[] = {
    4,  // kFLOAT
    2,  // kHALF
    1,  // kINT8
    4   // kINT32
};

// The following two API are implemented in TensorRT's header file, cannot load
// from the dynamic library. So create our own implementation and directly
// trigger the method from the dynamic library.
static nvinfer1::IBuilder* createInferBuilder(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IBuilder*>(
      dy::createInferBuilder_INTERNAL(logger, NV_TENSORRT_VERSION));
}
static nvinfer1::IRuntime* createInferRuntime(nvinfer1::ILogger* logger) {
  return static_cast<nvinfer1::IRuntime*>(
      dy::createInferRuntime_INTERNAL(logger, NV_TENSORRT_VERSION));
}
static nvonnxparser::IOnnxConfig* createONNXConfig() {
  // return static_cast<nvonnxparser::IOnnxConfig*>(
  //     dy::createONNXConfig_INTERNAL());
  return nullptr;
}
static nvonnxparser::IONNXParser* createONNXParser(
    const nvonnxparser::IOnnxConfig& config) {
  // return static_cast<nvonnxparser::IONNXParser*>(
  //     dy::createONNXParser_INTERNAL(config));
  return nullptr;
}

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINFO:
        LOG(INFO) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }

  static nvinfer1::ILogger& Global() {
    static nvinfer1::ILogger* x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}
};

std::string AbsPath(const std::string& dir, const std::string& file) {
  return dir + "/" + file;
}

// Load ONNX model and translate to TensorRT format.
void OnnxToGIEModel(const std::string& directory, const std::string& model_file,
                    unsigned int max_batch_size,
                    nvinfer1::IHostMemory*& gie_model_stream,
                    nvinfer1::ILogger* logger) {
  // create the builder
  nvinfer1::IBuilder* builder = createInferBuilder(logger);

  nvonnxparser::IOnnxConfig* config = createONNXConfig();
  config->setModelFileName(AbsPath(directory, model_file).c_str());

  nvonnxparser::IONNXParser* parser = tensorrt::createONNXParser(*config);

  // Optional - uncomment below lines to view network layer information
  // config->setPrintLayerInfo(true);
  // parser->reportParsingInfo();

  if (!parser->parse(AbsPath(directory, model_file).c_str(),
                     nvinfer1::DataType::kFLOAT)) {
    std::string msg("failed to parse onnx file");
    logger->log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }

  if (!parser->convertToTRTNetwork()) {
    std::string msg("ERROR, failed to convert onnx network into TRT network");
    logger->log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  nvinfer1::INetworkDefinition* network = parser->getTRTNetwork();

  // Build the engine
  builder->setMaxBatchSize(max_batch_size);
  builder->setMaxWorkspaceSize(1 << 20);

  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
  parser->destroy();

  // serialize the engine, then close everything down
  gie_model_stream = engine->serialize();
  engine->destroy();
  builder->destroy();
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
