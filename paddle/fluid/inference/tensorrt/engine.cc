#include "paddle/fluid/inference/tensorrt/engine.h"

#include <NvInfer.h>
#include <cuda.h>
#include <glog/logging.h>
#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"

namespace dy = paddle::platform::dynload;

namespace paddle {

#define SAFE_DESTROY(ptr__) \
  if (ptr__ != nullptr) {   \
    ptr__->destroy();       \
    ptr__ = nullptr;        \
  }

size_t AccumDims(nvinfer1::Dims dims) {
  LOG(INFO) << "to get nbdims";
  LOG(INFO) << "ndims " << dims.nbDims;
  size_t num = dims.nbDims == 0 ? 0 : 1;
  for (int i = 0; i < dims.nbDims; i++) {
    PADDLE_ENFORCE_GT(dims.d[i], 0);
    num *= dims.d[i];
  }
  return num;
}

const int kDataTypeSize[] = {
    32,  // kFLOAT
    16,  // kHALF
    8,   // kINT8
    32   // kINT32
};

void TensorrtEngine::Build(const PbType& paddle_model) {
  PADDLE_ENFORCE(false, "not implemented");
}

void TensorrtEngine::Execute(int batch_size) {
  infer_context_->enqueue(batch_size, buffers_.data(), *stream_, nullptr);
}

TensorrtEngine::~TensorrtEngine() {
  SAFE_DESTROY(infer_engine_)
  SAFE_DESTROY(infer_builder_)
  SAFE_DESTROY(infer_network_)

  // clean buffer
  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      PADDLE_ENFORCE_EQ(0, cudaFree(buffer));
      buffer = nullptr;
    }
  }
}

namespace {

class Logger : public nvinfer1::ILogger {
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
};

// The following two API are implemented in TensorRT's header file, cannot load
// from the dynamic library. So create our own implementation and directly
// trigger the method from the dynamic library.
nvinfer1::IBuilder* createInferBuilder(nvinfer1::ILogger& logger) {
  return static_cast<nvinfer1::IBuilder*>(
      dy::createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}
nvinfer1::IRuntime* createInferRuntime(nvinfer1::ILogger& logger) {
  return static_cast<nvinfer1::IRuntime*>(
      dy::createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}
}  // namespace

void TensorrtEngine::InitNetwork() {
  LOG(INFO) << "init network";
  Logger logger;
  infer_builder_ = createInferBuilder(logger);
  infer_network_ = infer_builder_->createNetwork();
}

void TensorrtEngine::FreezeNetwork() {
  LOG(INFO) << "freeze network";
  PADDLE_ENFORCE(infer_builder_,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE(infer_network_,
                 "Call InitNetwork first to initialize network.");
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);
  LOG(INFO) << "building cuda engine";
  infer_engine_ = infer_builder_->buildCudaEngine(*infer_network_);
  PADDLE_ENFORCE(infer_engine_, "build cuda engine failed!");

  // all the following data are no longer needed.
  SAFE_DESTROY(infer_builder_)
  // SAFE_DESTROY(infer_network_)

  LOG(INFO) << "create execution context";
  infer_context_ = infer_engine_->createExecutionContext();

  // allocate GPU buffers.
  buffers_.resize(buffer_sizes_.size(), nullptr);
  for (auto& item : buffer_sizes_) {
    if (item.second == 0) {
      auto slot_offset = infer_engine_->getBindingIndex(item.first.c_str());
      LOG(INFO) << "to accudims";
      item.second = kDataTypeSize[static_cast<int>(
                        infer_engine_->getBindingDataType(slot_offset))] *
                    AccumDims(infer_engine_->getBindingDimensions(slot_offset));
    }
    LOG(INFO) << "malloc buffer";
    PADDLE_ENFORCE_EQ(0, cudaMalloc(&buffer(item.first), item.second));
  }
}

nvinfer1::ITensor* TensorrtEngine::DeclInput(const std::string& name,
                                             data_type dtype,
                                             const dim_type dim) {
  LOG(INFO) << "declare input " << name;
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate input name %s",
                    name);

  PADDLE_ENFORCE(infer_network_, "should initnetwork first");
  auto* input = infer_network_->addInput(name.c_str(), dtype, dim);
  PADDLE_ENFORCE(input, "infer network add input %s failed", name);

  buffer_sizes_[name] = kDataTypeSize[static_cast<int>(dtype)] * AccumDims(dim);
  return input;
}

void TensorrtEngine::DeclOutput(nvinfer1::ILayer* layer, int offset,
                                const std ::string& name) {
  LOG(INFO) << "declare output " << name;
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  LOG(INFO) << "get output " << offset;
  auto* output = layer->getOutput(offset);
  PADDLE_ENFORCE(output != nullptr);
  LOG(INFO) << "output set name " << name;
  output->setName(name.c_str());
  LOG(INFO) << "output mark output";
  infer_network_->markOutput(*output);
  LOG(INFO) << "to get dims";
  buffer_sizes_[name] = 0;
  //  * data_size;
}

void*& TensorrtEngine::buffer(const std::string& name) {
  PADDLE_ENFORCE(infer_engine_ != nullptr, "call freezenetwork first.");
  auto it = buffer_sizes_.find(name);
  PADDLE_ENFORCE(it != buffer_sizes_.end());
  auto slot_offset = infer_engine_->getBindingIndex(name.c_str());
  return buffers_[slot_offset];
}

void TensorrtEngine::SetInputFromCPU(const std::string& name, void* data,
                                     size_t size) {
  LOG(INFO) << "set input from cpu " << name;
  void* buf = buffer(name);
  PADDLE_ENFORCE_EQ(
      0, cudaMemcpyAsync(data, buf, size, cudaMemcpyHostToDevice, *stream_));
}

void* TensorrtEngine::GetOutput(const std::string& name) {
  return buffer(name);
}

}  // namespace paddle
