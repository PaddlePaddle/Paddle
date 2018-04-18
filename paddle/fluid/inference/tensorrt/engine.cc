#include "paddle/fluid/inference/tensorrt/engine.h"

#include <NvInfer.h>
#include <cuda.h>
#include <glog/logging.h>
#include "paddle/fluid/platform/dynload/tensorrt.h"
#include "paddle/fluid/platform/enforce.h"

namespace dy = paddle::platform::dynload;

namespace paddle {

size_t AccumDims(nvinfer1::Dims dims) {
  size_t num = dims.nbDims == 0 ? 0 : 1;
  for (int i = 0; i < dims.nbDims; i++) {
    PADDLE_ENFORCE_GT(dims.d[i], 0);
    LOG(INFO) << "dim.d: " << i << " " << dims.d[i];
    num *= dims.d[i];
  }
  return num;
}

const int kDataTypeSize[] = {
    4,  // kFLOAT
    2,  // kHALF
    1,  // kINT8
    4   // kINT32
};

void TensorrtEngine::Build(const PbType& paddle_model) {
  PADDLE_ENFORCE(false, "not implemented");
}

void TensorrtEngine::Execute(int batch_size) {
  infer_context_->enqueue(batch_size, buffers_.data(), *stream_, nullptr);
  cudaStreamSynchronize(*stream_);
}

TensorrtEngine::~TensorrtEngine() {
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
  Logger logger;
  infer_builder_.reset(createInferBuilder(logger));
  infer_network_.reset(infer_builder_->createNetwork());
}

void TensorrtEngine::FreezeNetwork() {
  PADDLE_ENFORCE(infer_builder_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE(infer_network_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);

  infer_engine_.reset(infer_builder_->buildCudaEngine(*infer_network_));
  PADDLE_ENFORCE(infer_engine_ != nullptr, "build cuda engine failed!");

  infer_context_.reset(infer_engine_->createExecutionContext());

  // allocate GPU buffers.
  buffers_.resize(buffer_sizes_.size(), nullptr);
  for (auto& item : buffer_sizes_) {
    if (item.second == 0) {
      auto slot_offset = infer_engine_->getBindingIndex(item.first.c_str());
      item.second = kDataTypeSize[static_cast<int>(
                        infer_engine_->getBindingDataType(slot_offset))] *
                    AccumDims(infer_engine_->getBindingDimensions(slot_offset));
    }
    PADDLE_ENFORCE_EQ(0, cudaMalloc(&buffer(item.first), item.second));
  }
}

nvinfer1::ITensor* TensorrtEngine::DeclInput(const std::string& name,
                                             data_type dtype,
                                             const dim_type& dim) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate input name %s",
                    name);

  PADDLE_ENFORCE(infer_network_ != nullptr, "should initnetwork first");
  auto* input = infer_network_->addInput(name.c_str(), dtype, dim);
  PADDLE_ENFORCE(input, "infer network add input %s failed", name);

  buffer_sizes_[name] = kDataTypeSize[static_cast<int>(dtype)] * AccumDims(dim);
  return input;
}

void TensorrtEngine::DeclOutput(nvinfer1::ILayer* layer, int offset,
                                const std ::string& name) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto* output = layer->getOutput(offset);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  infer_network_->markOutput(*output);
  buffer_sizes_[name] = 0;
  //  * data_size;
}

void* TensorrtEngine::GetOutputInGPU(const std::string& name) {
  return buffer(name);
}

void TensorrtEngine::GetOutputInCPU(const std::string& name, void* dst,
                                    size_t max_size) {
  // determine data size
  auto it = buffer_sizes_.find(name);
  PADDLE_ENFORCE(it != buffer_sizes_.end());
  PADDLE_ENFORCE_GT(it->second, 0);
  PADDLE_ENFORCE_GE(max_size, it->second);

  PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(dst, buffer(name), it->second,
                                       cudaMemcpyDeviceToHost, *stream_));
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
  void* buf = buffer(name);
  PADDLE_ENFORCE_EQ(
      0, cudaMemcpyAsync(buf, data, size, cudaMemcpyHostToDevice, *stream_));
}

}  // namespace paddle
