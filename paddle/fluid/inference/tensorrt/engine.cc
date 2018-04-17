#include "paddle/fluid/inference/tensorrt/engine.h"

#include "NvInfer.h"

namespace paddle {

size_t AccumDims(const nvinfer1::Dim& dims) {
  size_t num = dims.nbDims == 0 ? 0 : 1;
  for (int i = 0; i < dims.nbDims, i++) {
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

enum class DataTypeSize { kFLOAT = 32, kHALF = };

virtual void TensorrtEngine::Build(const PbType& paddle_model) override {
  PADDLE_ENFORCE(false, "not implemented");
}

virtual void TensorrtEngine::Execute(int batch_size) override {
  // TODO
  infer_context_->enqueue(batch_size, )
}

virtual ~TensorrtEngine() {
  SAFE_DESTROY(infer_engine_)
  SAFE_DESTROY(infer_builder_)
  SAFE_DESTROY(infer_network_)
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
  infer_builder_ = createInferBuilder(logger);
  infer_network_ = trt_builder_->createNetwork();
}

void TensorrtEngine::FreezeNetwork() {
  PADDLE_ENFORCE(infer_builder_,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE(infer_network_,
                 "Call InitNetwork first to initialize network.");
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);
  infer_engine_ = infer_builder_->buildCudaEngine(*infer_network_);
  PADDLE_ENFORCE(infer_engine_, "build cuda engine failed!");

  // all the following data are no longer needed.
  SAFE_DESTROY(infer_builder_)
  // SAFE_DESTROY(infer_network_)

  infer_context_ = infer_engine_->createExecutionContext();

  // allocate GPU buffers.
  buffers_.resize(buffer_sizes_.size());
  for (const auto& item : buffer_sizes_) {
    int slot_offset = infer_engine_->getBindingIndex(item.first.c_str());
    PADDLE_ENFORCE_EQ(0, cudaMalloc(&buffers_, item.second));
  }
}

nvinfer1::ITensor* TensorrtEngine::DeclInput(const std::string& name,
                                             data_type dtype,
                                             const dim_type& dim) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate input name %s",
                    name);

  paddle_enforce(infer_network_, "should initnetwork first");
  auto* input = infer_network_->addinput(name, dtype, dim);
  paddle_enforce(res, "infer network add input %s failed", name);

  buffer_sizes_[name] = kdatatypesize[static_cast<int>(dtype)] * accumdims(dim);
  return input;
}

void tensorrtengine::DeclOutput(ilayer* layer, int offset,
                                const std ::string& name) {
  paddle_enforce_eq(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto* output = layer->getoutput(offset);
  output->setname(name);
  infer_network_->markoutput(*output);

  auto dims = output->getdimentions();
  buffer_sizes_[name] = kdatatypesize[static_cast<int>(output->gettype())] *
                        accumdims(output->getdimentions());
}

void* tensorrtengine::buffer(const std::string& name) {
  paddle_enforce_ne(infer_engine_, nullptr, "call freezenetwork first.");
  auto it = buffer_sizes_.find(name);
  paddle_enforce(it != buffer_sizes_.end());
  auto slot_offset = infer_engine_->getBindingIndex(name.c_str());
  return buffer_sizes_[slot_offset];
}

void TensorrtEngine::SetInputFromCPU(const std::string& name, void* data,
                                     size_t size) {
  void* buffer = buffer(name);
  PADDLE_ENFORCE(
      0, cudaMemcpyAsync(data, buffer, size, cudaMemcpyHostToDevice, *stream_));
}

void* TensorrtEngine::GetOutput(const std::string& name) {
  return buffer(name);
}

}  // namespace paddle

#define SAFE_DESTROY(ptr__) \
  if (ptr__ != nullptr) {   \
    ptr__->destroy();       \
    ptr__ = nullptr;        \
  }
