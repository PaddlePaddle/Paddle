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

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace paddle {

// Data type.
enum PaddleDType {
  FLOAT32,
  INT64,
  // TODO(Superjomn) support more data types if needed.
};

/*
 * Memory menage for PaddleTensor.
 * The PaddleBuf holds a buffer for data input or output. The memory can be
 * allocated by user or by PaddleBuf itself, but in any case, the PaddleBuf
 * should be reused for better performance.
 *
 * For user allocated memory, the following API can be used:
 * - PaddleBuf(void* data, size_t length) to set an external memory by
 * specifying
 *   the memory address and length.
 * - Reset(void* data, size_t length) to reset the PaddleBuf with an external
 * memory.
 * ATTENTION, for user allocated memory, deallocation should be done by users
 * externally after the program finished. The PaddleBuf won't do any allocation
 * or deallocation.
 *
 * To have the PaddleBuf allocate and manage the memory:
 * - PaddleBuf(size_t length) will allocate a memory of size `length`.
 * - Resize(size_t length) resize the memory to no less than `length`, ATTENTION
 *   if the allocated memory is larger than `length`, nothing will done.
 */
class PaddleBuf {
 public:
  // PaddleBuf allocate memory internally, and manage it.
  explicit PaddleBuf(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  // Set external memory, the PaddleBuf won't manage it.
  PaddleBuf(void* data, size_t length)
      : data_(data), length_(length), memory_owned_{false} {}
  // Copy only available when memory is managed externally.
  explicit PaddleBuf(const PaddleBuf&);

  // Resize the memory.
  void Resize(size_t length);
  // Reset to external memory, with address and length set.
  void Reset(void* data, size_t length);
  // Tell whether the buffer is empty.
  bool empty() const { return length_ == 0; }
  // Get the memory address.
  void* data() const { return data_; }
  // Get the memory length.
  size_t length() const { return length_; }

  ~PaddleBuf() { Free(); }
  PaddleBuf& operator=(const PaddleBuf&);
  PaddleBuf& operator=(PaddleBuf&&);
  PaddleBuf() = default;
  PaddleBuf(PaddleBuf&& other);

 private:
  void Free();
  void* data_{nullptr};  // pointer to the data memory.
  size_t length_{0};     // number of memory bytes.
  bool memory_owned_{true};
};

// Basic input and output data structure for PaddlePredictor.
struct PaddleTensor {
  PaddleTensor() = default;
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
  std::vector<std::vector<size_t>> lod;  // Tensor+LoD equals LoDTensor
};

enum class PaddlePlace { kUNK = -1, kCPU, kGPU };
// Tensor without copy, currently only supports AnalysisPredictor.
class ZeroCopyTensor {
 public:
  void Reshape(const std::vector<int>& shape);

  // Get the memory in CPU or GPU with specific data type, should Reshape first
  // to tell the data size.
  // Once can directly call this data to feed the data.
  // This is for write the input tensor.
  template <typename T>
  T* mutable_data(PaddlePlace place);
  // Get the memory directly, will return the place and memory size by pointer.
  // This is for reading the output tensor.
  template <typename T>
  T* data(PaddlePlace* place, int* size);

  std::vector<int64_t> shape();

  void SetLoD(const std::vector<std::vector<size_t>>& x);
  std::vector<std::vector<size_t>> lod() const;

 protected:
  explicit ZeroCopyTensor(void* scope) : scope_{scope} {}
  void SetName(const std::string& name) { name_ = name; }
  void* FindTensor() const;

 private:
  std::string name_;
  bool input_or_output_;
  friend class AnalysisPredictor;
  void* scope_{nullptr};
};

/*
 * A simple Inference API for Paddle.
 */
class PaddlePredictor {
 public:
  struct Config;
  PaddlePredictor() = default;
  PaddlePredictor(const PaddlePredictor&) = delete;
  PaddlePredictor& operator=(const PaddlePredictor&) = delete;

  // Predict an record.
  // The caller should be responsible for allocating and releasing the memory of
  // `inputs`. `inputs` should be available until Run returns. Caller should be
  // responsible for the output tensor's buffer, either allocated or passed from
  // outside.
  virtual bool Run(const std::vector<PaddleTensor>& inputs,
                   std::vector<PaddleTensor>* output_data,
                   int batch_size = -1) = 0;

  // Zero copy input and output optimization.
  // Get the input or output tensors, and operate on their memory directly,
  // without copy.
  virtual std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string& name) {
    return nullptr;
  }
  virtual std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string& name) {
    return nullptr;
  }
  virtual bool ZeroCopyRun() { return false; }

  // Clone a predictor that share the model weights, the Cloned predictor should
  // be thread-safe.
  virtual std::unique_ptr<PaddlePredictor> Clone() = 0;

  // Destroy the Predictor.
  virtual ~PaddlePredictor() = default;

  // The common configs for all the predictors.
  struct Config {
    std::string model_dir;  // path to the model directory.
  };
};

struct NativeConfig : public PaddlePredictor::Config {
  // GPU related fields.
  bool use_gpu{false};
  int device{0};
  float fraction_of_gpu_memory{-1.f};  // Change to a float in (0,1] if needed.

  // Specify the exact path of program and parameter files.
  std::string prog_file;
  std::string param_file;

  // Specify the variable's name of each input if input tensors don't follow the
  // `feeds` and `fetches` of the phase `save_inference_model`.
  bool specify_input_name{false};
};

// A factory to help create different predictors.
//
// Usage:
//
// NativeConfig config;
// ... // change the configs.
// auto native_predictor = CreatePaddlePredictor(config);
//
// FOR EXTENSION DEVELOPER:
// Different predictors are designated by config type. Similar configs can be
// merged, but there shouldn't be a huge config containing different fields for
// more than one kind of predictors.
template <typename ConfigT>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

// NOTE The following APIs are too trivial, we will discard it in the following
// versions.
enum class PaddleEngineKind {
  kNative = 0,         // Use the native Fluid facility.
  kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
  kAnalysis,           // More optimization.
  kAnakin              // Use Anakin for inference, not mature yet.
};

template <typename ConfigT, PaddleEngineKind engine>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

// ==
//
// -----------------------------------------------------------------------------------
// NOTE: The following APIs are not mature yet, we are still working on them.

namespace contrib {

// Accelerate GPU computation with TensorRT engine.
struct MixedRTConfig : public NativeConfig {
  // Determine whether a subgraph will be executed by TRT.
  int min_subgraph_size{1};
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int max_batch_size{1};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int workspace_size{1 << 30};
  //  We transform the Ops that can be converted into TRT layer in the model,
  //  and aggregate these Ops into subgraphs for TRT execution.
  //  We set this variable to control the minimum number of nodes in the
  //  subgraph, 3 as default value.
  int minimum_subgraph_size = 3;
  // Reserved configuration
  // We just support "FP32" now, "FP16" and "INT8" will be supported.
  std::string precision_mode = "FP32";
};

// NOTE WIP, not stable yet.
struct AnalysisConfig : public NativeConfig {
  enum class IrPassMode {
    kSystem,   // Use system default passes, not customize.
    kInclude,  // Specify the passes in `ir_passes`.
    kExclude   // Specify the disabled passes in `ir_passes`.
  };

  // Determine whether to perform graph optimization.
  bool enable_ir_optim = true;
  // Manually determine the IR passes to run.
  IrPassMode ir_mode{IrPassMode::kExclude};
  // passes to be excluded/included
  std::vector<std::string> ir_passes{"embedding_fc_lstm_fuse_pass"};

  // NOT stable yet.
  bool use_feed_fetch_ops{true};

  // NOTE this is just for internal development, please not use it.
  // NOT stable yet.
  bool _use_mkldnn{false};
};

// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  enum TargetType { NVGPU = 0, X86 };
  int device;
  std::string model_file;
  int max_batch_size{-1};
  TargetType target_type;
};

}  // namespace contrib

int PaddleDtypeSize(PaddleDType dtype);

}  // namespace paddle
