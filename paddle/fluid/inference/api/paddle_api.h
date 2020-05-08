// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

/*! \file paddle_api.h
 */

/*! \mainpage Paddle Inference APIs
 * \section intro_sec Introduction
 * The Paddle inference library aims to offer an high performance inference SDK
 * for Paddle users.
 */

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <vector>

/*! \namespace paddle
 */
namespace paddle {

/// \brief Paddle data type.
enum PaddleDType {
  FLOAT32,
  INT64,
  INT32,
  UINT8,
  // TODO(Superjomn) support more data types if needed.
};

/// \brief Memory manager for PaddleTensor.
///
/// The PaddleBuf holds a buffer for data input or output. The memory can be
/// allocated by user or by PaddleBuf itself, but in any case, the PaddleBuf
/// should be reused for better performance.
///
/// For user allocated memory, the following API can be used:
/// - PaddleBuf(void* data, size_t length) to set an external memory by
/// specifying the memory address and length.
/// - Reset(void* data, size_t length) to reset the PaddleBuf with an external
/// memory.
/// ATTENTION, for user allocated memory, deallocation should be done by users
/// externally after the program finished. The PaddleBuf won't do any allocation
/// or deallocation.
///
/// To have the PaddleBuf allocate and manage the memory:
/// - PaddleBuf(size_t length) will allocate a memory of size `length`.
/// - Resize(size_t length) resize the memory to no less than `length`,
/// ATTENTION
///  if the allocated memory is larger than `length`, nothing will done.
///
/// Usage:
///
/// Let PaddleBuf manage the memory internally.
/// \code{cpp}
/// const int num_elements = 128;
/// PaddleBuf buf(num_elements/// sizeof(float));
/// \endcode
///
/// Or
/// \code{cpp}
/// PaddleBuf buf;
/// buf.Resize(num_elements/// sizeof(float));
/// \endcode
/// Works the exactly the same.
///
/// One can also make the `PaddleBuf` use the external memory.
/// \code{cpp}
/// PaddleBuf buf;
/// void* external_memory = new float[num_elements];
/// buf.Reset(external_memory, num_elements*sizeof(float));
/// ...
/// delete[] external_memory; // manage the memory lifetime outside.
/// \endcode
///
class PaddleBuf {
 public:
  ///
  /// \brief PaddleBuf allocate memory internally, and manage it.
  ///
  /// \param[in] length The length of data.
  ///
  explicit PaddleBuf(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  ///
  /// \brief Set external memory, the PaddleBuf won't manage it.
  ///
  /// \param[in] data The start address of the external memory.
  /// \param[in] length The length of data.
  ///
  PaddleBuf(void* data, size_t length)
      : data_(data), length_(length), memory_owned_{false} {}
  ///
  /// \brief Copy only available when memory is managed externally.
  ///
  /// \param[in] other another `PaddleBuf`
  ///
  explicit PaddleBuf(const PaddleBuf& other);
  ///
  /// \brief Resize the memory.
  ///
  /// \param[in] length The length of data.
  ///
  void Resize(size_t length);
  ///
  /// \brief Reset to external memory, with address and length set.
  ///
  /// \param[in] data The start address of the external memory.
  /// \param[in] length The length of data.
  ///
  void Reset(void* data, size_t length);
  ///
  /// \brief Tell whether the buffer is empty.
  ///
  bool empty() const { return length_ == 0; }
  ///
  /// \brief Get the data's memory address.
  ///
  void* data() const { return data_; }
  ///
  /// \brief Get the memory length.
  ///
  size_t length() const { return length_; }

  ~PaddleBuf() { Free(); }
  PaddleBuf& operator=(const PaddleBuf&);
  PaddleBuf& operator=(PaddleBuf&&);
  PaddleBuf() = default;
  PaddleBuf(PaddleBuf&& other);

 private:
  void Free();
  void* data_{nullptr};  ///< pointer to the data memory.
  size_t length_{0};     ///< number of memory bytes.
  bool memory_owned_{true};
};

///
/// \brief Basic input and output data structure for PaddlePredictor.
///
struct PaddleTensor {
  PaddleTensor() = default;
  std::string name;  ///<  variable name.
  std::vector<int> shape;
  PaddleBuf data;  ///<  blob of data.
  PaddleDType dtype;
  std::vector<std::vector<size_t>> lod;  ///<  Tensor+LoD equals LoDTensor
};

enum class PaddlePlace { kUNK = -1, kCPU, kGPU };

/// \brief Represents an n-dimensional array of values.
/// The ZeroCopyTensor is used to store the input or output of the network.
/// Zero copy means that the tensor supports direct copy of host or device data
/// to device,
/// eliminating additional CPU copy. ZeroCopyTensor is only used in the
/// AnalysisPredictor.
/// It is obtained through PaddlePredictor::GetinputTensor()
/// and PaddlePredictor::GetOutputTensor() interface.
class ZeroCopyTensor {
 public:
  /// \brief Reset the shape of the tensor.
  /// Generally it's only used for the input tensor.
  /// Reshape must be called before calling mutable_data() or copy_from_cpu()
  /// \param shape The shape to set.
  void Reshape(const std::vector<int>& shape);

  /// \brief Get the memory pointer in CPU or GPU with specific data type.
  /// Please Reshape the tensor first before call this.
  /// It's usually used to get input data pointer.
  /// \param place The place of the tensor.
  template <typename T>
  T* mutable_data(PaddlePlace place);

  /// \brief Get the memory pointer directly.
  /// It's usually used to get the output data pointer.
  /// \param[out] place To get the device type of the tensor.
  /// \param[out] size To get the data size of the tensor.
  /// \return The tensor data buffer pointer.
  template <typename T>
  T* data(PaddlePlace* place, int* size) const;

  /// \brief Copy the host memory to tensor data.
  /// It's usually used to set the input tensor data.
  /// \param data The pointer of the data, from which the tensor will copy.
  template <typename T>
  void copy_from_cpu(const T* data);

  /// \brief Copy the tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  template <typename T>
  void copy_to_cpu(T* data);

  /// \brief Return the shape of the Tensor.
  std::vector<int> shape() const;

  /// \brief Set lod info of the tensor.
  /// More about LOD can be seen here:
  ///  https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#lodtensor
  /// \param x the lod info.
  void SetLoD(const std::vector<std::vector<size_t>>& x);
  /// \brief Return the lod info of the tensor.
  std::vector<std::vector<size_t>> lod() const;
  /// \brief Return the name of the tensor.
  const std::string& name() const { return name_; }
  void SetPlace(PaddlePlace place, int device = -1) {
    place_ = place;
    device_ = device;
  }

  /// \brief Return the data type of the tensor.
  /// It's usually used to get the output tensor data type.
  /// \return The data type of the tensor.
  PaddleDType type() const;

 protected:
  explicit ZeroCopyTensor(void* scope) : scope_{scope} {}
  void SetName(const std::string& name) { name_ = name; }
  void* FindTensor() const;

 private:
  std::string name_;
  bool input_or_output_;
  friend class AnalysisPredictor;
  void* scope_{nullptr};
  // The corresponding tensor pointer inside Paddle workspace is cached for
  // performance.
  mutable void* tensor_{nullptr};
  PaddlePlace place_;
  PaddleDType dtype_;
  int device_;
};

/// \brief A Predictor for executing inference on a model.
/// Base class for AnalysisPredictor and NativePaddlePredictor.
class PaddlePredictor {
 public:
  struct Config;
  PaddlePredictor() = default;
  PaddlePredictor(const PaddlePredictor&) = delete;
  PaddlePredictor& operator=(const PaddlePredictor&) = delete;

  /// \brief This interface takes input and runs the network.
  /// There are redundant copies of data between hosts in this operation,
  /// so it is more recommended to use the zecopyrun interface
  /// \param[in] inputs An list of PaddleTensor as the input to the network.
  /// \param[out] output_data Pointer to the tensor list, which holds the output
  /// paddletensor
  /// \param[in] batch_size This setting has been discarded and can be ignored.
  /// \return Whether the run is successful
  virtual bool Run(const std::vector<PaddleTensor>& inputs,
                   std::vector<PaddleTensor>* output_data,
                   int batch_size = -1) = 0;

  /// \brief  Used to get the name of the network input.
  /// Be inherited by AnalysisPredictor, Only used in ZeroCopy scenarios.
  /// \return Input tensor names.
  virtual std::vector<std::string> GetInputNames() { return {}; }

  /// \brief Get the input shape of the model.
  /// \return A map contains all the input names and shape defined in the model.
  virtual std::map<std::string, std::vector<int64_t>> GetInputTensorShape() {
    return {};
  }

  /// \brief Used to get the name of the network output.
  /// Be inherited by AnalysisPredictor, Only used in ZeroCopy scenarios.
  /// \return Output tensor names.
  virtual std::vector<std::string> GetOutputNames() { return {}; }

  /// \brief Get the input ZeroCopyTensor by name.
  /// Be inherited by AnalysisPredictor, Only used in ZeroCopy scenarios.
  /// The name is obtained from the GetInputNames() interface.
  /// \param name The input tensor name.
  /// \return Return the corresponding input ZeroCopyTensor.
  virtual std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string& name) {
    return nullptr;
  }

  /// \brief Get the output ZeroCopyTensor by name.
  /// Be inherited by AnalysisPredictor, Only used in ZeroCopy scenarios.
  /// The name is obtained from the GetOutputNames() interface.
  /// \param name The output tensor name.
  /// \return Return the corresponding output ZeroCopyTensor.
  virtual std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string& name) {
    return nullptr;
  }
  /// \brief Run the network with zero-copied inputs and outputs.
  /// Be inherited by AnalysisPredictor and only used in ZeroCopy scenarios.
  /// This will save the IO copy for transfering inputs and outputs to predictor
  /// workspace
  /// and get some performance improvement.
  /// To use it, one should call the AnalysisConfig.SwitchUseFeedFetchOp(true)
  /// and then use the `GetInputTensor` and `GetOutputTensor`
  /// to directly write or read the input/output tensors.
  /// \return Whether the run is successful
  virtual bool ZeroCopyRun() { return false; }

  /// \brief Clone an existing predictor
  /// When using clone, the same network will be created,
  /// and the parameters between them are shared.
  /// \return unique_ptr which contains the pointer of predictor
  virtual std::unique_ptr<PaddlePredictor> Clone() = 0;

  /// \brief Destroy the Predictor.
  virtual ~PaddlePredictor() = default;

  virtual std::string GetSerializedProgram() const {
    assert(false);  // Force raise error.
    return "NotImplemented";
  }

  /// \brief Base class for NativeConfig and AnalysisConfig.
  struct Config {
    std::string model_dir; /*!< path to the model directory. */
  };
};

///
/// \brief configuration manager for `NativePredictor`.
///
/// `AnalysisConfig` manages configurations of `NativePredictor`.
/// During inference procedure, there are many parameters(model/params path,
/// place of inference, etc.)
///
struct NativeConfig : public PaddlePredictor::Config {
  /// GPU related fields.
  bool use_gpu{false};
  int device{0};
  float fraction_of_gpu_memory{
      -1.f};  ///< Change to a float in (0,1] if needed.

  std::string prog_file;
  std::string
      param_file;  ///< Specify the exact path of program and parameter files.

  bool specify_input_name{false};  ///< Specify the variable's name of each
                                   ///< input if input tensors don't follow the
                                   ///< `feeds` and `fetches` of the phase
                                   ///< `save_inference_model`.

  /// Set and get the number of cpu math library threads.
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads) {
    cpu_math_library_num_threads_ = cpu_math_library_num_threads;
  }
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

 protected:
  int cpu_math_library_num_threads_{1};  ///< number of cpu math library (such
                                         ///< as MKL, OpenBlas) threads for each
                                         ///< instance.
};

///
/// \brief A factory to help create different predictors.
///
/// Usage:
///
/// \code{.cpp}
/// NativeConfig config;
/// ... // change the configs.
/// auto native_predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// FOR EXTENSION DEVELOPER:
/// Different predictors are designated by config type. Similar configs can be
/// merged, but there shouldn't be a huge config containing different fields for
/// more than one kind of predictors.
////
template <typename ConfigT>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

/// NOTE The following APIs are too trivial, we will discard it in the following
/// versions.
///
enum class PaddleEngineKind {
  kNative = 0,         ///< Use the native Fluid facility.
  kAutoMixedTensorRT,  ///< Automatically mix Fluid with TensorRT.
  kAnalysis,           ///< More optimization.
};

template <typename ConfigT, PaddleEngineKind engine>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

int PaddleDtypeSize(PaddleDType dtype);

std::string get_version();

}  // namespace paddle
