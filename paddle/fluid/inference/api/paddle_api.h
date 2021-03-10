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
#include "crypto/cipher.h"
#include "paddle_infer_declare.h"  // NOLINT
#include "paddle_infer_tensor.h"   // NOLINT
                                   /*! \namespace paddle
                                    */
namespace paddle {

using PaddleDType = paddle_infer::DataType;
using PaddlePlace = paddle_infer::PlaceType;
using PaddleBuf = paddle_infer::HostBuffer;
using PaddleTensor = paddle_infer::HostTensor;

/// \brief Represents an n-dimensional array of values.
/// The ZeroCopyTensor is used to store the input or output of the network.
/// Zero copy means that the tensor supports direct copy of host or device data
/// to device,
/// eliminating additional CPU copy. ZeroCopyTensor is only used in the
/// AnalysisPredictor.
/// It is obtained through PaddlePredictor::GetinputTensor()
/// and PaddlePredictor::GetOutputTensor() interface.

class PD_INFER_DECL ZeroCopyTensor : public paddle_infer::TensorHandle {
 public:
  /// \brief Copy the host memory to tensor data.
  /// It's usually used to set the input tensor data.
  /// \param data The pointer of the data, from which the tensor will copy.
  template <typename T>
  void copy_from_cpu(const T* data) {
    return CopyFromCpu(data);
  }
  /// \brief Copy the tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  template <typename T>
  void copy_to_cpu(T* data) {
    return CopyToCpu(data);
  }

 private:
  friend class AnalysisPredictor;
  explicit ZeroCopyTensor(void* scope) : paddle_infer::TensorHandle{scope} {}
};

/// \brief A Predictor for executing inference on a model.
/// Base class for AnalysisPredictor and NativePaddlePredictor.
class PD_INFER_DECL PaddlePredictor {
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
  /// To use it, one should call the AnalysisConfig.SwitchUseFeedFetchOp(false)
  /// and then use the `GetInputTensor` and `GetOutputTensor`
  /// to directly write or read the input/output tensors.
  /// \return Whether the run is successful
  virtual bool ZeroCopyRun() { return false; }

  ///
  /// \brief Clear the intermediate tensors of the predictor
  ///
  ///
  virtual void ClearIntermediateTensor() {}

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  virtual uint64_t TryShrinkMemory() { return 0; }

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
struct PD_INFER_DECL NativeConfig : public PaddlePredictor::Config {
  NativeConfig();
  /// GPU related fields.
  bool use_xpu{false};
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

struct AnalysisConfig;
struct NativeConfig;
struct DemoConfig;

template <>
PD_INFER_DECL std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig>(const AnalysisConfig& config);

template <>
PD_INFER_DECL std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<NativeConfig>(const NativeConfig& config);

template <>
PD_INFER_DECL std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<DemoConfig>(const DemoConfig& config);

/// NOTE The following APIs are too trivial, we will discard it in the following
/// versions.
///
enum class PaddleEngineKind {
  kNative = 0,         ///< Use the native Fluid facility.
  kAutoMixedTensorRT,  ///< Automatically mix Fluid with TensorRT.
  kAnalysis,           ///< More optimization.
};

template <typename ConfigT, PaddleEngineKind engine>
PD_INFER_DECL std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(
    const ConfigT& config);

template <>
PD_INFER_DECL std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    NativeConfig, PaddleEngineKind::kNative>(const NativeConfig& config);

template <>
PD_INFER_DECL std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    AnalysisConfig, PaddleEngineKind::kAnalysis>(const AnalysisConfig& config);

PD_INFER_DECL size_t PaddleDtypeSize(PaddleDType dtype);

PD_INFER_DECL std::string get_version();

PD_INFER_DECL std::string UpdateDllFlag(const char* name, const char* value);

PD_INFER_DECL std::shared_ptr<framework::Cipher> MakeCipher(
    const std::string& config_file);

}  // namespace paddle
