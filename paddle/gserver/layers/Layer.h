/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include "ModelConfig.pb.h"
#include "paddle/function/Function.h"
#include "paddle/gserver/activations/ActivationFunction.h"
#include "paddle/math/CpuSparseMatrix.h"
#include "paddle/parameter/Argument.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/Weight.h"
#include "paddle/utils/ClassRegistrar.h"
#include "paddle/utils/Util.h"

/// Macro for registering a layer type.
/// Example: REGISTER_LAYER(crf_error, CRFDecodingErrorLayer);
#define REGISTER_LAYER(__type_name, __class_name) \
  static InitFunction __reg_type_##__type_name(   \
      []() { Layer::registrar_.registerClass<__class_name>(#__type_name); })

#define REGISTER_LAYER_CREATE_FUNC(__type_name, createFunction) \
  static InitFunction __reg_type_##__type_name(                 \
      []() { Layer::registrar_.registerClass(#__type_name, createFunction); })

namespace paddle {

class Layer;
typedef std::shared_ptr<Layer> LayerPtr;
typedef std::map<std::string, LayerPtr> LayerMap;
class NeuralNetwork;

/// layer state, used for RNN and LSTM layers
struct LayerState {
  std::vector<MatrixPtr> value;
};
typedef std::shared_ptr<LayerState> LayerStatePtr;

/// Paddle device ID, MKLDNN is -2, CPU is -1
enum PADDLE_DEVICE_ID {
  MKLDNN_DEVICE = -2,
  CPU_DEVICE = -1,
};

/**
 * @brief Base class for layer.
 * Define necessary variables and functions for every layer.
 */
class Layer {
 protected:
  /// Layer config
  LayerConfig config_;
  /// whether to use GPU
  bool useGpu_;
  /// Device Id. MKLDNN is -2, CPU is -1, and GPU is 0, 1, 2 ...
  int deviceId_;
  /// Input layers
  std::vector<LayerPtr> inputLayers_;
  /// Argument of input layers
  std::vector<std::string> inputArgument_;

  /// Parameter for each input layer.
  /// Parameters_[i] is nullptr if inputLayers_[i] does not need parameter.
  std::vector<ParameterPtr> parameters_;

  /// nullptr if bias is not needed.
  ParameterPtr biasParameter_;

  /// Output
  Argument output_;
  /// Several outputs stored on different devices, used in 'parallel_nn' case,
  /// and record them by deviceId_.
  /// Also used in 'use_mkldnn' case.
  std::vector<Argument> outputOtherDevice_;
  /// If there are several outputs, map them by each name.
  /// MKLDNNLayer use it only to merge output grad
  std::map<std::string, Argument*> outputMap_;
  /// Used to merge grad on different devices.
  MatrixPtr tmpGrad_;

  std::unique_ptr<ActivationFunction> activation_;

  /// Current passType, PASS_TRAIN or PASS_TEST
  PassType passType_;

  /// Random 0-1 matrix for dropOut
  MatrixPtr dropOutMask_;

  /// Whether the layer need to compute gradient
  bool needGradient_;
  /// Whether the layer need to compute re-sequence information
  bool needSequenceInfo_;

  /// Mark input grad in(true) or out(false) of backward function.
  std::vector<bool> markInBackward_;

  /// Layer forward function
  std::vector<std::shared_ptr<FunctionBase>> forward_;
  /// Layer backward function
  std::vector<std::shared_ptr<FunctionBase>> backward_;

 public:
  /**
   * Wait until all input value ready.
   * Called before Layer::forward() function.
   */
  virtual void waitInputValue();

  /**
   * Copy layer's output_ to other device.
   * If output layer is in other device, called after Layer::forward() function.
   */
  virtual void copyOutputToOtherDevice();

  /**
   * Wait until all output grad ready and merge them to output_.grad.
   * Called before Layer::backward() function.
   */
  virtual void waitAndMergeOutputGrad();

  /**
   * Notify previous layer the output grad ready.
   * Called after Layer::backward() function.
   */
  virtual void markAllInputGrad();

 protected:
  /**
   * Create layer function. Function is called in forward or backward.
   * \param function, Layer::forward_ or Layer::backward_
   * \param name, function name
   * \param config, initialization configuration for the function
   */
  void createFunction(std::vector<std::shared_ptr<FunctionBase>>& function,
                      const std::string& name,
                      const FuncConfig& config) {
    if (useGpu_) {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-GPU"));
    } else {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-CPU"));
    }
    auto& func = function.back();
    func->init(config);
  }

  /**
   * Notify specified layer the output grad ready.
   * Called in the backward function.
   * If do mark input grad in the backward function, you should to ensure
   * that all input grad will be marked in the backward function.
   */
  void markInputGrad(int inputIndex);

  /**
   * Get the argument of input layer.
   */
  const Argument& getInput(size_t inputIndex) const {
    return inputLayers_[inputIndex]->getOutput(deviceId_);
  }

  /**
   * Get the argument of input layer.
   */
  const Argument& getInput(const Layer& inputLayer) const {
    return inputLayer.getOutput(deviceId_);
  }

  /**
   * Get the argument of input layer with deviceId.
   */
  const Argument& getInput(size_t inputIndex, int deviceId) const {
    return inputLayers_[inputIndex]->getOutput(deviceId);
  }

  /**
   * Get the forward-input value.
   */
  const MatrixPtr& getInputValue(int inputIndex) {
    return inputLayers_[inputIndex]->getOutput(deviceId_).value;
  }

  /**
   * Get the forward-input value.
   */
  const MatrixPtr& getInputValue(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).value;
  }

  /**
   * Get the forward-input value with deviceId.
   */
  const MatrixPtr& getInputValue(int inputIndex, int deviceId) {
    return inputLayers_[inputIndex]->getOutput(deviceId).value;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(int inputIndex) {
    return inputLayers_[inputIndex]->getOutput(deviceId_).grad;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).grad;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(int inputIndex, int deviceId) {
    return inputLayers_[inputIndex]->getOutput(deviceId).grad;
  }

  /**
   * Get the forward-input label.
   */
  const IVectorPtr& getInputLabel(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).ids;
  }

  /**
   * Change the size of output (value, grad).
   * Reset to value zero if isValueClean = true,
   * Reset to grad zero if isGradClean = true.
   */
  void resetSpecifyOutput(Argument& output,
                          size_t height,
                          size_t width,
                          bool isValueClean,
                          bool isGradClean);

  /**
   * Add output argument to other devices.
   */
  void addOutputArgument(int deviceId);

 public:
  explicit Layer(const LayerConfig& config, bool useGpu = FLAGS_use_gpu);
  virtual ~Layer() {}

  /// Register a Layer
  static ClassRegistrar<Layer, LayerConfig> registrar_;

  /**
   * Get the flag whether layer need to compute gradient.
   */
  bool needGradient() const { return needGradient_; }

  /**
   * Set the flag whether layer need to compute gradient.
   */
  void setNeedGradient(bool need) { needGradient_ = need; }

  /**
   * Set the flag whether layer need to re-compute sequence information,
   * which includes sequenceStartPositions or subSequenceStartPositions.
   */
  void setNeedSequenceInfo(bool need) { needSequenceInfo_ = need; }

  /**
   * Get layer's name.
   */
  const std::string& getName() const { return config_.name(); }

  /**
   * Get layer's type.
   */
  const std::string& getType() const { return config_.type(); }

  /**
   * Get layer's size.
   */
  size_t getSize() const { return config_.size(); }

  /**
   * Get layer's deviceId.
   */
  int getDeviceId() const { return deviceId_; }

  /**
   * Add the inputLayer.
   */
  void addPrev(LayerPtr l) { inputLayers_.push_back(l); }

  /**
   * Get the size of inputLayer[i].
   */
  const LayerPtr& getPrev(size_t i) { return inputLayers_[i]; }

  /**
   * Get the forward-output value.
   */
  const MatrixPtr& getOutputValue() { return output_.value; }

  /**
   * Get the forward-output label.
   */
  const IVectorPtr& getOutputLabel() { return output_.ids; }

  /**
   * Get the backward-Loss value.
   */
  const MatrixPtr& getOutputGrad() { return output_.grad; }
  /**
   * If layer has multi-output, set output into outputMap_.
   */
  void setOutput(const std::string& name, Argument* output) {
    outputMap_[name] = output;
  }

  /**
   * Get the output map size, if layer has multi-output.
   */
  size_t getOutputMapSize() { return outputMap_.size(); }

  /**
   * Get the output based on layer's name.
   */
  Argument& getOutput(const std::string& str = "") {
    if (str == "") {
      return output_;
    } else {
      auto output = outputMap_.find(str);
      if (output != outputMap_.end()) {
        return *output->second;
      } else {
        LOG(FATAL) << "No specific output " << str;
        return *((Argument*)nullptr);
      }
    }
  }

  /**
   * Get the output based on deviceId.
   */
  const Argument& getOutput(int deviceId) const {
    if (deviceId == getDeviceId()) {
      return output_;
    } else {
      for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
        if (outputOtherDevice_[i].deviceId == deviceId) {
          return outputOtherDevice_[i];
        }
      }

      LOG(FATAL) << "No specific device output ";
      return *((Argument*)nullptr);
    }
  }

  /**
   * Get layer's parameters.
   */
  const std::vector<ParameterPtr>& getParameters() { return parameters_; }

  /**
   * Get layer's bias-parameters.
   */
  const ParameterPtr& getBiasParameter() { return biasParameter_; }

  /**
   * Create pointer of layer.
   */
  static LayerPtr create(const LayerConfig& config);

  /**
   * Resize the output matrix size.
   */
  void resizeOutput(size_t height, size_t width);

  /**
   * Resize the output matrix size,
   * and reset value to zero.
   */
  void reserveOutput(size_t height, size_t width);

  /**
   * Resize the output matrix size,
   * and reset value and grad to zero.
   */
  void resetOutput(size_t height, size_t width);

  /**
   * Clear the gradient of output.
   */
  void zeroGrad();

  /**
   * Intialization.
   * For example, adding input layers from layerMap and parameterMap.
   */
  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  /**
   * Intialization for sub network if there has sub network.
   * @param rootNetwork root network
   * @param config model config
   * @param parameterTypes parameter's type
   * @param useGpu whether to use gpu or not
   */
  virtual void initSubNetwork(NeuralNetwork* rootNetwork,
                              const ModelConfig& config,
                              const std::vector<ParameterType>& parameterTypes,
                              bool useGpu) {}

  /**
   * @brief Access SubNetwork Object.
   *        If subnetwork exists, then invoke callback with subnetwrk.
   * @param callback if sub-network is exist, the callback is invoked.
   */
  virtual void accessSubNetwork(
      const std::function<void(NeuralNetwork&)>& callback) {}

  /**
   * If use sparse row matrix as parameter,
   * prefetch feature ids in input label.
   */
  virtual void prefetch() {}

  /**
   * Forward propagation.
   * All inherited implementation should call Layer::foward() function.
   */
  virtual void forward(PassType passType) {
    passType_ = passType;
    if (!inputLayers_.empty() && needSequenceInfo_) {
      const Argument& input = getInput(0);
      output_.sequenceStartPositions = input.sequenceStartPositions;
      output_.subSequenceStartPositions = input.subSequenceStartPositions;
      output_.cpuSequenceDims = input.cpuSequenceDims;
    }
  }

  /**
   * Reset the internal state variables.
   * Allocate them if they have not been allocated.
   * This function need to called before Layer::forward() for generating
   * sequence.
   *
   * This is used for sequence generation. When generating sequence, the
   * calculation at current timestamp depends on the state from previous
   * timestamp. The model needs to keep the information about the previous
   * timestamp in the state variables. Layers such as RecurrentLayer,
   * LstmLayer and ContextLayer have state variables.
   */
  virtual void resetState() {}

  /**
   * Set layer state.
   */
  virtual void setState(LayerStatePtr state) {}

  /**
   * Get layer state.
   * @return A copy of internal state.
   */
  virtual LayerStatePtr getState() { return nullptr; }

  /**
   * Show output state.
   */
  void showOutputStats();

  /**
   * Backward propagation.
   * Should only be called after Layer::forward() function.
   */
  virtual void backward(const UpdateCallback& callback = nullptr) = 0;

  /**
   * One pass is finished.
   */
  virtual void onPassEnd() {}

 protected:
  /**
   * Forward of activation function.
   */
  void forwardActivation();
  /**
   * Backward of activation function.
   */
  void backwardActivation();
  /**
   * Forward of dropOut.
   */
  void forwardDropOut();
  /**
   * Initilize the needGradient_ flag.
   */
  void initNeedFlags();
};

}  // namespace paddle
