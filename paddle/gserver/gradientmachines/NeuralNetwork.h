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
#include <map>
#include <memory>

#include "ModelConfig.pb.h"
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/gserver/layers/CostLayer.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/Layer.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {
/*
 * @brief  Init function for the parameters.
 * @param paramId: the id of the parameter to init.
 * @param para: the pointer to the parameter to init.
 * @param sharedParams: the pointer to an array of the parameter to be shared.
 *                      If it is null, no parameter sharing is used.
 *                      Only CPU paramters can be shared.
 * It handles CPU, CPU sparse, CPU sparse remote,
 * and GPU parameters differently. If the type
 * of a parameter is NORMAL. Basically nothing need to be done.
 * CPU value: NORMAL.
 * CPU param: NORMAL.
 *
 * CPU sparse value: NORMAL.
 * CPU sparse gradient: MAT_SPARSE_ROW_AUTO_GROW.
 *
 * CPU sparse remote value: MAT_SPARSE_ROW_PREFETCH(_FULL_SIZE).
 * CPU sparse remote gradient: MAT_SPARSE_ROW (!sharedParams)
 *                             MAT_SPARSE_ROW_AUTO_GROW (sharedParams)
 *
 * GPU value: NORMAL
 * GPU param: NORMAL
 */
void parameterInitNN(int paramId,
                     Parameter* para,
                     std::vector<ParameterPtr>* sharedParams);

class NeuralNetwork : public GradientMachine {
 public:
  virtual void init(const ModelConfig& config,
                    ParamInitCallback callback = nullptr,
                    const std::vector<ParameterType>& parameterTypes =
                        std::vector<ParameterType>{PARAMETER_VALUE,
                                                   PARAMETER_GRADIENT,
                                                   PARAMETER_MOMENTUM},
                    bool useGpu = FLAGS_use_gpu);

  /**
   * Connect two submodels and
   * down-submodel's output become up-submodel's input.
   * By default, connection is one by one,
   * If the agent height is smaller than real layer, *height* has to be filled.
   *
   * @param realLayer  The down-submodel's output layer.
   * @param agentLayer The up-submodel's input agent layer.
   */
  static void connect(LayerPtr agentLayer, LayerPtr realLayer, int height = 0);
  void connect(std::string agentLayerName,
               NeuralNetwork* srcNN,
               std::string realLayerName);

  virtual void prefetch(const std::vector<Argument>& inArgs);

  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  virtual Argument getLayerOutput(const std::string& layerName);

  const LayerPtr& getLayer(const std::string& layerName) const {
    auto it = layerMap_.find(layerName);
    CHECK(it != layerMap_.end()) << "Unknown layer " << layerName;
    return it->second;
  }

  virtual void onPassEnd();

#ifndef PADDLE_MOBILE_INFERENCE
  virtual Evaluator* makeEvaluator() const;

  virtual void eval(Evaluator* evaluator) const;
#endif

  virtual void resetState();
  virtual void setOutputGrad(const std::vector<Argument>& args);

  /// set machine state
  virtual void setState(const MachineState& machineState);

  /// get machine state
  virtual void getState(MachineState& machineState);

  static NeuralNetwork* create(const ModelConfig& config);

  ParameterMap* getParameterMap() { return &parameterMap_; }

  /**
   * @brief Access each layer as a for each loop.
   * @param callback invoke with each layer.
   */
  template <typename T>
  void forEachLayer(T callback) {
    for (auto& l : layers_) {
      if (callback(l)) {
        break;
      }
    }
  }

  static NeuralNetwork* newNeuralNetwork(const std::string& name = "",
                                         NeuralNetwork* rootNetwork = nullptr);

  const std::string& getName() const { return subModelName_; }

  /// some finish work, like convert the weight format of MKLDNNLayers
  void finish();

  /**
   * @brief   Release the middle layer's output memory.
   *
   * @note    This function is used for memory optimization in inference.
   */
  void releaseOutput();

 protected:
  /**
   * The constructor of NeuralNetwork.
   * The sub networks can get parameters_ and parameterMap_
   * from base NeuralNetwork.
   *
   * @param subModelName The name of sub-model.
   * @param rootNetwork  It used in MultiNetwork.
   */
  NeuralNetwork(std::string subModelName = "",
                NeuralNetwork* rootNetwork = nullptr)
      : subModelName_(subModelName), rootNetwork_(rootNetwork) {}

  std::string subModelName_;
  ModelConfig config_;
  std::vector<LayerPtr> layers_;
  ParameterMap parameterMap_;
  LayerMap layerMap_;

  std::vector<DataLayerPtr> dataLayers_;
  std::vector<LayerPtr> outputLayers_;
  std::vector<LayerPtr> middleLayers_;

  static std::map<std::string, bool> dllInitMap;

  NeuralNetwork* rootNetwork_;

  /// Whether parameter of this NN is initialized by its own
  /// (i.e., not by callback supplied with the caller)
  bool paramSelfInited_;
};

}  // namespace paddle
