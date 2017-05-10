/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Util.h"

#include "paddle/utils/CustomStackTrace.h"
#include "paddle/utils/Logging.h"

#include "MultiNetwork.h"
#include "NeuralNetwork.h"
#include "RecurrentGradientMachine.h"
#include "hl_gpu.h"
#include "paddle/gserver/layers/AgentLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {
void parameterInitNN(int paramId,
                     Parameter* para,
                     std::vector<ParameterPtr>* sharedParams) {
  // Create parameters values.
  if (!para->useGpu() && sharedParams) {
    para->enableSharedType(PARAMETER_VALUE,
                           (*sharedParams)[paramId]->getBuf(PARAMETER_VALUE),
                           (*sharedParams)[paramId]->getMat(PARAMETER_VALUE));
  } else {
    if (para->isSparseRemoteUpdate()) {
      para->enableType(PARAMETER_VALUE,
                       FLAGS_loadsave_parameters_in_pserver
                           ? Parameter::MAT_SPARSE_ROW_PREFETCH
                           : Parameter::MAT_SPARSE_ROW_PREFETCH_FULL_SIZE);
    } else {
      para->enableType(PARAMETER_VALUE);
    }
  }
  // Create parameter gradients.
  if (para->isSparseRemoteUpdate() && !sharedParams) {
    para->enableType(PARAMETER_GRADIENT, Parameter::MAT_SPARSE_ROW);
  } else if (para->isGradSparseUpdate()) {
    para->enableType(PARAMETER_GRADIENT, Parameter::MAT_SPARSE_ROW_AUTO_GROW);
  } else if (!para->isStatic()) {
    para->enableType(PARAMETER_GRADIENT);
  }
}

NeuralNetwork* NeuralNetwork::create(const ModelConfig& config) {
  if (config.type() == "recurrent_nn") {
    return newNeuralNetwork("root");
  } else if (config.type() == "multi_nn") {
    return new MultiNetwork("root");
  } else {
    return newNeuralNetwork();
  }
}

std::map<std::string, bool> NeuralNetwork::dllInitMap;

void NeuralNetwork::init(const ModelConfig& config,
                         ParamInitCallback callback,
                         const std::vector<ParameterType>& parameterTypes,
                         bool useGpu) {
  using std::placeholders::_1;
  using std::placeholders::_2;
  ParamInitCallback paramCallback = nullptr;
  if (callback != nullptr) {
    paramSelfInited_ = false;
    paramCallback = callback;
  } else {
    paramSelfInited_ = true;
    paramCallback = std::bind(parameterInitNN, _1, _2, nullptr);
  }
  config_ = config;

  if (rootNetwork_ != nullptr) {
    // direct use parameters_ and parameterMap_ from base network
    CHECK_EQ((size_t)config.parameters_size(),
             rootNetwork_->getParameters().size());
    parameters_ = rootNetwork_->getParameters();
    parameterMap_ = *(rootNetwork_->getParameterMap());
  } else {
    parameters_.reserve(config.parameters_size());
    for (const auto& para_config : config.parameters()) {
      auto parameter = std::make_shared<Parameter>(para_config,
                                                   useGpu,
                                                   /*initialize=*/false);
      paramCallback(parameters_.size(), parameter.get());
      if (!callback) {
        for (ParameterType type :
             (parameter->isStatic()
                  ? std::vector<ParameterType>{PARAMETER_VALUE}
                  : parameterTypes)) {
          if (type != PARAMETER_VALUE && type != PARAMETER_GRADIENT) {
            parameter->enableType(type);
          }
        }
      }
      parameter->setID(parameters_.size());
      parameters_.push_back(parameter);
      CHECK(!parameterMap_.count(parameter->getName()));
      parameterMap_[parameter->getName()] = parameter;
    }
  }

  auto layerCreate = [&](const LayerConfig& layer_config) {
    auto layer = Layer::create(layer_config);
    CHECK(layer) << "Create layer failed. Layer name:" << layer->getName();
    layers_.push_back(layer);
    CHECK(!layerMap_.count(layer->getName()));
    layerMap_[layer->getName()] = layer;
  };

  auto subModelConfig = std::find_if(config.sub_models().begin(),
                                     config.sub_models().end(),
                                     [=](const SubModelConfig& sub_model) {
                                       return sub_model.name() == subModelName_;
                                     });
  bool useSubModel = (subModelConfig != config.sub_models().end());
  CHECK_EQ(useSubModel, !subModelName_.empty());
  if (useSubModel) {
    layers_.reserve(subModelConfig->layer_names_size());
    for (const auto& layer_name : subModelConfig->layer_names()) {
      auto layer_config =
          std::find_if(config.layers().begin(),
                       config.layers().end(),
                       [=](const LayerConfig& layer_config) {
                         return layer_config.name() == layer_name;
                       });
      CHECK(layer_config != config.layers().end());
      layerCreate(*layer_config);
    }
  } else {
    layers_.reserve(config.layers_size());
    for (const auto& layer_config : config.layers()) {
      bool useLayer = true;
      if (config.has_external_config()) {
        useLayer = true;
        for (const auto& name : config.external_config().layer_names()) {
          if (layer_config.name() == name) {
            useLayer = false;
            break;
          }
        }
      }
      if (useLayer) {
        layerCreate(layer_config);
      }
    }
  }

  for (const auto& layer : layers_) {
    layer->init(layerMap_, parameterMap_);
    layer->initSubNetwork(this /*root*/, config_, parameterTypes, useGpu);
  }

  for (const auto& layer_name :
       (useSubModel ? subModelConfig->input_layer_names()
                    : config.input_layer_names())) {
    auto it = layerMap_.find(layer_name);
    CHECK(it != layerMap_.end());
    dataLayers_.push_back(std::dynamic_pointer_cast<DataLayer>(it->second));
  }

  for (const auto& layer_name :
       (useSubModel ? subModelConfig->output_layer_names()
                    : config.output_layer_names())) {
    auto it = layerMap_.find(layer_name);
    CHECK(it != layerMap_.end());
    outputLayers_.push_back(it->second);
  }
}

void NeuralNetwork::connect(LayerPtr agentLayer,
                            LayerPtr realLayer,
                            int height) {
  AgentLayer* agent = dynamic_cast<AgentLayer*>(agentLayer.get());
  CHECK_NOTNULL(agent);
  agent->setRealLayer(realLayer, height);
}

void NeuralNetwork::connect(std::string agentLayerName,
                            NeuralNetwork* srcNN,
                            std::string realLayerName) {
  connect(this->getLayer(agentLayerName), srcNN->getLayer(realLayerName));
}

void NeuralNetwork::prefetch(const std::vector<Argument>& inArgs) {
  CHECK_EQ(inArgs.size(), dataLayers_.size());

  if (paramSelfInited_) {
    for (auto& para : parameters_) {
      if (para->isSparseRemoteUpdate()) {
        auto mat = dynamic_cast<SparsePrefetchRowCpuMatrix*>(
            para->getMat(PARAMETER_VALUE).get());
        para->clearGradient();
        mat->clearIndices();
      }
    }
  }

  for (size_t i = 0; i != dataLayers_.size(); ++i) {
    if (FLAGS_parallel_nn) {
      const_cast<Argument&>(inArgs[i]).deviceId = -1;
    }
    dataLayers_[i]->setData(inArgs[i]);
  }

  for (auto& layer : layers_) {
    layer->prefetch();
  }

  if (paramSelfInited_) {
    for (auto& para : parameters_) {
      if (para->isSparseRemoteUpdate()) {
        auto mat = dynamic_cast<SparsePrefetchRowCpuMatrix*>(
            para->getMat(PARAMETER_VALUE).get());
        mat->setupIndices();
        auto matGrad = dynamic_cast<SparseRowCpuMatrix*>(
            para->getMat(PARAMETER_GRADIENT).get());
        matGrad->reserveStore();
      }
    }
  }
}

void NeuralNetwork::forward(const std::vector<Argument>& inArgs,
                            std::vector<Argument>* outArgs,
                            PassType passType) {
  CHECK_EQ(inArgs.size(), dataLayers_.size());
  outArgs->resize(outputLayers_.size());
  for (size_t i = 0; i != dataLayers_.size(); ++i) {
    dataLayers_[i]->setData(inArgs[i]);
  }

  {
    for (auto& layer : layers_) {
      REGISTER_TIMER_INFO("ForwardTimer", layer->getName().c_str());
      gLayerStackTrace.push(layer->getName());
      layer->forward(passType);
    }
  }

  outArgs->clear();
  outArgs->reserve(outputLayers_.size());
  for (auto& layer : outputLayers_) {
    outArgs->push_back(layer->getOutput());
  }
  if (passType == PASS_TEST) {
    gLayerStackTrace.clear();
  }
}

void NeuralNetwork::resetState() {
  for (auto& layer : layers_) {
    layer->resetState();
  }
}

void NeuralNetwork::setState(const MachineState& machineState) {
  for (size_t i = 0; i < layers_.size(); i++) {
    if (machineState[i] != nullptr) {
      layers_[i]->setState(machineState[i]);
    }
  }
}

void NeuralNetwork::getState(MachineState& machineState) {
  machineState.clear();
  machineState.reserve(layers_.size());
  for (auto& layer : layers_) {
    LayerStatePtr p = layer->getState();
    machineState.push_back(p);
  }
}

void NeuralNetwork::backward(const UpdateCallback& callback) {
  gLayerStackTrace.pop("");  // tell layer trace is during backward.
  FOR_EACH_R(layer, layers_) {
    REGISTER_TIMER_INFO("BackwardTimer", (*layer)->getName().c_str());
    if ((*layer)->needGradient()) {
      (*layer)->backward(callback);
    }
    gLayerStackTrace.pop((*layer)->getName());
  }
}

Argument NeuralNetwork::getLayerOutput(const std::string& layerName) {
  return getLayer(layerName)->getOutput();
}

void NeuralNetwork::onPassEnd() {
  for (auto& layer : layers_) {
    layer->onPassEnd();
  }
}

class CombinedEvaluator : public Evaluator {
public:
  void addEvaluator(std::unique_ptr<Evaluator>&& evaluator) {
    evaluators_.emplace_back(std::move(evaluator));
  }
  virtual void start() {
    for (auto& evaluator : evaluators_) {
      evaluator->start();
    }
  }

  virtual void finish() {
    for (auto& evaluator : evaluators_) {
      evaluator->finish();
    }
  }

  virtual void eval(const NeuralNetwork& nn) {
    for (auto& evaluator : evaluators_) {
      evaluator->eval(nn);
    }
  }
  virtual real evalImp(std::vector<Argument>& arguments) {
    (void)arguments;
    return -1;
  }
  virtual void printStats(std::ostream& os) const {
    for (auto& evaluator : evaluators_) {
      evaluator->printStats(os);
      os << ' ';
    }
  }

  virtual void distributeEval(ParameterClient2* client) {
    for (auto& evaluator : evaluators_) {
      evaluator->distributeEval(client);
    }
  }

protected:
  std::vector<std::unique_ptr<Evaluator>> evaluators_;

  // Evaluator interface
public:
  /**
   * @brief getNames will return all inside evaluators' names.
   * @param names [out]: return names.
   */
  void getNames(std::vector<std::string>* names) {
    for (auto& eval : evaluators_) {
      eval->getNames(names);
    }
  }

  /**
   * @brief getValue could get all inside evaluators' value.
   */
  real getValue(const std::string& name, Error* err) const {
    return this->getMethodHelper<real>(
        name, err, [&name, err](const std::unique_ptr<Evaluator>& eval) {
          return eval->getValue(name, err);
        });
  }

  /**
   * @brief getType could get all inside evaluators' type.
   */
  std::string getType(const std::string& name, Error* err) const {
    return this->getMethodHelper<std::string>(
        name, err, [&name, err](const std::unique_ptr<Evaluator>& eval) {
          return eval->getType(name, err);
        });
  }

private:
  template <typename T>
  T getMethodHelper(const std::string& name,
                    Error* err,
                    const std::function<T(const std::unique_ptr<Evaluator>&)>&
                        callback) const {
    for (auto& eval : evaluators_) {
      std::vector<std::string> names;
      eval->getNames(&names);
      if (std::find(names.begin(), names.end(), name) != names.end()) {
        return callback(eval);
      }
    }
    *err = Error("No such key %s", name.c_str());
    return T();
  }
};

Evaluator* NeuralNetwork::makeEvaluator() const {
  CombinedEvaluator* combinedEvaluator = new CombinedEvaluator();
  auto subModelConfig = std::find_if(config_.sub_models().begin(),
                                     config_.sub_models().end(),
                                     [=](const SubModelConfig& sub_model) {
                                       return sub_model.name() == subModelName_;
                                     });
  bool useSubModel = (subModelConfig != config_.sub_models().end());
  CHECK_EQ(useSubModel, !subModelName_.empty());
  if (useSubModel) {
    // create the evaluators that belong to CURRENT submodel
    for (int i = 0; i < subModelConfig->evaluator_names_size(); ++i) {
      // find evaluator by name
      auto thisEvalConfig = std::find_if(
          config_.evaluators().begin(),
          config_.evaluators().end(),
          [=](const EvaluatorConfig& ecfg) {
            return ecfg.name() == subModelConfig->evaluator_names(i);
          });
      bool validConfig = (thisEvalConfig != config_.evaluators().end());
      if (validConfig) {
        std::unique_ptr<Evaluator> evaluator(
            Evaluator::create(*thisEvalConfig));
        combinedEvaluator->addEvaluator(std::move(evaluator));
      }
    }
  } else {
    for (const EvaluatorConfig& evalConfig : config_.evaluators()) {
      std::unique_ptr<Evaluator> evaluator(Evaluator::create(evalConfig));
      combinedEvaluator->addEvaluator(std::move(evaluator));
    }
  }
  return combinedEvaluator;
}

void NeuralNetwork::eval(Evaluator* evaluator) const { evaluator->eval(*this); }

void NeuralNetwork::setOutputGrad(const std::vector<Argument>& args) {
  CHECK_GE(outputLayers_.size(), args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    outputLayers_[i]->getOutput().grad = args[i].grad;
  }
}

extern NeuralNetwork* newCustomNerualNetwork(const std::string& name,
                                             NeuralNetwork* network)
    __attribute__((weak));

NeuralNetwork* NeuralNetwork::newNeuralNetwork(const std::string& name,
                                               NeuralNetwork* rootNetwork) {
  if (newCustomNerualNetwork) {
    return newCustomNerualNetwork(name, rootNetwork);
  } else {
    return new NeuralNetwork(name, rootNetwork);
  }
}

}  // namespace paddle
