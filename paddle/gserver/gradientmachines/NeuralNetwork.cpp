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

#include "paddle/utils/Util.h"

#include "NeuralNetwork.h"
#include "hl_gpu.h"
#include "paddle/utils/CustomStackTrace.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/gserver/layers/MKLDNNLayer.h"
#endif

#ifndef PADDLE_MOBILE_INFERENCE
#include "MultiNetwork.h"
#include "RecurrentGradientMachine.h"
#include "paddle/gserver/layers/AgentLayer.h"
#endif

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
#ifndef PADDLE_MOBILE_INFERENCE
  if (config.type() == "recurrent_nn") {
    return newNeuralNetwork("root");
  } else if (config.type() == "multi_nn") {
    return new MultiNetwork("root");
  } else {
    return newNeuralNetwork();
  }
#else
  return new NeuralNetwork();
#endif
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

  for (const auto& layer : layers_) {
    const auto& name = layer->getName();
    bool isMiddleLayer = true;

    // if data layer
    for (const auto& dataLayer : dataLayers_) {
      if (name == dataLayer->getName()) {
        isMiddleLayer = false;
        break;
      }
    }

    // if output layer
    for (const auto& dataLayer : outputLayers_) {
      if (name == dataLayer->getName()) {
        isMiddleLayer = false;
        break;
      }
    }

    if (isMiddleLayer) {
      middleLayers_.push_back(layer);
    }
  }
}

void NeuralNetwork::connect(LayerPtr agentLayer,
                            LayerPtr realLayer,
                            int height) {
#ifndef PADDLE_MOBILE_INFERENCE
  AgentLayer* agent = dynamic_cast<AgentLayer*>(agentLayer.get());
  CHECK_NOTNULL(agent);
  agent->setRealLayer(realLayer, height);
#endif
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
        if (mat) mat->clearIndices();
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

  gLayerStackTrace.set_stage(true);

  {
    for (auto& layer : layers_) {
      REGISTER_TIMER_INFO("ForwardTimer", layer->getName().c_str());
      gLayerStackTrace.push(layer->getName());
      layer->forward(passType);
      gLayerStackTrace.pop(layer->getName());
    }
  }

  outArgs->clear();
  outArgs->reserve(outputLayers_.size());
  for (auto& layer : outputLayers_) {
    outArgs->push_back(layer->getOutput());
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
  gLayerStackTrace.set_stage(false);
  FOR_EACH_R(layer, layers_) {
    REGISTER_TIMER_INFO("BackwardTimer", (*layer)->getName().c_str());
    gLayerStackTrace.push((*layer)->getName());
    if ((*layer)->needGradient()) {
      (*layer)->backward(callback);
    }
    gLayerStackTrace.pop((*layer)->getName());
  }
}

void NeuralNetwork::finish() {
#ifdef PADDLE_WITH_MKLDNN
  FOR_EACH_R(layer, layers_) {
    MKLDNNLayerPtr dnnLayer = std::dynamic_pointer_cast<MKLDNNLayer>(*layer);
    if (dnnLayer) {
      dnnLayer->convertWeightsToPaddle();
    }
  }
#endif
}

Argument NeuralNetwork::getLayerOutput(const std::string& layerName) {
  return getLayer(layerName)->getOutput();
}

void NeuralNetwork::onPassEnd() {
  for (auto& layer : layers_) {
    layer->onPassEnd();
  }
}

void NeuralNetwork::releaseOutput() {
  for (auto& layer : middleLayers_) {
    Argument& arg = layer->getOutput();
    arg.value.reset();
  }
}

#ifndef PADDLE_MOBILE_INFERENCE

class CombinedEvaluator : public Evaluator {
 public:
  void addEvaluator(std::unique_ptr<Evaluator>&& evaluator) {
    evaluators_.emplace_back(std::move(evaluator));
  }
  void start() override {
    for (auto& evaluator : evaluators_) {
      evaluator->start();
    }
  }

  void finish() override {
    for (auto& evaluator : evaluators_) {
      evaluator->finish();
    }
  }

  void eval(const NeuralNetwork& nn) override {
    for (auto& evaluator : evaluators_) {
      evaluator->eval(nn);
    }
  }
  real evalImp(std::vector<Argument>& arguments) override {
    (void)arguments;
    return -1;
  }
  void printStats(std::ostream& os) const override {
    for (auto& evaluator : evaluators_) {
      evaluator->printStats(os);
      os << ' ';
    }
  }

  void distributeEval(ParameterClient2* client) override {
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
  void getNames(std::vector<std::string>* names) override {
    for (auto& eval : evaluators_) {
      eval->getNames(names);
    }
  }

  /**
   * @brief getValue could get all inside evaluators' value.
   */
  real getValue(const std::string& name, Error* err) const override {
    return this->getMethodHelper<real>(
        name, err, [&name, err](const std::unique_ptr<Evaluator>& eval) {
          return eval->getValue(name, err);
        });
  }

  /**
   * @brief getType could get all inside evaluators' type.
   */
  std::string getType(const std::string& name, Error* err) const override {
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

class SubnetEvaluator : public CombinedEvaluator {
 public:
  SubnetEvaluator(const std::string& layerName,
                  std::unique_ptr<Evaluator>&& evaluator)
      : layerName_(layerName) {
    addEvaluator(std::move(evaluator));
  }
  void eval(const NeuralNetwork& nn) override {
    const LayerPtr& layer = nn.getLayer(layerName_);
    CHECK(layer) << "Nonexisted layer: " << layerName_ << " in submodel "
                 << nn.getName();
    bool accessed = false;
    layer->accessSubNetwork([this, &accessed](NeuralNetwork& subnet) {
      subnet.eval(evaluators_[0].get());
      accessed = true;
    });
    CHECK(accessed) << "There is no subnetwork for layer " << layerName_
                    << " in submodel " << nn.getName();
  }

 protected:
  std::string layerName_;
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
    for (auto& layer : layers_) {
      layer->accessSubNetwork(
          [layer, combinedEvaluator](NeuralNetwork& subnet) {
            std::unique_ptr<Evaluator> subEvaluator(new SubnetEvaluator(
                layer->getName(),
                std::unique_ptr<Evaluator>(subnet.makeEvaluator())));
            combinedEvaluator->addEvaluator(std::move(subEvaluator));
          });
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

#endif

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
