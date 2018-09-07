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

#include "RecurrentGradientMachine.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include "NeuralNetwork.h"
#include "paddle/gserver/layers/AgentLayer.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

DEFINE_string(diy_beam_search_prob_so, "", "the diy beam search cost so");

static const char* DIY_CALC_PROB_SYMBOL_NAME = "calc_prob";
static const char* DIY_START_CALC_PROB_SYMBOL_NAME = "start_calc_prob";
static const char* DIY_FINISH_CALC_PROB_SYMBOL_NAME = "finish_calc_prob";

namespace paddle {

/**
 * Start Custom Calculate Probability callback type.
 *
 * @param nNode, nodes: the path will be explored. nNodes is array size.
 *                      nodes is array elements.
 *
 * @return: A custom handler id that will passed to another callback.
 */
typedef int (*DiyStartCalcProbCallback)(size_t nNodes, int* nodes);

/**
 * Doing Custom Calculation of Probability callback type.
 *
 * @param handler: User custom handler. The return value from start calc prob.
 * @param nNode, nodes: Array. The current path.
 * @param curProb: The current log probability that neural network returns.
 *
 * @return: Log probability which user calculated, it will be updated to this
 *          path.
 * @NOTE: Return -INFINITY will DROP this path IMMEDIATELY!!
 */
typedef real (*DiyCalcProbCallback)(
    int handler, size_t nNodes, int* nodes, real curProb, bool atEos);

/**
 * Finish Custom Calculation of Probability callback type.
 *
 * @param handler: User custom handler. The return value from start calc prob.
 */
typedef void (*DiyStopCalcProbCallback)(int handler);

static DiyCalcProbCallback gDiyProbMethod = nullptr;
static DiyStartCalcProbCallback gDiyProbStart = nullptr;
static DiyStopCalcProbCallback gDiyProbStop = nullptr;
static void* gDiyProbHandle = nullptr;

static void exit_diy_prob() { dlclose(gDiyProbHandle); }

template <typename SymbolType>
static inline SymbolType loadDiySymbol(const char* symbolName) {
  void* sym = dlsym(gDiyProbHandle, symbolName);
  CHECK(sym) << "Cannot load symbol " << symbolName << " from "
             << FLAGS_diy_beam_search_prob_so;
  return reinterpret_cast<SymbolType>(sym);
}

static InitFunction __init__diy_prob_method(
    [] {
      std::string soName = FLAGS_diy_beam_search_prob_so;
      if (!soName.empty()) {
        gDiyProbHandle = dlopen(soName.c_str(), RTLD_LAZY);
        CHECK(gDiyProbHandle) << "Cannot Open DIY Prob So " << soName;
        atexit(exit_diy_prob);
        gDiyProbMethod =
            loadDiySymbol<decltype(gDiyProbMethod)>(DIY_CALC_PROB_SYMBOL_NAME);
        gDiyProbStart = loadDiySymbol<decltype(gDiyProbStart)>(
            DIY_START_CALC_PROB_SYMBOL_NAME);
        gDiyProbStop = loadDiySymbol<decltype(gDiyProbStop)>(
            DIY_FINISH_CALC_PROB_SYMBOL_NAME);
      }
    },
    std::numeric_limits<int>::max());

class BeamSearchControlCallbacks {
 public:
  RecurrentGradientMachine::BeamSearchCandidatesAdjustCallback
      beamSearchCandidateAdjust;
  RecurrentGradientMachine::NormOrDropNodeCallback normOrDropNode;
  RecurrentGradientMachine::DropCallback stopDetermineCandidates;

  //! for gcc46 aggregate initialization is not very well, so we need to
  //! explicit
  BeamSearchControlCallbacks(
      const RecurrentGradientMachine::BeamSearchCandidatesAdjustCallback&
          candidateAdjust,
      const RecurrentGradientMachine::NormOrDropNodeCallback& norm,
      const RecurrentGradientMachine::DropCallback& stop)
      : beamSearchCandidateAdjust(candidateAdjust),
        normOrDropNode(norm),
        stopDetermineCandidates(stop) {}
};

class BeamSearchStatisticsCallbacks {
 public:
  RecurrentGradientMachine::EachStepCallback onEachStepStarted;
  RecurrentGradientMachine::EachStepCallback onEachStepStoped;

  BeamSearchStatisticsCallbacks(
      const RecurrentGradientMachine::EachStepCallback& start,
      const RecurrentGradientMachine::EachStepCallback& stop)
      : onEachStepStarted(start), onEachStepStoped(stop) {}
};

RecurrentGradientMachine::RecurrentGradientMachine(
    const std::string& subModelName, NeuralNetwork* rootNetwork)
    : NeuralNetwork(subModelName),
      rootNetwork_(rootNetwork),
      beamSearchCtrlCallbacks_(nullptr),
      beamSearchStatistics_(nullptr) {
  CHECK(!subModelName_.empty());
}

/**
 * bias layer, as input of memory frame 0 will give vector of zeros
 * if bias parameter is not set.
 *
 * boot bias layer create directly in recurrent gradient machine, because:
 *
 * 1. It is only one frame, so it should not be placed in layer group,
 *    which is one instance for every one frame.
 *
 * 2. It is no input layer, so it need resetHeight() before forward(),
 *    and resetHeight() must be called in recurrent gradient machine,
 *    so it's should not be placed in root network.
 */
class BootBiasLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;
  IVectorPtr cpuIds_;

 public:
  explicit BootBiasLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    if (!Layer::init(layerMap, parameterMap)) return false;

    if (biasParameter_) {
      biases_ =
          std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
    }
    return true;
  }

  void resetHeight(int height) {
    if (config_.has_bos_id()) {  // used as a constant id layerConfig
      IVector::resizeOrCreate(output_.ids, height, useGpu_);
      output_.ids->reset((int)config_.bos_id());
    } else {
      resetOutput(height, getSize());
    }
  }

  void forward(PassType passType) override {
    if (biases_) {
      MatrixPtr outV = getOutputValue();
      outV->addBias(*(biases_->getW()), 1);
      forwardActivation();
    }
  }

  void backward(const UpdateCallback& callback) override {
    if (biases_ && biases_->getWGrad()) {
      backwardActivation();
      biases_->getWGrad()->collectBias(*getOutputGrad(), 1);
      biases_->getParameterPtr()->incUpdate(callback);
    }
  }
};

void RecurrentGradientMachine::init(
    const ModelConfig& config,
    ParamInitCallback callback,
    const std::vector<ParameterType>& parameterTypes,
    bool useGpu) {
  NeuralNetwork::init(config, callback, parameterTypes, useGpu);
  useGpu_ = useGpu;

  auto subModelConfig =
      std::find_if(config.sub_models().begin(),
                   config.sub_models().end(),
                   [this](const SubModelConfig& sub_model) {
                     return sub_model.name() == this->subModelName_;
                   });
  CHECK(subModelConfig != config.sub_models().end());
  reversed_ = subModelConfig->reversed();
  generating_ = subModelConfig->has_generator();

  inFrameLines_.resize(subModelConfig->in_links_size());
  for (size_t i = 0; i < inFrameLines_.size(); ++i) {
    inFrameLines_[i].linkName = subModelConfig->in_links(i).link_name();
    inFrameLines_[i].inLayer =
        rootNetwork_->getLayer(subModelConfig->in_links(i).layer_name());
  }

  outFrameLines_.resize(subModelConfig->out_links_size());
  for (size_t i = 0; i < outFrameLines_.size(); ++i) {
    auto& linkPair = subModelConfig->out_links(i);
    outFrameLines_[i].layerName = linkPair.layer_name();
    outFrameLines_[i].agentLayer = rootNetwork_->getLayer(linkPair.link_name());
  }

  memoryFrameLines_.resize(subModelConfig->memories_size());
  for (size_t i = 0; i < memoryFrameLines_.size(); ++i) {
    auto& memoryConfig = subModelConfig->memories(i);
    memoryFrameLines_[i].layerName = memoryConfig.layer_name();
    memoryFrameLines_[i].linkName = memoryConfig.link_name();
    auto agentConfig =
        std::find_if(config.layers().begin(),
                     config.layers().end(),
                     [&memoryConfig](const LayerConfig& layerConfig) {
                       return layerConfig.name() == memoryConfig.link_name();
                     });
    CHECK(agentConfig != config.layers().end());
    if (memoryConfig.has_boot_layer_name()) {
      memoryFrameLines_[i].rootLayer =
          rootNetwork_->getLayer(memoryConfig.boot_layer_name());

      LayerConfig scatterConfig = *agentConfig;
      memoryFrameLines_[i].rootAgent.reset(
          new ScatterAgentLayer(scatterConfig));
      memoryFrameLines_[i].rootAgent->init(LayerMap(), parameterMap_);

      memoryFrameLines_[i].bootLayer = memoryFrameLines_[i].rootAgent;
    } else {
      LayerConfig biasConfig = *agentConfig;
      if (memoryConfig.has_boot_bias_parameter_name()) {
        biasConfig.set_bias_parameter_name(
            memoryConfig.boot_bias_parameter_name());
        biasConfig.set_active_type(memoryConfig.boot_bias_active_type());
      } else if (memoryConfig.has_boot_with_const_id()) {
        biasConfig.set_bos_id(memoryConfig.boot_with_const_id());
      }
      memoryFrameLines_[i].biasLayer.reset(new BootBiasLayer(biasConfig));
      memoryFrameLines_[i].biasLayer->init(LayerMap(), parameterMap_);

      memoryFrameLines_[i].bootLayer = memoryFrameLines_[i].biasLayer;
    }

    if (subModelConfig->has_generator()) {
      memoryFrameLines_[i].scatterAgents.resize(2);
      for (auto& agent : memoryFrameLines_[i].scatterAgents) {
        agent.reset(new ScatterAgentLayer(*agentConfig));
        agent->init(LayerMap(), parameterMap_);
      }
    }
  }

  if (subModelConfig->has_generator()) {
    generator_.config = subModelConfig->generator();
    eosFrameLine_.reset(new EosFrameLine);
    maxSequenceLength_ = generator_.config.max_num_frames();
  }

  // get parameters actually used by this Layer Group
  resizeOrCreateFrames(1);
  for (auto& para : frames_[0]->getParameters()) {
    if (para->getSharedCount() > 0) {
      parameterIds_.push_back(para->getID());
    }
  }
  for (auto& para : parameters_) {  // bias layer parameters
    if (para->getSharedCount() > 0) {
      parameterIds_.push_back(para->getID());
    }
  }
}

void RecurrentGradientMachine::resizeOrCreateFrames(int numFrames) {
  if ((size_t)numFrames <= frames_.size()) {
    return;
  }

  frames_.reserve(numFrames);
  for (auto& inFrameLine : inFrameLines_) {
    inFrameLine.agents.reserve(numFrames);
  }
  for (auto& outFrameLine : outFrameLines_) {
    outFrameLine.frames.reserve(numFrames);
  }
  for (auto& memoryFrameLine : memoryFrameLines_) {
    memoryFrameLine.frames.reserve(numFrames);
    memoryFrameLine.agents.reserve(numFrames);
  }
  if (eosFrameLine_) {
    eosFrameLine_->layers.reserve(numFrames);
  }

  ParamInitCallback subParamInitCb = [this](int paramId, Parameter* para) {
    para->enableSharedType(PARAMETER_VALUE,
                           this->parameters_[paramId]->getBuf(PARAMETER_VALUE),
                           this->parameters_[paramId]->getMat(PARAMETER_VALUE));
    para->enableSharedType(
        PARAMETER_GRADIENT,
        this->parameters_[paramId]->getBuf(PARAMETER_GRADIENT),
        this->parameters_[paramId]->getMat(PARAMETER_GRADIENT));
  };

  for (int i = frames_.size(); i < numFrames; ++i) {
    std::unique_ptr<NeuralNetwork> frame(
        NeuralNetwork::newNeuralNetwork(subModelName_));
    frame->init(config_, subParamInitCb);

    for (auto& inFrameLine : inFrameLines_) {
      inFrameLine.agents.push_back(frame->getLayer(inFrameLine.linkName));
    }

    for (auto& outFrameLine : outFrameLines_) {
      outFrameLine.frames.push_back(frame->getLayer(outFrameLine.layerName));
    }
    for (auto& memoryFrameLine : memoryFrameLines_) {
      memoryFrameLine.frames.push_back(
          frame->getLayer(memoryFrameLine.layerName));
      memoryFrameLine.agents.push_back(
          frame->getLayer(memoryFrameLine.linkName));
    }
    if (eosFrameLine_) {
      eosFrameLine_->layers.push_back(
          frame->getLayer(generator_.config.eos_layer_name()));
    }

    frames_.emplace_back(std::move(frame));
  }
}

void RecurrentGradientMachine::resizeBootFrame(int numSequences) {
  for (auto& memoryFrameLine : memoryFrameLines_) {
    if (memoryFrameLine.biasLayer) {
      auto biasLayer =
          dynamic_cast<BootBiasLayer*>(memoryFrameLine.biasLayer.get());
      CHECK_NOTNULL(biasLayer);
      biasLayer->resetHeight(numSequences);
    } else {  // check input root layer height
      CHECK_EQ(numSequences,
               memoryFrameLine.rootLayer->getOutput().getNumSequences());
    }
  }
}

void RecurrentGradientMachine::prefetch(const std::vector<Argument>& inArgs) {
  LOG(FATAL) << "should not use this function";
}

void RecurrentGradientMachine::checkInputConsistency(
    int inlinkId, const std::vector<Argument::SeqInfo>& seqInfo) {
  if (commonSeqInfo_.empty()) {
    commonSeqInfo_.resize(seqInfo.size());
    for (size_t i = 0; i < seqInfo.size(); ++i) {
      commonSeqInfo_[i].topLevelLength = seqInfo[i].topLevelLength;
      commonSeqInfo_[i].seqId = seqInfo[i].seqId;
    }
  } else {
    CHECK_EQ(commonSeqInfo_.size(), seqInfo.size())
        << " RecurrentGroup " << subModelName_ << " input " << inlinkId
        << " has mismatched number of sequences";
    for (size_t i = 0; i < seqInfo.size(); ++i) {
      CHECK_EQ(commonSeqInfo_[i].topLevelLength, seqInfo[i].topLevelLength)
          << " RecurrentGroup " << subModelName_ << " input " << inlinkId
          << " has mismatched sequence length";
      CHECK_EQ(commonSeqInfo_[i].seqId, seqInfo[i].seqId)
          << " RecurrentGroup " << subModelName_ << " input " << inlinkId
          << " has mismatched sequence length";
    }
  }
}

void RecurrentGradientMachine::calcNumSequencesAtEachStep() {
  int numSequences = commonSeqInfo_.size();
  numSeqs_.resize(maxSequenceLength_);
  for (int i = 0; i < numSequences; ++i) {
    for (int j = 0; j < commonSeqInfo_[i].topLevelLength; ++j) {
      numSeqs_[j] = i + 1;
    }
  }
}

void RecurrentGradientMachine::reorganizeInput(PassType passType) {
  info_.clear();
  info_.resize(inFrameLines_.size());

  commonSeqInfo_.clear();
  seqInfos_.clear();
  seqInfos_.resize(inFrameLines_.size());

  for (size_t i = 0; i < inFrameLines_.size(); i++) {
    const Argument& input = inFrameLines_[i].inLayer->getOutput();
    if (!input.hasSeq()) {
      continue;
    }
    input.getSeqInfo(&seqInfos_[i]);
    checkInputConsistency(i, seqInfos_[i]);
  }
  CHECK(!commonSeqInfo_.empty())
      << "At least one input needs to be sequence or subsequence";
  maxSequenceLength_ = commonSeqInfo_[0].topLevelLength;

  calcNumSequencesAtEachStep();

  for (size_t i = 0; i < inFrameLines_.size(); ++i) {
    const Argument& input = inFrameLines_[i].inLayer->getOutput();
    if (!input.hasSeq()) {
      seqInfos_[i] = commonSeqInfo_;
    }
    createInFrameInfo(i, input, passType);
  }

  {
    AsyncGpuBlock asyncGpuBlock;

    // inFrameLine select rows in real layer one time
    for (size_t i = 0; i < inFrameLines_.size(); i++) {
      selectRowsOneTime(inFrameLines_[i].inLayer,
                        info_[i].allIds,
                        &(inFrameLines_[i].outArg),
                        passType);
    }
  }
}

void RecurrentGradientMachine::reorganizeOutput(PassType passType) {
  calcSequenceStartPositions();
  for (size_t i = 0; i < outFrameLines_.size(); ++i) {
    Info info;
    auto& outFrameLine = outFrameLines_[i];
    ICpuGpuVectorPtr sequenceStartPositions;
    ICpuGpuVectorPtr subSequenceStartPositions;
    createOutFrameInfo(
        outFrameLine, info, sequenceStartPositions, subSequenceStartPositions);
    auto gatherAgent =
        dynamic_cast<GatherAgentLayer*>(outFrameLine.agentLayer.get());
    CHECK_NOTNULL(gatherAgent);
    gatherAgent->copyIdAndSequenceInfo(sequenceStartPositions,
                                       subSequenceStartPositions,
                                       info.allIds,
                                       info.idIndex);
  }
}

void RecurrentGradientMachine::connectFrames(PassType passType) {
  for (auto& memoryFrameLine : memoryFrameLines_) {
    if (memoryFrameLine.rootAgent) {
      auto scatterAgent =
          dynamic_cast<ScatterAgentLayer*>(memoryFrameLine.rootAgent.get());
      createMemoryFrameInfo(&memoryFrameLine, passType);
      scatterAgent->setRealLayerAndOutput(memoryFrameLine.rootLayer,
                                          memoryFrameLine.outArg,
                                          memoryFrameLine.allIds,
                                          /* idIndex */ 0,
                                          memoryFrameLine.allIds->getSize(),
                                          /* handleBackward */ true);
      if (memoryFrameLine.sequenceStartPositions) {
        int size = memoryFrameLine.sequenceStartPositions->getSize();
        scatterAgent->setSequenceStartPositions(
            memoryFrameLine.sequenceStartPositions,
            /* seqStartPosIndex */ 0,
            size);
      }
    }
  }

  for (auto& outFrameLine : outFrameLines_) {
    auto gatherAgent =
        dynamic_cast<GatherAgentLayer*>(outFrameLine.agentLayer.get());
    gatherAgent->clearRealLayers();
  }
  for (int i = 0; i < maxSequenceLength_; ++i) {
    // connect in_links
    for (size_t j = 0; j < inFrameLines_.size(); ++j) {
      Info& info = info_[j];
      // idSize denotes the sum number of tokens in each length i
      int idIndex = info.idIndex.empty() ? 0 : info.idIndex[i];
      int idSize = info.idIndex.empty() ? numSeqs_[i]
                                        : info.idIndex[i + 1] - info.idIndex[i];
      InFrameLine inFrameLine = inFrameLines_[j];
      auto scatterAgent =
          dynamic_cast<ScatterAgentLayer*>(inFrameLine.agents[i].get());
      scatterAgent->setRealLayerAndOutput(inFrameLine.inLayer,
                                          inFrameLine.outArg,
                                          info.allIds,
                                          idIndex,
                                          idSize,
                                          i == 0);
      if (info.sequenceStartPositions) {
        // size: the length of subsequence
        int size = info.seqStartPosIndex[i + 1] - info.seqStartPosIndex[i];
        scatterAgent->setSequenceStartPositions(
            info.sequenceStartPositions, info.seqStartPosIndex[i], size);
      }
    }

    // connect out_links
    for (auto& outFrameLine : outFrameLines_) {
      auto gatherAgent =
          dynamic_cast<GatherAgentLayer*>(outFrameLine.agentLayer.get());
      gatherAgent->addRealLayer(outFrameLine.frames[i]);
    }
    for (auto& memoryFrameLine : memoryFrameLines_) {
      NeuralNetwork::connect(
          memoryFrameLine.agents[i],
          i == 0 ? memoryFrameLine.bootLayer : memoryFrameLine.frames[i - 1],
          numSeqs_[i] /*height of agent*/);
    }
  }
}

void RecurrentGradientMachine::forward(const std::vector<Argument>& inArgs,
                                       std::vector<Argument>* outArgs,
                                       PassType passType) {
  /* inArgs and outArgs are not used.
     The inputs are inFrameLines_[i].inLayer.
     The outputs are outFramesLines_[i].agentLayer
   */

  if (generating_) {
    generateSequence();
    return;
  }  // else forward..

  reorganizeInput(passType);
  int numSequences = commonSeqInfo_.size();

  resizeOrCreateFrames(maxSequenceLength_);
  resizeBootFrame(numSequences);

  connectFrames(passType);

  REGISTER_TIMER_INFO("RecurrentFwTime", "RecurrentFwTime");
  // forward
  for (auto& memoryFrameLine : memoryFrameLines_) {
    memoryFrameLine.bootLayer->forward(passType);
  }
  for (int i = 0; i < maxSequenceLength_; ++i) {
    const std::vector<Argument> inArgs;
    std::vector<Argument> outArgs;
    frames_[i]->forward(inArgs, &outArgs, passType);
  }

  reorganizeOutput(passType);
}

void RecurrentGradientMachine::backward(const UpdateCallback& callback) {
  if (generating_) {
    return;
  }
  REGISTER_TIMER_INFO("RecurrentBwTime", "RecurrentBwTime");
  AsyncGpuBlock asyncGpuBlock;
  for (int i = maxSequenceLength_ - 1; i >= 0; --i) {
    frames_[i]->backward(nullptr);
  }
  for (auto& memoryFrameLine : memoryFrameLines_) {
    memoryFrameLine.bootLayer->backward(nullptr);
  }
}

void RecurrentGradientMachine::forwardBackward(
    const std::vector<Argument>& inArgs,
    std::vector<Argument>* outArgs,
    PassType passType,
    const UpdateCallback& callback) {
  LOG(FATAL) << "should not use this function";
}

void RecurrentGradientMachine::eval(Evaluator* evaluator) const {
  // call printers frame by frame
  for (int i = 0; i < maxSequenceLength_; ++i) {
    VLOG(2) << "Recurrent Layer Group eval frame " << i << " begin";
    evaluator->eval(*(frames_[i].get()));
    VLOG(2) << "Recurrent Layer Group eval frame " << i << " end";
  }
}

void RecurrentGradientMachine::registerBeamSearchControlCallbacks(
    const BeamSearchCandidatesAdjustCallback& adjustBeamSearch,
    const NormOrDropNodeCallback& normOrDropNode,
    const DropCallback& stopBeamSearch) {
  this->removeBeamSearchControlCallbacks();
  //! for gcc 46, aggregate initialization is not supported. TAT
  this->beamSearchCtrlCallbacks_ = new BeamSearchControlCallbacks(
      adjustBeamSearch, normOrDropNode, stopBeamSearch);
}

void RecurrentGradientMachine::removeBeamSearchControlCallbacks() {
  if (this->beamSearchCtrlCallbacks_) {
    delete this->beamSearchCtrlCallbacks_;
    this->beamSearchCtrlCallbacks_ = nullptr;
  }
}

void RecurrentGradientMachine::registerBeamSearchStatisticsCallbacks(
    const EachStepCallback& onEachStepStarted,
    const EachStepCallback& onEachStepStoped) {
  this->removeBeamSearchStatisticsCallbacks();
  this->beamSearchStatistics_ =
      new BeamSearchStatisticsCallbacks(onEachStepStarted, onEachStepStoped);
}

void RecurrentGradientMachine::removeBeamSearchStatisticsCallbacks() {
  if (this->beamSearchStatistics_) {
    delete this->beamSearchStatistics_;
    this->beamSearchStatistics_ = nullptr;
  }
}

namespace {
void lenToStarts(std::vector<int>& starts) {
  int pos = 0;
  starts.back() = 0;
  for (auto& start : starts) {
    int tmp = start;
    start = pos;
    pos += tmp;
  }
  starts.back() = pos;
}
}  // namespace

void RecurrentGradientMachine::calcSequenceStartPositions() {
  std::vector<int> starts(commonSeqInfo_.size() + 1);
  for (auto& seqInfo : commonSeqInfo_) {
    starts[seqInfo.seqId] = seqInfo.topLevelLength;
  }
  lenToStarts(starts);
  ICpuGpuVector::resizeOrCreate(sequenceStartPositions_, starts.size(), false);
  std::copy(starts.begin(),
            starts.end(),
            sequenceStartPositions_->getMutableData(false));
}

void RecurrentGradientMachine::checkOutputConsistency(
    OutFrameLine& outFrameLine) {
  bool hasSeq = outFrameLine.frames[0]->getOutput().hasSeq();
  for (int i = 0; i < maxSequenceLength_; ++i) {
    LayerPtr frame = outFrameLine.frames[i];
    CHECK_EQ(hasSeq, frame->getOutput().hasSeq());
    int numSequences = frame->getOutput().getNumSequences();
    CHECK_EQ(numSeqs_[i], numSequences);
  }
}

void RecurrentGradientMachine::createOutFrameInfo(
    OutFrameLine& outFrameLine,
    Info& info,
    ICpuGpuVectorPtr& sequenceStartPositions,
    ICpuGpuVectorPtr& subSequenceStartPositions) {
  checkOutputConsistency(outFrameLine);

  if (!outFrameLine.frames[0]->getOutput().hasSeq()) {
    createOutFrameInfo_seq(
        outFrameLine, info, sequenceStartPositions, subSequenceStartPositions);
  } else {
    createOutFrameInfo_subseq(
        outFrameLine, info, sequenceStartPositions, subSequenceStartPositions);
  }
}

void RecurrentGradientMachine::createOutFrameInfo_seq(
    OutFrameLine& outFrameLine,
    Info& info,
    ICpuGpuVectorPtr& sequenceStartPositions,
    ICpuGpuVectorPtr& subSequenceStartPositions) {
  std::vector<int> allIds;
  info.idIndex.resize(1, 0);  // first idIndex = 0

  const int* starts = sequenceStartPositions_->getData(false);

  for (int i = 0; i < maxSequenceLength_; ++i) {
    LayerPtr frame = outFrameLine.frames[i];
    size_t numSequences = frame->getOutput().getNumSequences();
    for (size_t j = 0; j < numSequences; ++j) {
      int seqStart = starts[commonSeqInfo_[j].seqId];
      int seqLength = commonSeqInfo_[j].topLevelLength;
      allIds.push_back(reversed_ ? (seqStart + seqLength - 1 - i)
                                 : (seqStart + i));
    }
    info.idIndex.push_back(allIds.size());
  }
  sequenceStartPositions = sequenceStartPositions_;
  copyScattedId(allIds, &info.allIds, allIds.size());
  CHECK_EQ(info.idIndex.size(), static_cast<size_t>(maxSequenceLength_ + 1));
}

void RecurrentGradientMachine::createOutFrameInfo_subseq(
    OutFrameLine& outFrameLine,
    Info& info,
    ICpuGpuVectorPtr& sequenceStartPositions,
    ICpuGpuVectorPtr& subSequenceStartPositions) {
  size_t numSequences = commonSeqInfo_.size();
  std::vector<int> allIds;
  info.idIndex.resize(1, 0);  // first idIndex = 0

  const int* starts = sequenceStartPositions_->getData(false);
  std::vector<int> subStarts(starts[numSequences] + 1);
  for (int i = 0; i < maxSequenceLength_; ++i) {
    LayerPtr frame = outFrameLine.frames[i];
    size_t numSequences = frame->getOutput().getNumSequences();
    const int* seqStarts =
        frame->getOutput().sequenceStartPositions->getData(false);
    for (size_t j = 0; j < numSequences; ++j) {
      subStarts[starts[commonSeqInfo_[j].seqId] + i] =
          seqStarts[j + 1] - seqStarts[j];
    }
  }
  lenToStarts(subStarts);

  for (int i = 0; i < maxSequenceLength_; ++i) {
    LayerPtr frame = outFrameLine.frames[i];
    size_t numSequences = frame->getOutput().getNumSequences();
    for (size_t j = 0; j < numSequences; ++j) {
      int pos = starts[commonSeqInfo_[j].seqId] + i;
      int subSeqStart = subStarts[pos];
      int subSeqEnd = subStarts[pos + 1];
      for (int k = subSeqStart; k < subSeqEnd; ++k) {
        allIds.push_back(k);
      }
    }
    info.idIndex.push_back(allIds.size());
  }

  ICpuGpuVector::resizeOrCreate(
      subSequenceStartPositions, subStarts.size(), false);
  int* cpuSubSequenceStartPositions =
      subSequenceStartPositions->getMutableData(false);
  std::copy(subStarts.begin(), subStarts.end(), cpuSubSequenceStartPositions);
  ICpuGpuVector::resizeOrCreate(
      sequenceStartPositions, numSequences + 1, false);
  int* cpuSequenceStartPositions =
      sequenceStartPositions->getMutableData(false);
  for (size_t i = 0; i <= numSequences; ++i) {
    cpuSequenceStartPositions[i] = subStarts[starts[i]];
  }
  copyScattedId(allIds, &info.allIds, allIds.size());
  CHECK_EQ(info.idIndex.size(), static_cast<size_t>(maxSequenceLength_ + 1));
}

/* create scattered id infomation for all realLayer of inFrameLines one time.
 * If hasSubseq, will also create scattered sequenceStartPositions infomation
 * for all realLayer of inFrameLines one time.
 */
void RecurrentGradientMachine::createInFrameInfo(int inlinkId,
                                                 const Argument& input,
                                                 PassType passType) {
  if (!input.hasSeq()) {
    createInFrameInfo_nonseq(inlinkId, input, passType);
  } else if (!input.hasSubseq()) {
    createInFrameInfo_seq(inlinkId, input, passType);
  } else {
    createInFrameInfo_subseq(inlinkId, input, passType);
  }
}

void RecurrentGradientMachine::createInFrameInfo_nonseq(int inlinkId,
                                                        const Argument& input,
                                                        PassType passType) {
  std::vector<int> allIds;

  auto& seqInfo = seqInfos_[inlinkId];
  Info* inlinkInfo = &info_[inlinkId];
  inlinkInfo->idIndex.clear();
  for (size_t i = 0; i < seqInfo.size(); ++i) {
    allIds.push_back(seqInfo[i].seqId);
  }
  // copy and check scatterId
  copyScattedId(allIds, &inlinkInfo->allIds, input.getBatchSize());
}

void RecurrentGradientMachine::createInFrameInfo_seq(int inlinkId,
                                                     const Argument& input,
                                                     PassType passType) {
  std::vector<int> allIds;
  auto& seqInfo = seqInfos_[inlinkId];
  Info* inlinkInfo = &info_[inlinkId];
  inlinkInfo->idIndex.resize(1, 0);  // first idIndex = 0

  for (int i = 0; i < maxSequenceLength_; ++i) {
    for (int j = 0; j < numSeqs_[i]; ++j) {
      int seqLength = seqInfo[j].topLevelLength;
      int seqStart = seqInfo[j].seqStart;
      allIds.push_back(reversed_ ? (seqStart + seqLength - 1 - i)
                                 : (seqStart + i));
    }
    inlinkInfo->idIndex.push_back(allIds.size());
  }

  // copy and check scatterId
  copyScattedId(allIds, &inlinkInfo->allIds, input.getBatchSize());
  CHECK_EQ(inlinkInfo->idIndex.size(),
           static_cast<size_t>(maxSequenceLength_ + 1));
}
void RecurrentGradientMachine::createInFrameInfo_subseq(int inlinkId,
                                                        const Argument& input,
                                                        PassType passType) {
  std::vector<int> allIds;

  auto& seqInfo = seqInfos_[inlinkId];

  Info* inlinkInfo = &info_[inlinkId];
  inlinkInfo->idIndex.resize(1, 0);  // first idIndex = 0
  std::vector<int> sequenceStartPositions;
  const int* subSequenceStartPositions = nullptr;

  subSequenceStartPositions = input.subSequenceStartPositions->getData(false);
  inlinkInfo->seqStartPosIndex.clear();
  inlinkInfo->seqStartPosIndex.push_back(0);  // first seqStartPosIndex = 0
  for (int i = 0; i < maxSequenceLength_; ++i) {
    sequenceStartPositions.push_back(0);  // first element = 0
    for (int j = 0; j < numSeqs_[i]; ++j) {
      int subSeqStart = subSequenceStartPositions[seqInfo[j].subSeqStart + i];
      int subSeqEnd = subSequenceStartPositions[seqInfo[j].subSeqStart + i + 1];
      for (int k = subSeqStart; k < subSeqEnd; ++k) {
        allIds.push_back(k);
      }
      sequenceStartPositions.push_back(sequenceStartPositions.back() +
                                       subSeqEnd - subSeqStart);
    }
    inlinkInfo->idIndex.push_back(allIds.size());
    inlinkInfo->seqStartPosIndex.push_back(sequenceStartPositions.size());
  }
  // inFrameLine create sequenceStartPositions one time
  CHECK_EQ(
      sequenceStartPositions.size(),
      static_cast<size_t>(maxSequenceLength_ + input.getNumSubSequences()));
  CHECK_EQ(inlinkInfo->seqStartPosIndex.size(),
           static_cast<size_t>(maxSequenceLength_ + 1));
  createSeqPos(sequenceStartPositions, &inlinkInfo->sequenceStartPositions);

  // copy and check scatterId
  copyScattedId(allIds, &inlinkInfo->allIds, input.getBatchSize());
  CHECK_EQ(inlinkInfo->idIndex.size(),
           static_cast<size_t>(maxSequenceLength_ + 1));
}

/* like createInFrameInfo, but for all realLayer of memoryFrameLines*/
void RecurrentGradientMachine::createMemoryFrameInfo(
    MemoryFrameLine* memoryFrameLine, PassType passType) {
  const Argument& input = (*memoryFrameLine).rootLayer->getOutput();
  size_t numSequences = input.getNumSequences();
  std::vector<int> allIds;
  bool seqFlag = input.hasSeq();
  CHECK(!input.hasSubseq())
      << "Subsequence boot layer for memory is not supported";

  if (seqFlag) {  // for sequenceScatterAgentLayer
    std::vector<int> sequenceStartPositions;
    sequenceStartPositions.push_back(0);  // first element = 0
    const int* starts = input.sequenceStartPositions->getData(false);
    for (size_t i = 0; i < numSequences; ++i) {
      // memory info adopt info of inlinks[0]
      int seqId = seqInfos_[0][i].seqId;
      for (int k = starts[seqId]; k < starts[seqId + 1]; ++k) {
        allIds.push_back(k);
      }
      sequenceStartPositions.push_back(sequenceStartPositions.back() +
                                       starts[seqId + 1] - starts[seqId]);
    }
    createSeqPos(sequenceStartPositions,
                 &(*memoryFrameLine).sequenceStartPositions);

  } else {  // for scatterAgentLayer
    for (size_t i = 0; i < numSequences; ++i) {
      allIds.push_back(seqInfos_[0][i].seqId);
    }
  }
  // copy and check scatterId
  copyScattedId(allIds, &(*memoryFrameLine).allIds, input.getBatchSize());
  // memoryFrameLine select rows in real layer one time
  selectRowsOneTime((*memoryFrameLine).rootLayer,
                    (*memoryFrameLine).allIds,
                    &(*memoryFrameLine).outArg,
                    passType);
}

void RecurrentGradientMachine::copyScattedId(std::vector<int>& srcIds,
                                             IVectorPtr* dstIds,
                                             int size) {
  int idSize = srcIds.size();
  CHECK_EQ(idSize, size);
  IVector::resizeOrCreate(*dstIds, idSize, useGpu_);
  (*dstIds)->copyFrom(srcIds.data(), idSize);
  // check
  std::sort(srcIds.begin(), srcIds.end());
  for (int i = 0; i < idSize; ++i) {
    CHECK_EQ(srcIds[i], i);
  }
}

void RecurrentGradientMachine::selectRowsOneTime(LayerPtr layer,
                                                 const IVectorPtr& allIds,
                                                 Argument* arg,
                                                 PassType passType) {
  Argument& src = layer->getOutput();
  if (src.value) {
    const MatrixPtr& realV = src.value;
    int height = realV->getHeight();
    int width = realV->getWidth();
    Matrix::resizeOrCreate(
        arg->value, height, width, /* trans */ false, useGpu_);
    arg->value->zeroMem();
    arg->value->selectRows(*realV, *allIds);
    if (passType != PASS_TEST) {
      Matrix::resizeOrCreate(
          arg->grad, height, width, /* trans */ false, useGpu_);
      arg->grad->zeroMem();
    }
  }
  if (src.ids) {
    IVector::resizeOrCreate(arg->ids, src.ids->getSize(), useGpu_);
    arg->ids->selectFrom(*src.ids, *allIds);
  }
}

void RecurrentGradientMachine::createSeqPos(
    const std::vector<int>& sequenceStartPosition,
    ICpuGpuVectorPtr* sequenceStartPositions) {
  int size = sequenceStartPosition.size();
  const int* data = sequenceStartPosition.data();
  ICpuGpuVector::resizeOrCreate(*sequenceStartPositions, size, false);
  (*sequenceStartPositions)->copyFrom(data, size, false);
}

size_t RecurrentGradientMachine::getGenBatchSize() {
  size_t numSequences = 0;
  for (auto& memoryFrameLine : memoryFrameLines_) {
    if (!memoryFrameLine.rootLayer) continue;
    Argument& bootArg = memoryFrameLine.rootLayer->getOutput();
    size_t batchSize = bootArg.getNumSequences();
    if (numSequences) {
      CHECK_EQ(numSequences, batchSize);
    } else {
      numSequences = batchSize;
    }
  }
  CHECK(numSequences)
      << "Fail to get batch size in generation. "
         "At least one of the Memory layer MUST have a layer that is NOT in "
         "the layer group to boot it, and this boot layer is used to "
         "decide batch_size in generation process.";
  return numSequences;
}

void RecurrentGradientMachine::generateSequence() {
  CHECK_NOTNULL(eosFrameLine_.get());
  CHECK_GE(outFrameLines_.size(), 1UL);
  size_t numSequences = getGenBatchSize();

  resizeBootFrame(numSequences);
  // We create only two sub-network in generation, one stores states of all
  // layers in previous time step and the other storing the states at current
  // time step.
  resizeOrCreateFrames(2);

  // outFrameLines_.size() > 1UL
  dataArgsSize_ = outFrameLines_.size() - 1;
  dataArgs_.resize(dataArgsSize_);
  dataArgsFrame_.clear();
  dataArgsFrame_.resize(dataArgsSize_);

  // connect boot frame memory links
  std::vector<int> ids(numSequences);
  for (size_t i = 0; i < numSequences; ++i) {
    ids[i] = i;
  }
  for (auto& memoryFrameLine : memoryFrameLines_) {
    if (memoryFrameLine.rootAgent) {
      auto scatterAgent =
          dynamic_cast<ScatterAgentLayer*>(memoryFrameLine.rootAgent.get());
      scatterAgent->setRealLayer(memoryFrameLine.rootLayer, ids);
    }
    NeuralNetwork::connect(
        memoryFrameLine.agents[0], memoryFrameLine.bootLayer, ids.size());
  }

  // boot layer forward
  AsyncGpuBlock asyncGpuBlock;

  for (auto& memoryFrameLine : memoryFrameLines_) {
    memoryFrameLine.bootLayer->forward(PASS_TEST);
  }

  // init outArg
  size_t resultNum = generator_.config.num_results_per_sample();
  size_t maxGenWordCount =
      generator_.config.max_num_frames() * numSequences * resultNum;
  IVector::resizeOrCreate(generator_.outArg.ids, maxGenWordCount, false);
  if (resultNum > 1) {
    CHECK_LE(resultNum, static_cast<size_t>(generator_.config.beam_size()));
    Matrix::resizeOrCreate(generator_.outArg.in,
                           /* height */ numSequences,
                           /* width */ resultNum,
                           false,
                           /* useGpu */ false);
  }
  ICpuGpuVector::resizeOrCreate(generator_.outArg.sequenceStartPositions,
                                numSequences + 1,
                                /* useGpu */ false);
  if (getBeamSize() > 1) {
    beamSearch(numSequences);
  } else {
    oneWaySearch(numSequences);
  }
  if (dataArgsSize_) createDataOutlink();

  size_t size = generator_.ids.size();
  generator_.outArg.ids->resize(size);
  generator_.outArg.ids->copyFrom(generator_.ids.data(), size);

  OutFrameLine& outFrameLine = outFrameLines_[0];
  auto dataAgent = dynamic_cast<DataLayer*>(outFrameLine.agentLayer.get());
  CHECK_NOTNULL(dataAgent);
  dataAgent->setData(generator_.outArg);
  dataAgent->prefetch();
}

void RecurrentGradientMachine::oneWaySearch(size_t batchSize) {
  OutFrameLine& outFrameLine = outFrameLines_[0];

  // finalPaths_[0] stores the generated results of the
  // entire batch, so its size exactly equals to batchSize.
  finalPaths_.clear();
  finalPaths_.resize(1);
  std::vector<Path>& finalPaths = finalPaths_[0];
  finalPaths.resize(batchSize);

  seqIds_.resize(batchSize);
  std::vector<int> scatterIds;
  for (size_t i = 0; i < batchSize; ++i) {
    finalPaths[i].seqId = i;
    seqIds_[i] = i;
  }

  // forward
  for (int i = 0; i < maxSequenceLength_; ++i) {
    if (i && scatterIds.empty()) break;
    int machineCur = i % 2;
    int machinePrev = (i - 1) % 2;
    // connect memory links
    if (i) {
      seqIds_.clear();
      for (size_t j = 0; j < batchSize; ++j) {
        if (finalPaths[j].seqId != -1) seqIds_.push_back(j);
      }

      for (auto& memoryFrameLine : memoryFrameLines_) {
        auto scatterAgent = dynamic_cast<ScatterAgentLayer*>(
            memoryFrameLine.scatterAgents[machineCur].get());
        scatterAgent->setRealLayer(memoryFrameLine.frames[machinePrev],
                                   scatterIds);
        scatterAgent->forward(PASS_TEST);
        NeuralNetwork::connect(memoryFrameLine.agents[machineCur],
                               memoryFrameLine.scatterAgents[machineCur]);
      }
    }
    const std::vector<Argument> inArgs;
    std::vector<Argument> outArgs;
    frames_[machineCur]->forward(inArgs, &outArgs, PASS_TEST);

    const IVectorPtr& idVec = outFrameLine.frames[machineCur]->getOutput().ids;
    for (size_t j = 0; j < seqIds_.size(); ++j) {
      finalPaths[seqIds_[j]].ids.push_back(idVec->getElement(j));
      finalPaths[seqIds_[j]].machineIdVec.push_back(j);
    }

    copyDataOutlinkFrame(machineCur);

    // check eos
    const IVectorPtr& eosVec =
        eosFrameLine_->layers[machineCur]->getOutput().ids;
    scatterIds.clear();
    for (size_t j = 0; j < seqIds_.size(); ++j) {
      if (eosVec->getElement(j) == 1U) {
        // path.seqId = -1 indicates end of generation
        // of an input sequence
        finalPaths[seqIds_[j]].seqId = -1;
      } else {
        scatterIds.push_back(j);
      }
    }
  }

  batchMachineIdVec_.clear();
  batchMachineStartPos_.clear();
  int* starts = generator_.outArg.sequenceStartPositions->getMutableData(false);
  starts[0] = 0;
  generator_.ids.clear();
  for (size_t i = 0; i < batchSize; ++i) {
    generator_.ids.insert(generator_.ids.end(),
                          finalPaths[i].ids.begin(),
                          finalPaths[i].ids.end());
    starts[i + 1] = generator_.ids.size();
    batchMachineIdVec_.insert(batchMachineIdVec_.end(),
                              finalPaths[i].machineIdVec.begin(),
                              finalPaths[i].machineIdVec.end());
  }
}

void RecurrentGradientMachine::connectPrevFrame(int stepId,
                                                std::vector<Path>& paths) {
  int machineCur = stepId % 2;
  int machinePrev = (stepId - 1) % 2;
  int beam = getBeamSize();
  machineIds_.clear();
  topIds_.clear();
  seqIds_.clear();

  for (size_t j = 0; j < paths.size(); ++j) {
    machineIds_.push_back(paths[j].machineId);
    topIds_.push_back(paths[j].machineId * beam + paths[j].topIndex);
    seqIds_.push_back(paths[j].seqId);
  }

  for (auto& memoryFrameLine : memoryFrameLines_) {
    bool isOutIds = (memoryFrameLine.layerName == outFrameLines_[0].layerName);
    auto scatterAgent = dynamic_cast<ScatterAgentLayer*>(
        memoryFrameLine.scatterAgents[machineCur].get());
    scatterAgent->setRealLayer(memoryFrameLine.frames[machinePrev],
                               isOutIds ? topIds_ : machineIds_);
    scatterAgent->forward(PASS_TEST);
    NeuralNetwork::connect(memoryFrameLine.agents[machineCur],
                           memoryFrameLine.scatterAgents[machineCur]);
  }
}

void RecurrentGradientMachine::forwardFrame(int machineCur) {
  // forward
  const std::vector<Argument> inArgs;
  std::vector<Argument> outArgs;
  frames_[machineCur]->forward(inArgs, &outArgs, PASS_TEST);

  copyDataOutlinkFrame(machineCur);

  IVectorPtr& ids = outFrameLines_[0].frames[machineCur]->getOutput().ids;
  MatrixPtr in = outFrameLines_[0].frames[machineCur]->getOutput().in;
  IVectorPtr& eos = eosFrameLine_->layers[machineCur]->getOutput().ids;
  if (useGpu_) {
    IVector::resizeOrCreate(cpuId_, ids->getSize(), false /* useGpu */);
    cpuId_->copyFrom(*ids);
    Matrix::resizeOrCreate(cpuProb_,
                           in->getHeight(),
                           in->getWidth(),
                           false /* trans */,
                           false /* useGpu */);
    cpuProb_->copyFrom(*in);
    IVector::resizeOrCreate(cpuEos_, eos->getSize(), false /* useGpu */);
    cpuEos_->copyFrom(*eos);
  } else {
    cpuId_ = ids;
    cpuProb_ = in;
    cpuEos_ = eos;
  }
}

void RecurrentGradientMachine::singlePathExpand(Path& curPath,
                                                size_t curPathId,
                                                std::vector<Path>& newPaths,
                                                size_t expandWidth) {
  int calc_id =
      gDiyProbStart ? gDiyProbStart(curPath.ids.size(), curPath.ids.data()) : 0;

  const int* idVec = cpuId_->getData();
  const real* probMat = cpuProb_->getData();
  const int* eosVec = cpuEos_->getData();

  for (size_t k = 0; k < expandWidth; k++) {
    int index = curPathId * expandWidth + k;
    int id = idVec[index];
    real prob = probMat[index];
    /*
     * Ordinarily, beam search greedily expands the most promising expandWidth
     * paths that currently are ALWAYS returned by MaxIdLayer.
     * In one condition, if user customizes the beam search procedure by
     * restricting the expansion within a user defined subset,
     * as a result, MaxIdLayer possibly COULD NOT return expandWidth
     * vaild expansions, and it will use -1 to indicate the end of valid
     * expansion candidates.
     */
    if (id == -1) break;

    real newLogProb = generator_.config.log_prob() ? std::log(prob) : prob;
    Path newPath(
        curPath, id, newLogProb, curPathId /*machineId*/, k /*topIndex*/);
    if (this->beamSearchCtrlCallbacks_) {
      if (beamSearchCtrlCallbacks_->stopDetermineCandidates(
              newPath.seqId, newPath.ids, newPath.probHistory))
        return;
    }
    // outFrameLines_.size() > 1UL
    if (dataArgsSize_) {
      newPath.machineIdVec = curPath.machineIdVec;
      newPath.machineIdVec.push_back(curPathId);
    }
    bool atEos =
        eosVec[index] == 1U || newPath.ids.size() >= (size_t)maxSequenceLength_;
    // adjustNewPath
    newPath.adjustProb(calc_id, atEos);
    if (this->beamSearchCtrlCallbacks_) {
      this->beamSearchCtrlCallbacks_->normOrDropNode(
          newPath.seqId, newPath.ids, newPath.probHistory, &newPath.logProb);
    }
    if (!newPath.isDropable()) {
      atEos ? finalPaths_[curPath.seqId].push_back(newPath)
            : newPaths.push_back(newPath);
    }
  }  // for expandWidth

  if (gDiyProbStop) {
    gDiyProbStop(calc_id);
  }
}

void RecurrentGradientMachine::beamExpand(std::vector<Path>& paths,
                                          std::vector<Path>& newPaths) {
  size_t candidatePathCount = paths.size();
  // idVec.size() could be larger than candidatePathCount * beam,
  // so user can drop some node customly.
  CHECK_EQ(cpuId_->getSize() % candidatePathCount, 0UL);
  size_t expandWidth = cpuId_->getSize() / candidatePathCount;

  // iterate over each sequence
  size_t totalExpandCount = 0;
  int prevSeqId = -1;
  int curSeqId = 0;
  for (size_t j = 0; j <= candidatePathCount; j++) {
    // expansions of a single sequence are all processed
    curSeqId = (j < candidatePathCount ? paths[j].seqId : curSeqId + 1);
    if (prevSeqId != -1 && curSeqId != prevSeqId) {
      totalExpandCount += beamShrink(newPaths, prevSeqId, totalExpandCount);
    }
    if (j == candidatePathCount) return;
    singlePathExpand(paths[j], j, newPaths, expandWidth);

    prevSeqId = paths[j].seqId;
  }  // for paths
}

// Drop extra nodes to beam size.
size_t RecurrentGradientMachine::beamShrink(std::vector<Path>& newPaths,
                                            size_t seqId,
                                            size_t totalExpandCount) {
  size_t minNewPathSize =
      std::min(getBeamSize(), newPaths.size() - totalExpandCount);
  if (!minNewPathSize) {
    return 0;
  }
  std::nth_element(newPaths.begin() + totalExpandCount,
                   newPaths.begin() + totalExpandCount + minNewPathSize,
                   newPaths.end(),
                   Path::greaterPath);
  newPaths.resize(totalExpandCount + minNewPathSize);

  real minPathLogProb =
      std::min_element(newPaths.end() - minNewPathSize, newPaths.end())
          ->logProb;
  real maxPathLogProb =
      std::max_element(newPaths.end() - minNewPathSize, newPaths.end())
          ->logProb;

  // Remove the already formed paths that are relatively short
  finalPaths_[seqId].erase(
      std::remove_if(finalPaths_[seqId].begin(),
                     finalPaths_[seqId].end(),
                     [&](Path& p) { return p.logProb < minPathLogProb; }),
      finalPaths_[seqId].end());
  for (auto p : finalPaths_[seqId]) {
    if (minFinalPathLogProb_[seqId] > p.logProb) {
      minFinalPathLogProb_[seqId] = p.logProb;
    }
  }

  if (finalPaths_[seqId].size() >= getBeamSize() &&
      minFinalPathLogProb_[seqId] >= maxPathLogProb) {
    newPaths.resize(totalExpandCount);
    return 0;
  }
  return minNewPathSize;
}

void RecurrentGradientMachine::fillGenOutputs() {
  size_t numResults = generator_.config.num_results_per_sample();
  for (size_t i = 0; i < finalPaths_.size(); ++i) {
    size_t minFinalPathsSize = std::min(numResults, finalPaths_[i].size());
    std::partial_sort(finalPaths_[i].begin(),
                      finalPaths_[i].begin() + minFinalPathsSize,
                      finalPaths_[i].end(),
                      Path::greaterPath);
    finalPaths_[i].resize(minFinalPathsSize);
  }

  generator_.ids.clear();
  int* starts = generator_.outArg.sequenceStartPositions->getMutableData(false);
  starts[0] = 0;
  if (numResults > 1) {
    int idsProbSaveSize = 0;
    for (auto inSeq : finalPaths_) {
      for (auto path : inSeq) idsProbSaveSize += path.ids.size();
      idsProbSaveSize += inSeq.size();
    }
    Matrix::resizeOrCreate(
        generator_.outArg.value, idsProbSaveSize, 1, false, false);
    real* idsProb = generator_.outArg.value->getData();

    real* probs = generator_.outArg.in->getData();
    size_t curPos = 0;
    for (size_t i = 0; i < finalPaths_.size(); ++i) {
      for (size_t j = 0; j < finalPaths_[i].size(); ++j) {
        Path& path = finalPaths_[i][j];
        size_t genLen = path.ids.size();
        generator_.ids.push_back(genLen);  // sequence size
        generator_.ids.insert(
            generator_.ids.end(), path.ids.begin(), path.ids.end());
        generator_.ids.push_back(-1);  // end of sequence

        memcpy(idsProb + curPos, path.idsProb.data(), sizeof(real) * genLen);
        curPos += genLen;
        idsProb[curPos++] = -1.0;
        probs[i * numResults + j] = path.logProb;
      }
      starts[i + 1] = generator_.ids.size();
    }
  } else {
    for (size_t i = 0; i < finalPaths_.size(); ++i) {
      CHECK(!finalPaths_[i].empty());
      Path& path = finalPaths_[i][0];
      generator_.ids.insert(
          generator_.ids.end(), path.ids.begin(), path.ids.end());
      starts[i + 1] = starts[i] + path.ids.size();
    }
  }
}

void RecurrentGradientMachine::copyDataOutlinkFrame(size_t machineCur) {
  for (size_t i = 0; i < dataArgsSize_; i++) {
    Argument outFrame;
    outFrame.resizeAndCopyFrom(
        outFrameLines_[i + 1].frames[machineCur]->getOutput(), useGpu_);
    dataArgsFrame_[i].emplace_back(outFrame);
  }
}

void RecurrentGradientMachine::createDataOutlinkSelRowsInfo(
    bool isSeq, std::vector<Argument>& outArgs) {
  batchMachineIdVec_.clear();

  size_t seqIdx = 0;
  for (size_t i = 0; i < finalPaths_.size(); ++i) {
    for (size_t j = 0; j < finalPaths_[i].size(); ++j) {
      std::vector<int>& machineIdVec = finalPaths_[i][j].machineIdVec;
      if (isSeq) {
        for (size_t i = 0; i < machineIdVec.size(); ++i) {
          size_t rowId = machineIdVec[i];
          int* seqPos =
              outArgs[i].sequenceStartPositions->getMutableData(false);
          batchMachineIdVec_.push_back(seqPos[rowId]);
        }
      } else {
        batchMachineIdVec_.insert(
            batchMachineIdVec_.end(), machineIdVec.begin(), machineIdVec.end());
      }
      seqIdx++;
    }
  }
}

void RecurrentGradientMachine::createDataOutlinkCopySizeInfo(
    bool isSeq, std::vector<Argument>& outArgs, std::vector<int>& copySize) {
  size_t totalSeqNum = std::accumulate(
      finalPaths_.begin(),
      finalPaths_.end(),
      0UL,
      [](size_t a, const std::vector<Path>& b) { return a + b.size(); });
  copySize.resize(totalSeqNum, 1);

  batchMachineStartPos_.resize(totalSeqNum + 1, 0);
  if (isSeq) {
    ICpuGpuVectorPtr inputSeqStartPos = outArgs[0].sequenceStartPositions;
    CHECK_EQ(static_cast<size_t>(inputSeqStartPos->getSize() - 1),
             getBeamSize() > 1 ? finalPaths_.size() : finalPaths_[0].size());
    int* starts = inputSeqStartPos->getMutableData(false);
    int seqId = 0;
    for (size_t i = 0; i < finalPaths_.size(); ++i) {
      for (size_t j = 0; j < finalPaths_[i].size(); ++j) {
        copySize[seqId] = getBeamSize() > 1 ? starts[i + 1] - starts[i]
                                            : starts[j + 1] - starts[j];
        batchMachineStartPos_[seqId + 1] =
            batchMachineStartPos_[seqId] + finalPaths_[i][j].ids.size();
        seqId++;
      }
    }
  } else {
    for (size_t i = 0; i < finalPaths_[0].size(); ++i)
      batchMachineStartPos_[i + 1] =
          batchMachineStartPos_[i] + finalPaths_[0][i].ids.size();
  }
}

void RecurrentGradientMachine::createDataOutlink() {
  for (size_t i = 0; i < dataArgsSize_; i++) {
    bool isSeq = dataArgsFrame_[i][0].hasSeq();
    std::vector<int> copySize;
    createDataOutlinkCopySizeInfo(isSeq, dataArgsFrame_[i], copySize);
    createDataOutlinkSelRowsInfo(isSeq, dataArgsFrame_[i]);

    dataArgs_[i].concat(dataArgsFrame_[i],
                        batchMachineIdVec_,
                        batchMachineStartPos_,
                        copySize,
                        useGpu_,
                        HPPL_STREAM_1,
                        PASS_TEST);
    auto dataAgent =
        dynamic_cast<DataLayer*>(outFrameLines_[i + 1].agentLayer.get());
    CHECK_NOTNULL(dataAgent);
    dataAgent->setData(dataArgs_[i]);
  }
}

void RecurrentGradientMachine::beamSearch(size_t batchSize) {
  finalPaths_.clear();
  finalPaths_.resize(batchSize);
  seqIds_.resize(batchSize);
  minFinalPathLogProb_.clear();
  minFinalPathLogProb_.resize(batchSize, 0);

  std::vector<Path> paths;
  std::vector<Path> newPaths;
  for (size_t i = 0; i < batchSize; ++i) {
    paths.push_back(Path(i));
    if (this->beamSearchCtrlCallbacks_) {
      paths.back().recordHistory();
    }
  }

  // restart beam search
  stopBeamSearch_ = false;
  for (int i = 0; i < maxSequenceLength_; ++i) {
    int machineCur = i % 2;
    std::unique_ptr<
        ScopedCallbacks<const RecurrentGradientMachine::EachStepCallback&, int>>
        statisticsBlock;
    if (this->beamSearchStatistics_) {
      auto ptr =
          new ScopedCallbacks<const RecurrentGradientMachine::EachStepCallback&,
                              int>(beamSearchStatistics_->onEachStepStarted,
                                   beamSearchStatistics_->onEachStepStoped,
                                   i);
      statisticsBlock.reset(ptr);
    }
    if (stopBeamSearch_) break;

    if (i) connectPrevFrame(i, paths);

    if (this->beamSearchCtrlCallbacks_) {
      std::vector<std::vector<int>*> prefixes;
      prefixes.resize(paths.size());
      std::transform(
          paths.begin(), paths.end(), prefixes.begin(), [](const Path& p) {
            return const_cast<std::vector<int>*>(&p.ids);
          });
      beamSearchCtrlCallbacks_->beamSearchCandidateAdjust(
          prefixes, frames_[machineCur].get(), i);
    }

    forwardFrame(machineCur);
    beamExpand(paths, newPaths);
    if (newPaths.empty()) break;

    paths = newPaths;
    newPaths.clear();
  }  // end for machineCur
  fillGenOutputs();
}

void RecurrentGradientMachine::Path::adjustProb(int calc_id, bool atEos) {
  if (gDiyProbMethod) {
    logProb = gDiyProbMethod(calc_id, ids.size(), ids.data(), logProb, atEos);
  }
}

}  // namespace paddle
