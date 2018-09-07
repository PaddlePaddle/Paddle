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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/**
 * AgentLayer use as a virtual input of another layer in config,
 * before execute forward/backward, setRealLayer() should be
 * called to set one and only one real layer
 */
class AgentLayer : public Layer {
 protected:
  LayerPtr realLayer_;
  int numSamples_;

 public:
  explicit AgentLayer(const LayerConfig& config) : Layer(config) {}

  ~AgentLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  // if *numSamples* set,
  // real layer output will only use first *numSamples* rows
  void setRealLayer(LayerPtr layer, int numSamples = 0) {
    realLayer_ = layer;
    numSamples_ = numSamples;
  }

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override {}
};

/**
 * Like AgentLayer, but it can gather many real layers. Each real
 * layer give a few rows of a sequence, after gather all real layers,
 * GatherAgentLayer collect a complete sequence.
 */
class GatherAgentLayer : public Layer {
 protected:
  std::vector<LayerPtr> realLayers_;
  std::vector<IVectorPtr> idsVec_;
  // we don't clear idsVec_ vector to aviod IVector alloc/free
  IVectorPtr allIds_;
  std::vector<int> idIndex_;

 public:
  explicit GatherAgentLayer(const LayerConfig& config) : Layer(config) {}

  virtual ~GatherAgentLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  // call before addRealLayer
  void clearRealLayers() { realLayers_.clear(); }

  void copyIdAndSequenceInfo(ICpuGpuVectorPtr sequenceStartPositions,
                             ICpuGpuVectorPtr subSequenceStartPositions,
                             const IVectorPtr& allIds,
                             const std::vector<int>& idIndex);

  // add one real layer, can call many times
  void addRealLayer(LayerPtr layer) { realLayers_.push_back(layer); }

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;
  void forwardValue(PassType passType);
  void forwardIds(PassType passType);
};

/**
 * Like AgentLayer, but only select a few rows in real layer.
 * [idIndex, idIndex + idSize) of *ids* in setRealLayerAndOutput()
 * are the selected row ids. It's used to scatter one layer's output
 * to many small submodels. ScatterAgentLayer can support ids real layer,
 * if it is, the agent will select a few ids in real layer.
 */
class ScatterAgentLayer : public Layer {
 protected:
  LayerPtr realLayer_;
  IVectorPtr ids_;
  IVectorPtr cpuIds_;
  Argument realOutArg_;
  int idIndex_;
  int idSize_;
  int seqStartPosIndex_;
  int numSequences_;  // number of sequences in this scatterAgentLayer
  bool handleBackward_;

  // use to store expanded cpuStartPositions or subSequenceStartPositions
  // of real layer.
  ICpuGpuVectorPtr inputStartPos_;

  // true for setRealLayer, false for setRealLayerAndOutput
  bool selectionMode_;

 public:
  explicit ScatterAgentLayer(const LayerConfig& config) : Layer(config) {}

  virtual ~ScatterAgentLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  /**
   * @brief set real layer in generation
   *
   * @param layer[input]    realLayer
   * @param ids[input]      row id in real layer
   * @param copyId[input]   whether to copy a cpu version of ids,
   *                        false(default) in ScatterAgentLayer, and
   *                        true in SequenceScatterAgentLayer.
   */
  void setRealLayer(LayerPtr layer, const std::vector<int>& ids) {
    realLayer_ = layer;
    IVector::resizeOrCreate(ids_, ids.size(), useGpu_);
    ids_->copyFrom(ids.data(), ids.size());
    if (useGpu_) {
      IVector::resizeOrCreate(cpuIds_, ids.size(), false);
      cpuIds_->copyFrom(ids.data(), ids.size());
    } else {
      cpuIds_ = ids_;
    }
    selectionMode_ = true;
  }

  // set real layer and output, [idIndex, idIndex + idSize) of *ids*
  // are selected row for realOutArg in realLayer
  void setRealLayerAndOutput(LayerPtr layer,
                             const Argument& outArg,
                             const IVectorPtr& ids,
                             int idIndex,
                             int idSize,
                             bool handleBackward) {
    realLayer_ = layer;
    realOutArg_ = outArg;
    ids_ = ids;
    idIndex_ = idIndex;
    idSize_ = idSize;
    handleBackward_ = handleBackward;
    selectionMode_ = false;
  }

  void setSequenceStartPositions(const ICpuGpuVectorPtr& sequenceStartPositions,
                                 int seqStartPosIndex,
                                 int numSequences) {
    realOutArg_.sequenceStartPositions = sequenceStartPositions;
    seqStartPosIndex_ = seqStartPosIndex;
    numSequences_ = numSequences;
  }

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

  void forwardWithSelection(PassType passType);
};

}  // namespace paddle
