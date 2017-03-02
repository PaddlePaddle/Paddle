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
 * like AgentLayer, but use first *numSamples* sequences
 */
class SequenceAgentLayer : public AgentLayer {
public:
  explicit SequenceAgentLayer(const LayerConfig& config) : AgentLayer(config) {}
  ~SequenceAgentLayer() {}

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
  void copyIdAndSequenceInfo(const Argument& input,
                             const IVectorPtr& allIds,
                             const std::vector<int>& idIndex);

  // add one real layer, can call many times
  void addRealLayer(LayerPtr layer) { realLayers_.push_back(layer); }

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;
};

/**
 * Like GatherAgentLayer, but select a few sequence in real layer.
 * *ids* in addRealLayer() are the ids of selected sequence.
 * It's used to reorder sequence output.
 */
class SequenceGatherAgentLayer : public GatherAgentLayer {
public:
  explicit SequenceGatherAgentLayer(const LayerConfig& config)
      : GatherAgentLayer(config) {}
  virtual ~SequenceGatherAgentLayer() {}

  void forward(PassType passType);
  void backward(const UpdateCallback& callback) {
    // same as GatherAgentLayer
    GatherAgentLayer::backward(callback);
  }
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
  void setRealLayer(LayerPtr layer,
                    const std::vector<int>& ids,
                    bool copyId = false) {
    realLayer_ = layer;
    IVector::resizeOrCreate(ids_, ids.size(), useGpu_);
    ids_->copyFrom(ids.data(), ids.size());
    if (copyId) {
      if (useGpu_) {
        IVector::resizeOrCreate(cpuIds_, ids.size(), false);
        cpuIds_->copyFrom(ids.data(), ids.size());
      } else {
        cpuIds_ = ids_;
      }
    }
  }

  // set real layer and output, [idIndex, idIndex + idSize) of *ids*
  // are selected row for realOutArg in realLayer
  void setRealLayerAndOutput(LayerPtr layer,
                             const Argument& outArg,
                             const IVectorPtr& ids,
                             int idIndex,
                             int idSize) {
    realLayer_ = layer;
    realOutArg_ = outArg;
    ids_ = ids;
    idIndex_ = idIndex;
    idSize_ = idSize;
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
};

/**
 * Like ScatterAgentLayer, but select a few sequence in real layer.
 * *ids* in setRealLayer() or setRealLayerAndOutput() are the ids of
 * selected sequence. It's used to reorder sequence input.
 */
class SequenceScatterAgentLayer : public ScatterAgentLayer {
protected:
  // use to store expanded cpuStartPositions or subSequenceStartPositions
  // of real layer.
  ICpuGpuVectorPtr inputStartPos_;

public:
  explicit SequenceScatterAgentLayer(const LayerConfig& config)
      : ScatterAgentLayer(config) {}
  virtual ~SequenceScatterAgentLayer() {}

  void forward(PassType passType);
  void backward(const UpdateCallback& callback) {
    ScatterAgentLayer::backward(callback);
  }
};

}  // namespace paddle
