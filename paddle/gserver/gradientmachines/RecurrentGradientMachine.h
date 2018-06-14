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
#include "GradientMachine.h"
#include "NeuralNetwork.h"

#include "paddle/utils/Locks.h"

namespace paddle {

/**
 * Private data class declares.
 * Used for user customized beam search.
 */
class BeamSearchControlCallbacks;
class BeamSearchStatisticsCallbacks;

class RecurrentGradientMachine : public NeuralNetwork {
 public:
  RecurrentGradientMachine(const std::string& subModelName,
                           NeuralNetwork* rootNetwork);

  // Disable copy and assign.
  RecurrentGradientMachine(const RecurrentGradientMachine& other) = delete;
  RecurrentGradientMachine& operator=(const RecurrentGradientMachine& other) =
      delete;

  virtual ~RecurrentGradientMachine() {
    this->removeBeamSearchStatisticsCallbacks();
    this->removeBeamSearchControlCallbacks();
  }

  virtual void init(const ModelConfig& config,
                    ParamInitCallback callback,
                    const std::vector<ParameterType>& parameterTypes,
                    bool useGpu);

  virtual void prefetch(const std::vector<Argument>& inArgs);

  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  void forwardBackward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType,
                       const UpdateCallback& callback);

  virtual void resetState() {}
  virtual void eval(Evaluator* evaluator) const;

  const std::vector<int>& getParameterIds() { return parameterIds_; }

  /**
   * @brief BeamSearchCandidatesAdjustCallback
   *
   * Adjust searching candidates to restrict beam search
   * searching within a limited subset of all possibile paths.
   *
   * The first parameter is the prefixes of all formed paths in current
   * beam search step, whose type is basically int[][].
   *
   * The second parameter is a pointer to the network used to generate sequence,
   * user can use this pointer to tranverse each layer in the network to
   * modify behaivors of a particular layer.
   *
   * The third parameter is an integer to indicate the iteration number of
   * beam search, so that user can customize different operations in different
   * beam search iterations.
   */
  typedef std::function<void(
      const std::vector<std::vector<int>*>&, NeuralNetwork*, const int)>
      BeamSearchCandidatesAdjustCallback;

  /**
   * @brief DropCallback
   *
   * Drop a whole prefix or one candidate in beam search or not.
   *
   * The first parameter is sequence index in a batch
   *
   * The second parameter is one path in beam search,
   * which is made up of node indices.
   *
   * The third parameter is probabilites for each node in this path.
   *
   * Return true if this prefix or candidate is expected to be dropped.
   */
  typedef std::function<bool(
      int seqId, const std::vector<int>&, const std::vector<real>&)>
      DropCallback;

  /**
   * @brief NormOrDropNodeCallback
   *
   * Normalize a path's probabilities or just drop it by modifying path.logProb
   *
   * The first parameter is sequence index in a batch
   *
   * The second parameter is path.ids
   *
   * The third parameter is probabilites for each node in this path.
   *
   * The fourth parameter is the probability of the whole path.
   */
  typedef std::function<void(
      int seqId, const std::vector<int>&, std::vector<real>&, real*)>
      NormOrDropNodeCallback;

  /**
   * @brief Register beam search control callbacks. Used for prediction.
   *
   * @param queryBeamSearch: Give the sequences already formed, return the
   * nodes expected to be expanded.
   * Input: A pointer to an array holding pathes which have been expanded
   * Return: A pointer to an array holding nodes wanted to be expanded.
   *
   * @param dropOneNode: Early drop a node in one beam search step.
   * Given the path formed and probability history, decide whether a node
   * should be dropped or not.
   *
   * @param stopBeamSearch: Early stop a path in one beam search step.
   * Given the path and probability history, decide whether a path
   * should be dropped or not.
   */
  void registerBeamSearchControlCallbacks(
      const BeamSearchCandidatesAdjustCallback& adjustBeamSearch,
      const NormOrDropNodeCallback& normOrDropNode,
      const DropCallback& stopBeamSearch);

  /**
   * @brief Remove user costumized beam search callbacks,
   *
   * make sequence generation acts like normal beam search.
   */
  void removeBeamSearchControlCallbacks();

  /**
   * @brief EachStepCallback
   *
   * Invoke with beam search step.
   */
  typedef std::function<void(int)> EachStepCallback;

  /**
   * @brief register statistics methods for performance profile of beam search.
   *
   * @param onEachStepStarted: invoke once a beam search step starts.
   * Its input is index of the beam search step.
   *
   * @param onEachStepStoped: invoke once a beam search step ends.
   * Its input is index of the beam search step.
   */
  void registerBeamSearchStatisticsCallbacks(
      const EachStepCallback& onEachStepStarted,
      const EachStepCallback& onEachStepStoped);

  /**
   * @brief Remove beam search callbacks.
   */
  void removeBeamSearchStatisticsCallbacks();

  /**
   * @brief Stop beam search for current source.
   *
   * Will restart beam search in the next forward
   */
  void stopBeamSearch();

  struct Path {
    /**
     * @brief ids, path of beam search.
     */
    std::vector<int> ids;

    /**
     * @brief idsProb, log probability of each generated word.
     */
    std::vector<real> idsProb;

    /**
     * @brief logProb, current probability of path.
     */
    real logProb;

    int machineId;  // index of sample in frame
    int topIndex;   // index of MaxIdLayer output in one sample
    int seqId;      // index of sequence in batch generation
    std::vector<int> machineIdVec;

    /**
     * @brief A record of each node's probality in a formed path in beam search.
     *
     * @note  It could be empty when history is not recorded. If the history is
     *        wanted to be recorded, recordHistory() MUST be invoked first.
     */
    std::vector<real> probHistory;

    /**
     * @brief Path default ctor, first logProb is 0.
     */
    Path() {
      logProb = 0;
      seqId = 0;
    }
    explicit Path(size_t seqId) : seqId(seqId) { logProb = 0; }

    /**
     * @brief Create a new path based on an old path and
     * a new node with probability.
     *
     * @param old       old path
     * @param newId     index of the new node
     * @param logProb   probability of the new node.
     * @param machineId sample index of a frame in RNN
     * @param topIndex  index of MaxIdLayer output in one sample
     */
    Path(Path& old, int newId, real logProb, int machineId, int topIndex)
        : ids(old.ids),
          idsProb(old.idsProb),
          logProb(old.logProb + logProb),
          machineId(machineId),
          topIndex(topIndex),
          seqId(old.seqId) {
      ids.push_back(newId);
      idsProb.push_back(logProb);
      if (!old.probHistory.empty()) {
        this->probHistory = old.probHistory;
        // probHistory store current prob, not sum
        this->probHistory.push_back(logProb);
      }
    }

    /**
     * @brief operator <
     *
     * Path a < Path b means log probability of a is smaller than that of b
     */
    bool operator<(const Path& other) const {
      return (logProb < other.logProb);
    }

    static bool greaterPath(const Path& a, const Path& b) { return (b < a); }

    /**
     * @brief Start recording history in this path.
     */
    void recordHistory() { this->probHistory.push_back(this->logProb); }

    /**
     * @brief Adjust probability for DIY beam search interface.
     * In normal situation, it will do nothing.
     *
     * @param calc_id: the object id for DIY beam search interface.
     * @param atEos: at end of sequence or not.
     */
    void adjustProb(int calc_id, bool atEos = false);

    /**
     * @brief isDropable indacating whether the current node will be
     * dropped or not in beam search.
     *
     * @note: if logProb is -inf, current node will be dropped.
     * @return true to drop the current node.
     */
    bool isDropable() const { return std::isinf(logProb) && logProb < 0; }
  };

  /**
   * @brief access beam search results.
   * @return beam search results.
   */
  const std::vector<std::vector<Path>>& getFinalPaths() const {
    return this->finalPaths_;
  }

 protected:
  std::vector<Argument::SeqInfo> commonSeqInfo_;
  ICpuGpuVectorPtr sequenceStartPositions_;
  void calcSequenceStartPositions();
  void checkInputConsistency(int inlinkId,
                             const std::vector<Argument::SeqInfo>& seqInfo);
  void reorganizeInput(PassType passType);
  void reorganizeOutput(PassType passType);
  void connectFrames(PassType passType);
  void calcNumSequencesAtEachStep();

  void resizeOrCreateFrames(int numFrames);
  void resizeBootFrame(int numSequences);

  void generateSequence();
  void oneWaySearch(size_t batchSize);
  void beamSearch(size_t batchSize);

  struct InFrameLine {
    std::string linkName;
    LayerPtr inLayer;
    std::vector<LayerPtr> agents;  // Scatter Agents to reform batch input
    Argument outArg;               // scatter output argument
  };
  std::vector<InFrameLine> inFrameLines_;

  struct OutFrameLine {
    std::string layerName;
    LayerPtr agentLayer;
    std::vector<LayerPtr> frames;
  };
  std::vector<OutFrameLine> outFrameLines_;

  struct MemoryFrameLine {
    std::string layerName;
    std::string linkName;
    LayerPtr bootLayer;  // actually used biasLayer or rootAgent
    LayerPtr biasLayer;
    LayerPtr rootLayer;  // layer in root network to boot this memory
    LayerPtr rootAgent;  // agent to link rootLayer
    std::vector<LayerPtr> frames;
    std::vector<LayerPtr> agents;
    std::vector<LayerPtr> scatterAgents;  // scatter agent used by beam search
    Argument outArg;                      // scatter output argument
    // Different memoryFrameLine have different element as follows
    IVectorPtr allIds;  // scattered id of realLayer
    ICpuGpuVectorPtr
        sequenceStartPositions;  // scattered sequenceStartPositions
  };
  std::vector<MemoryFrameLine> memoryFrameLines_;

  // Each inFrameLines(inlinks) has its own info(elements) below,
  // and all outFrameLines(outlinks) share the info with one inFrameLine,
  // which is assigned by targetInfoInlinkId_.
  struct Info {
    // The original positions in the original batch
    IVectorPtr allIds;  // scattered id of realLayer [batchSize]

    // index of allIds for each step [maxSequenceLength_]
    // idIndex[i] is the total length of the first i sequences
    std::vector<int> idIndex;

    ICpuGpuVectorPtr
        sequenceStartPositions;         // scattered sequenceStartPositions
    std::vector<int> seqStartPosIndex;  // index of sequenceStartPositions
  };
  std::vector<Info> info_;  // for input

  // numSeqs_[i] is the number sequences which is longer than i (for sequence
  // data) or has more than i subsequences (for subsequence data)
  // Equivalently, numSeqs_[i] is the number of sequences at step i;
  std::vector<int> numSeqs_;

  std::vector<std::vector<Argument::SeqInfo>> seqInfos_;

  void checkOutputConsistency(OutFrameLine& outFrameLine);

  /* create scattered id infomation for all realLayer of inFrameLines one time.
   *  If hasSubseq, will also create scattered sequenceStartPositions infomation
   *  for all realLayer of inFrameLines one time.
   */
  void createInFrameInfo(int inlinks_id,
                         const Argument& input,
                         PassType passType);
  void createInFrameInfo_nonseq(int inlinks_id,
                                const Argument& input,
                                PassType passType);
  void createInFrameInfo_seq(int inlinks_id,
                             const Argument& input,
                             PassType passType);
  void createInFrameInfo_subseq(int inlinks_id,
                                const Argument& input,
                                PassType passType);

  void createOutFrameInfo(OutFrameLine& outFrameLine,
                          Info& info,
                          ICpuGpuVectorPtr& sequenceStartPositions,
                          ICpuGpuVectorPtr& subSequenceStartPositions);
  void createOutFrameInfo_seq(OutFrameLine& outFrameLine,
                              Info& info,
                              ICpuGpuVectorPtr& sequenceStartPositions,
                              ICpuGpuVectorPtr& subSequenceStartPositions);
  void createOutFrameInfo_subseq(OutFrameLine& outFrameLine,
                                 Info& info,
                                 ICpuGpuVectorPtr& sequenceStartPositions,
                                 ICpuGpuVectorPtr& subSequenceStartPositions);

  void createMemoryFrameInfo(MemoryFrameLine* memoryFrameLine,
                             PassType passType);

  void copyScattedId(std::vector<int>& srcIds, IVectorPtr* dstIds, int size);

  void selectRowsOneTime(LayerPtr layer,
                         const IVectorPtr& allIds,
                         Argument* arg,
                         PassType passType);

  void createSeqPos(const std::vector<int>& sequenceStartPosition,
                    ICpuGpuVectorPtr* sequenceStartPositions);

  // for generator
  struct EosFrameLine {
    std::vector<LayerPtr> layers;
  };
  std::unique_ptr<EosFrameLine> eosFrameLine_;

  struct Generator {
    GeneratorConfig config;
    std::vector<int> ids;       // store generated sequences
    std::vector<real> idsProb;  // log probability of each generated word
    Argument outArg;            // final output argument
  };
  bool generating_;
  Generator generator_;

  std::vector<std::unique_ptr<NeuralNetwork>> frames_;

  NeuralNetwork* rootNetwork_;
  bool reversed_;

  int maxSequenceLength_;  // Max top-level length
  bool useGpu_;
  bool stopBeamSearch_;

  std::vector<int>
      parameterIds_;  // parameters actually used by this Layer Group

  // store final argument of outFrameLines_
  std::vector<Argument> dataArgs_;
  // store each frame's output argument of outFrameLines_
  std::vector<std::vector<Argument>> dataArgsFrame_;
  size_t dataArgsSize_;  // size of dataArgs_ = size of dataArgsFrame_

  IVectorPtr cpuId_;
  MatrixPtr cpuProb_;
  IVectorPtr cpuEos_;

 private:
  /*
   * @return beam size in beam search
   */
  size_t getBeamSize() { return generator_.config.beam_size(); }

  /*
   * @return number of sequence in a batch in generation
   */
  size_t getGenBatchSize();

  /*
   * @brief store output of the machineCur-th frame during generation, for
   * creating the final outlink after the entire generation process is finished.
   *
   * In generation, if the layer group has more than 1 outlink, the first
   * one is reserved to store the generated word indices, the others are data
   * outlinks, that can be used like a common layer in the network.
   *
   * @param machineCur : index to access the layer group frame in
   * currrent generation step.
   */
  void copyDataOutlinkFrame(size_t machineCur);

  /*
   * @brief In generation, if the layer group has more than 1 outlink, outlink
   * except the first one is a data outlink. In RecurrentLayerGroup, each time
   * step is a separate Network, outputs of a layer inside the
   * RecurrentLayerGroup are stored in separate Arguments. If one layer is
   * specified as an outlink of RecurrentLayerGroup. This function will
   * collect outputs in each time step of each generated sequence which are
   * dispersed in separate Arguments to form a new single Argument as output of
   * RecurrentLayerGroup.
   */
  void createDataOutlink();

  /*
   * @brief decide to select how many rows from the Matrix stored the forward
   * pass results from a start position.
   *
   * @param isSeq: a flag indicating whetehr the layer to be output of the
   * RecurrentGradientMachine is a sequence or not
   * @param outArgs: all of the the returned Arguments of the forward pass
   * during the generation process.
   * @param copySize: the returned result, number of rows to select from the
   * Matrix stored the forward pass results from a start position.
   */
  void createDataOutlinkCopySizeInfo(bool isSeq,
                                     std::vector<Argument>& outArgs,
                                     std::vector<int>& copySize);

  /*
   * @brief decide index of the start row for each time step of a generated
   * sequence in Matrix stored the entire beam search batch's forward pass
   * results.
   *
   * @param isSeq: a flag indicating whether the layer to be output of the
   * RecurrentGradientMachine is a sequence or not
   * @param outArgs: all of the returned Arguments of the forward pass
   * during the generation process.
   */
  void createDataOutlinkSelRowsInfo(bool isSeq, std::vector<Argument>& outArgs);

  /*
   * @brief used in beam search, connect previous frame to form recurrent link
   * @param stepId : iteration number of generation process.
   * It equals to the length of longest half-generated sequence.
   * @param paths : half-generated paths that are going to be expanded
   * in current beam search iteration.
   */
  void connectPrevFrame(int stepId, std::vector<Path>& paths);

  /*
   * @brief used in beam search, forward current recurrent frame
   * @param machineCur : index to access the layer group frame in
   * currrent generation step.
   */
  void forwardFrame(int machineCur);

  /*
   * @brief reduce all expanded paths to beam size.
   *
   * @param newPaths : newPaths[totalExpandCount : ] stores all expanded paths
   * for the seqId-th sequence
   * @param seqId : sequence index in a batch
   * @param totalExpandCount : number of already shrinked paths in newPaths
   * @return size of retained paths at the end of a beam search iteration
   */
  size_t beamShrink(std::vector<Path>& newPaths,
                    size_t seqId,
                    size_t totalExpandCount);

  /*
   * @brief expand a single path to expandWidth new paths
   * with highest probability
   * @param curPath : path to be expanded
   * @param curPathId : index of curPath in member newPaths
   * @param expandWidth : number of paths to be expanded
   */
  void singlePathExpand(Path& curPath,
                        size_t curPathId,
                        std::vector<Path>& newPaths,
                        size_t expandWidth);

  /*
   * @brief A new beam search iteration. Each half-generated paths in previous
   * beam search iteration are further expanded to beam_size new paths
   * with highest probabilities, and then all the expanded paths are again
   * reduced to beam_size paths according to their log probabilities.
   * @param paths : half-generated paths in previous iteration.
   * @param newPaths : paths expanded and then reduces in current iteration.
   */
  void beamExpand(std::vector<Path>& paths, std::vector<Path>& newPaths);

  /*
   * @brief fill sequence start positions and some other information that are
   * uesed by the "text_printer" evaluator.
   */
  void fillGenOutputs();

  std::vector<int> machineIds_;
  std::vector<int> topIds_;
  std::vector<int> seqIds_;
  std::vector<int> batchMachineIdVec_;
  std::vector<int> batchMachineStartPos_;
  std::vector<std::vector<Path>> finalPaths_;
  std::vector<real> minFinalPathLogProb_;
  BeamSearchControlCallbacks* beamSearchCtrlCallbacks_;
  BeamSearchStatisticsCallbacks* beamSearchStatistics_;
};
}  // namespace paddle
