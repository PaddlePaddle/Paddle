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

#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <stddef.h>
#include <stdlib.h>

#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/ParameterOptimizer.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/ThreadLocal.h"

#include "ParameterService.pb.h"

#include "ProtoServer.h"

DECLARE_int32(port);

namespace paddle {

// @TODO(yanfei):
// if armed with high density computation resource per node, pserver could also
// utilize GPU to reduce overhead. if this mechanism is used, it could pipeline
// network receiving and GPU computation to reduce the network overhead even
// further. the pipeline could help to accelerate BIG model training.
// @TODO:(yanfei)
// for cpu and less/low gpu machine, the time exhausted by forward and backward
// could be larger than optimization at pserver. However, if armed with lots of
// gpus per node and if the model size is so large enough that limited cpu
// computation causes big optmization latency, the GPU may be required by
// pserver.

/**
 * Client interface for the parameter server
 *
 * it implements several rpc API for remote parameter client usage.
 * for sync-sgd, client needs one controller thread to build connections
 * to all pservers, these controller connections do barriers
 * synchronization with these connections used for transfering data.
 * each data connection uses block based fine grained synchronization
 * to gain better scalability. Merging gradients from different trainers
 * are concurrently executed with block units, so that some network
 * overhead will be hidden in merging gradient.
 * for async-sgd, the difference is that pserver will do optimization
 * immediately if the gradients are ready, so that pserver needs to
 * prepare separate buffer to store value for sending back to trainer
 * to prevent from being polluted.
 */
class ParameterServer2 : public ProtoServer {
protected:
  /// parameter_ mutex.
  RWLock parameterMutex_;

  typedef std::pair<size_t, int64_t> BlockKey;
  struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
      return std::hash<size_t>()(key.first) + key.second;
    }
  };

  // TODO(yanfei):
  // if index data structure is based on parameters instead of blocks, the
  // lookup performance could be better. In addition, the block memory
  // access almost exhibits good locality, so index data structure and
  // block data structure can be refined further, especially if gpu is used
  // for pserver.
  /**
   * all parameters are stored in CpuVector with a blockMap_ data structure
   * to index block data required by requests.
   */
  typedef std::unordered_map<BlockKey, int64_t, BlockKeyHash> BlockMap;
  /// <(para, block), global offset(byte) in all parameters>
  BlockMap blockOffsetMap_;
  /// <(para, block), global idx [0, nBlocksInAllParameters]>
  BlockMap blockIdMap_;

  std::vector<CpuVectorPtr> vectors_;
  std::vector<CpuMatrixPtr> matrices_;
  std::vector<CpuMemHandlePtr> dataMems_;

  // TODO(yanfei):
  // if storing sparse_remote_update() flag in request instead of
  // reading configMap_, and storing config within new block wise
  // overview data structure, the config mapping, block mapping
  // can be unified in single clean data structure. Use para_id
  // to index parameters, use offset to index block within parameter
  // and keep two index into single one.
  /**
   * mapping between parameter and config
   * different parameter allows different config, such as decay_rate.
   * for each request, it need to read config for adding gradient
   * and optmization.
   */
  std::unordered_map<size_t, ParameterConfig> configMap_;

  /**
   * to parallelize the multi-thread and multi-connnection
   * computation at pserver, it use block unit to reduce
   * the contention for computation, even further use block
   * level optimizater control for each block for some special
   * reason annotated below.
   */
  struct BlockInfo {
    const ParameterConfig* config;
    std::unique_ptr<std::mutex> lock;
    /// global offset for all parameters
    uint64_t offset;
    /**
     *
     * Async sgd in pserver is very different from sync sgd.
     * Each trainer follows startBatch, update*, finishBatch as in
     * sync sgd, but all these actions are almost executed by
     * multi-core and multi-thread simutaneously, so that async
     * sgd optimization is based on block level in reality, then
     * per block optimization is necessary indeed. In addition,
     * per block optimization is also perfered for performance
     * with multithreads.
     */
    std::unique_ptr<ParameterOptimizer> optimizer;
  };
  std::vector<BlockInfo> blockInfos_;

  typedef std::vector<std::pair<int64_t, int64_t>> BlockSegments;
  /// Because some blocks might not be fully used. We keep a
  /// record of which segments are used.
  BlockSegments usedSegments_;

  /// record pserver status, all status defined in ParameterService.pb
  PServerStatus status_;
  /// record all samples processed which could be used by optimizater
  std::atomic<int64_t> numSamplesProcessed_;
  double cost_;
  int mpiSize_;
  int dataSize_;
  /// configuration for current parameter optimizer
  OptimizationConfig config_;

  /**
   * The ReadWriteBuffer is based on std::vector, but aligned for avx/sse
   * compute. And add some helper method to allocate memory aligned blocks.
   *
   * @param T          type of element.
   * @param AlignBytes the memory aligned bytes for allocated blocks.
   */
  template <typename T, size_t AlignBytes>
  class ReadWriteBuffer
      : public std::vector<T, AlignedAllocator<T, AlignBytes>> {
  public:
    static_assert(sizeof(T) % AlignBytes == 0 || AlignBytes % sizeof(T) == 0,
                  "Type T must be able to aligned.");

    /**
     * @brief IsTLargerThanAlign compiled time calculated constant for is type
     * T larger than alignments.
     */
    constexpr static bool IsTLargerThanAlign = sizeof(T) >= AlignBytes;

    static_assert(std::is_pod<T>::value, "T must be POD type.");

    /**
     * @brief if AlignBytes > sizeof(T), then will calcuate how many elements
     * can be stored in AlignBytes.
     */
    constexpr static size_t AlignElementCount = AlignBytes / sizeof(T);

    static_assert(AlignElementCount ==
                          (AlignElementCount & -AlignElementCount) ||
                      AlignBytes > sizeof(T),
                  "AlignElementCount should be exp of 2");

    /**
     * @brief Resize Buffer, with block count that will be allocated. Each block
     * will be memory aligned in AlignBytes.
     * @param size The element count in all blocks.
     * @param alignBlockCount The block count that will be allocated.
     */
    void resizeWithAlignHints(size_t size, size_t alignBlockCount = 1) {
      if (IsTLargerThanAlign) {  //! So, each elements is memory aligned.
        this->resize(size);
      } else {
        //! at most, we need such elements in buffer to make sure each block is
        //! aligned.
        this->resize(size + alignBlockCount * (AlignElementCount - 1));
      }
    }

    /**
     * @brief reset aligned allocate blocks.
     */
    void resetAlignAlloc() { this->curOffset_ = 0; }

    /**
     * @brief get next aligned block address.
     * @param blockSize is the element count in each block.
     * @return Aligned block address.
     */
    T* nextBlock(size_t blockSize) {
      T* r = &this->operator[](curOffset_);
      curOffset_ += blockSize;

      if (!IsTLargerThanAlign) {
        curOffset_ =
            (curOffset_ + AlignElementCount - 1) & ~(AlignElementCount - 1);
      }
      return r;
    }

  private:
    size_t curOffset_;
  };

  /// to buffer the data from network for further processing to
  /// reduce redundant memory allocation.
  ThreadLocal<ReadWriteBuffer<real, ALIGN_HINT>> readWriteBuffer_;

  /// size of the parameter
  int64_t size_;

  /// for synchronized training, check details in addGradient()
  /// and doOperation()
  ThreadBarrier gradientReadyBarrier_;
  ThreadBarrier parameterReadyBarrier_;
  ThreadBarrier passBarrier_;
  ThreadLocal<std::vector<SendParameterRequest>> requestVec_;
  ThreadLocal<std::vector<ProtoResponseCallbackEx>> callbackVec_;

  std::atomic<int> numPassFinishClients_;
  bool allClientPassFinish_;

  std::vector<std::unique_ptr<ThreadBarrier>> synchronizeBarriers_;
  std::atomic<int> serverId_;

  /**
   *
   * for lagged async gradient gradient commit control in Async Sgd.
   * discard lagged gradients from too slow nodes, whose gradients
   * exhibits bad quality.
   * Algorithm:
   * pserver:
   * 1. initial asyncUpdaterSteps = 0, asyncTrainerSteps_[N] = 0.
   * syncUpdaterSteps means
   *    the version of parameter value.
   * 2. when pull arrives, record asyncUpdateSteps_ into
   * syncTrainerSteps_[trainer_id]
   * 3. when push arrives, compare asyncUpdateSteps_ with
   * syncTrainerSteps_[trainer_id]
   *    if delta > threshold, discard current gradient, else commit
   *    gradient.
   * 4. reset asyncUpdaterSteps_ and asyncTrainerSteps_[N] when pass
   * finished
   * Note:
   * it can not discard all lag-gradient strictly in some special
   * condition. part of gradients could be discarded if
   * ConcurrentRemoteParameterUpdater is sed.
   * this algorithm is implemented in asynSGD()
   */
  int64_t asyncLaggedThreshold_;
  std::atomic<int64_t> asyncUpdateSteps_;
  std::vector<int64_t> asyncTrainerSteps_;
  size_t asyncLaggedGradientsNum_;
  /// stat all async update
  std::vector<size_t> asyncUpdateStat_;
  /// stat per trainer_id
  std::vector<size_t> asyncTrainerDiscardStat_;
  /// stat per trainer_id
  std::vector<size_t> asyncTrainerCommitStat_;

  /// only used by controller and other control cmd from trainer number 0
  std::unique_ptr<SyncThreadPool> syncThreadPool_;

  /// pserver for sparse remote update parameters
  bool isSparseServer_;

  /// barrier performance tuning sync-sgd required
  std::atomic<int64_t> batchId_;

  /// the beginning of addGradient without network overhead
  ThreadLocal<struct timeval> addGradBegin_;

  /**
   * tuning barrier performance
   * to better control log for sparse and dense parameter,
   * we use different log entities for different parameterServer
   * objects.
   * it will output lots of performance stats to perceive the
   * overhead of network, fluctuation of computation from
   * forwardbackward and network, computation from optimization
   * at pserver end, barrier overhead, etc. to understand tuning
   * data, focus on the synchronization between addGradient and
   * doOperation which indirectly call op_SGD operation controlled
   * by remote updater controller
   */
  std::unique_ptr<StatSet> statSet_;

public:
  struct Buffer {
    real* base;
    size_t size;
  };

protected:
  /// async gradient commit control
  bool asyncGrdientCommitCheckAndStat(const SendParameterRequest& request);
  void printAsyncGradientCommitStatAndReset();

public:
  /// disable default parameter for overloading
  /// @rdmaCpu:the id of cpu core hosting RDMA server(0-N)
  /// -1 means using TCP transport instead of RDMA
  ParameterServer2(const std::string& addr, int port, int rdmaCpu = -1);

  ~ParameterServer2() {}

  static const std::string kRetMsgInvalidMatrixHandle;
  static const std::string kRetMsgInvalidVectorHandle;
  static const std::string kRetMsgUnknownOperation;

  /// service functions
  template <typename Dtype>
  void reduceAndSendData(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback);

  void templateReduceSum(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback);

  /**
   * @brief framework for sending parameters
   *
   * @note  different parameter data type can be sent to pserver.
   *        in most case, the api is used to send gradients from
   *        trainer to pserver.
   *        it also can be used to retrieve parameters from pserver
   */
  void sendParameter(const SendParameterRequest& request,
                     std::unique_ptr<MsgReader> msgReader,
                     ProtoResponseCallbackEx callback);

  void sendData(const SendDataRequest& request,
                std::unique_ptr<MsgReader> msgReader,
                ProtoResponseCallbackEx callback);

  /**
   * @brief send config to pserver
   *
   * @note  it can help pserver to understand the configuration for
   * optimization,
   *        logging control, duplicated initialization, etc.
   */
  void setConfig(const SetConfigRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief get status for pserver
   *
   * @note  used to check if parameters are ready at pserver
   */
  void getStatus(const GetStatusRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief set status for pserver
   *
   * @note  used to check if parameters are ready at pserver, since parameters
   *        at pserver are initialized by trainer
   */
  void setStatus(const SetStatusRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief framework for doing some operation at pserver end
   *
   * @note  if sync-sgd is used, controller will calling op_SGD action
   *        for gradient optimization.
   *        check avaiable operations in opFuncs[]
   */
  void doOperation(const DoOperationRequest& request,
                   ProtoResponseCallback callback);

  /// Create a column vector. The size is the dimension of parameter
  void createVector(const CreateVectorRequest& request,
                    ProtoResponseCallback callback);

  void releaseVector(const ReleaseVectorRequest& request,
                     ProtoResponseCallback callback);

  /// Create a column major matrix. The number of rows is the dimension of
  /// parameter. The number of columns is specifed by num_cols.
  void createMatrix(const CreateMatrixRequest& request,
                    ProtoResponseCallback callback);

  void releaseMatrix(const ReleaseMatrixRequest& request,
                     ProtoResponseCallback callback);
  /**
   * @brief stateful control for indicationg sync pass start
   *
   * @note  it is valuable for logging and state control,
   *        especially for sync-sgd control
   */
  void waitPassStart(const WaitPassStartRequest& request,
                     ProtoResponseCallback callback);

  /**
   * @brief stateful control for indicationg sync pass end
   *
   * @note  it is valuable for logging and state control,
   *        especially for sync-sgd control
   */
  void waitPassFinish(const WaitPassFinishRequest& request,
                      ProtoResponseCallback callback);

  /**
   * @brief synchronize all distributed trainers
   *
   * @note  it's general api for synchronizing trainer and pserver
   */
  void synchronize(const SynchronizeRequest& request,
                   ProtoResponseCallback callback);

  /**
   * @brief stateful control for indicating async pass is finished
   *
   * @note  it is valuable for logging control, state reset, etc.
   */
  void asyncFinishPass(const SynchronizeRequest& request,
                       ProtoResponseCallback callback);

  void loadValueVector(const LoadValueRequest& request,
                       ProtoResponseCallback callback);

  void saveValueVector(const SaveValueRequest& request,
                       ProtoResponseCallback callback);

public:
  /**
   * @brief initialize parameter server
   */
  bool init();

  /**
   * @brief set parameters at pserver
   *
   * @note  do parameter initialization if neccessy.
   */
  void setParameter(const SendParameterRequest& request,
                    std::vector<Buffer>& inputBuffers,
                    SendParameterResponse* response,
                    std::vector<Buffer>* outputBuffers);

  /**
   * @brief receive gradients and do optimization for async-sgd
   *
   * @note  this api asynchronizately receives all data from all
   *        trainers, and immediately do optimization and return
   *        optimizated value for trainer.
   *        this above routine are block based atomic updating,
   *        which means different block could based different stale
   *        gradient.
   *        it will discard some lagged gradients by default for
   *        better convergence.
   */
  void asyncSGD(const SendParameterRequest& request,
                std::vector<Buffer>& inputBuffers,
                SendParameterResponse* response,
                std::vector<Buffer>* outputBuffers);

  /**
   * @brief merge gradients from all trainer
   *
   * @note  this api use block based parallelization as fine grained
   *        parallelization which benifits lock contention and latency
   *        hidden for communication, also can harness multi-core
   *        efficiently.
   *        it also implements the synchronization for sync-sgd
   */
  void addGradient(const SendParameterRequest& request,
                   std::vector<Buffer>& inputBuffers,
                   SendParameterResponse* response,
                   std::vector<Buffer>* outputBuffers);

  /**
   * @brief get dense parameters from pserver
   *
   * @note  for some specified condition, trainer will get parameters from
   *        pservers.
   *        e.g.
   *        if all parameters are stored at perver end for big model training
   *        trainer can use it to retrieve all parameters if necessary.
   */
  void getParameter(const SendParameterRequest& request,
                    std::vector<Buffer>& inputBuffers,
                    SendParameterResponse* response,
                    std::vector<Buffer>* outputBuffers);

  /**
   * @brief get sparse value from parameter server
   *
   * @note  with sparse enabled, pservers own all latest value
   *        while trainer only retrieve value that only are needed.
   *        e.g.
   *        trainer will do prefetch action to retrieve necessary latest
   *        value from pserver for sparse calculation.
   */
  void getParameterSparse(const SendParameterRequest& request,
                          std::vector<Buffer>& inputBuffers,
                          SendParameterResponse* response,
                          std::vector<Buffer>* outputBuffers);

protected:
  void mergeSegments(BlockSegments* segments);

  /// set the unused segments to zero
  void clearUnusedSegments(CpuVector* vec);

  // TODO(yanfei):
  // if read data and do optimization interleavely block by block,
  // the performance could be better for gaining less network congestion.
  /// read all data from connection and store it in static pre-allocated buffer
  void readAllBlocks(MsgReader* msgReader,
                     std::vector<ParameterServer2::Buffer>* buffers);

  const ParameterConfig& getParameterConfig(const ParameterBlock& block) {
    CHECK_LT(block.para_id(), -1UL) << "invalid parameter id:"
                                    << block.para_id();
    const auto it = configMap_.find(block.para_id());
    CHECK(it != configMap_.end()) << "can not find parameter id: "
                                  << block.para_id();
    return it->second;
  }

  /// it implictly check blockOffsetMap_ while retrieving blockId
  const ParameterConfig& getParameterConfig(int64_t blockId) const {
    CHECK(blockId >= 0 && blockId < (int64_t)blockInfos_.size())
        << "block idx out of range, id: " << blockId
        << " info size: " << blockInfos_.size();
    return *(blockInfos_[blockId].config);
  }

  template <class Response>
  bool isValidVectorHandle(int64_t handle, Response* response) {
    if (handle < 0 || (size_t)handle >= vectors_.size()) {
      LOG(ERROR) << "Invalid vector handle " << handle;
      response->set_return_message(kRetMsgInvalidVectorHandle);
      return false;
    }
    return true;
  }

  template <class Response>
  bool isValidMatrixHandle(int64_t handle, Response* response) {
    if (handle < 0 || (size_t)handle >= matrices_.size()) {
      LOG(ERROR) << "Invalid matrix handle " << handle;
      response->set_return_message(kRetMsgInvalidMatrixHandle);
      return false;
    }
    return true;
  }

  /**
   * @brief get block offset
   *
   * @note  block.begin_dim is added to the block offset.
   *        return -1 if block cannot be found
   */
  int64_t getBlockOffset(const ParameterBlock& block) const {
    BlockKey key(block.para_id(), block.block_id());
    auto it = blockOffsetMap_.find(key);
    if (it == blockOffsetMap_.end()) {
      return -1;
    }
    return it->second;
  }

  /// return -1 if block cannot be found
  int64_t getBlockId(const ParameterBlock& block) const {
    BlockKey key(block.para_id(), block.block_id());
    auto it = blockIdMap_.find(key);
    if (it == blockIdMap_.end()) {
      return -1;
    }
    return it->second;
  }

  /**
   * @brief prepare data for sending back
   *
   * @note  modify reponse and outputBuffers for sending parameter
   *        back to client. The buffer for socket sending uses
   *        vectors_[parameterType] directly
   *        for dense with sync-sgd
   */
  void sendBackParameter(const ParameterBlock& block,
                         int parameterType,
                         SendParameterResponse* response,
                         std::vector<Buffer>* outputBuffers);

  /**
   * @brief prepare data for sending back
   *
   * @note  modify response and outputBuffers for sending parameter
   *        back to client. The buffer for socket sending uses buffer->base
   *        The parameter values are copied from vectors_[parameterType]
   *        to buffer->base.
   *        for dense with async-sgd
   */
  void sendBackParameter(const ParameterBlock& block,
                         int parameterType,
                         SendParameterResponse* response,
                         Buffer* buffer,
                         std::vector<Buffer>* outputBuffers);
  /**
   * @brief prepare data for sending back
   *
   * @note  specified for sparse
   */
  void sendBackParameterSparse(const ParameterBlock& block,
                               int parameterType,
                               SendParameterResponse* response,
                               Buffer* buffer,
                               size_t width,
                               std::vector<Buffer>* outputBuffers);

  /**
   * framework routine for block parallelization
   * e.g.
   * for optimization on all blocks at pserver end, this routine can facilitize
   * the parallelize of do optimization on all blocks with multithreads.
   */
  typedef std::function<void(int64_t blockId, const VectorPtr vecs[])> ExecFunc;
  void parallelExecForEachBlock(ExecFunc func);
  void blockTraverse(BlockInfo& info,
                     const ParameterConfig& config,
                     int64_t offset,
                     size_t size,
                     const VectorPtr vecs[],
                     const ParameterOptimizer::TraverseCallback& callback);

public:
  typedef void (ParameterServer2::*OperatorFunction)(const Operation& operation,
                                                     OperationResult* result);

  /**
   * doOperation will call following operations indirectly
   * e.g.
   * for sync-sgd control, the controller in remote updater will send op_SGD
   * command to pserver, then send sendParameter request to pserver immediately.
   * the two function at pserver end will do cooperation to achieve the sync-sgd
   * gradient merge and optimization.
   * the most following operations are specified for owlqn, all operations are
   * under the context of doOperation function
   */
  static OperatorFunction opFuncs[];

  void op_SGD(const Operation& operation, OperationResult* result);

  void op_RESET(const Operation& operation, OperationResult* result);

  void op_utv(const Operation& operation, OperationResult* result);

  void op_au_bv(const Operation& operation, OperationResult* result);

  void op_COPY(const Operation& operation, OperationResult* result);

  void op_au(const Operation& operation, OperationResult* result);

  void op_au_bv_cw(const Operation& operation, OperationResult* result);

  void op_make_steepest_desc_dir(const Operation& operation,
                                 OperationResult* result);

  void op_fix_dir_signs(const Operation& operation, OperationResult* result);

  void op_dir_deriv(const Operation& operation, OperationResult* result);

  void op_fix_omega_signs(const Operation& operation, OperationResult* result);

  void op_cost(const Operation& operation, OperationResult* result);

  void op_start_pass(const Operation& operation, OperationResult* result);
  void op_finish_pass(const Operation& operation, OperationResult* result);

  void op_apply(const Operation& operation, OperationResult* result);

  void op_randomize(const Operation& operation, OperationResult* result);

  void op_load(const Operation& operation, OperationResult* result);
  void op_save(const Operation& operation, OperationResult* result);

  /**
   * @brief output log in at the middle stage of training
   *
   * @note  flush log histroy and state at the end for sgd
   */
  void tuningSgdMidOutput();

  /**
   * @brief output log in at the end stage of training
   *
   * @note  flush log histroy and state at the end for sgd. it will also
   *        flush some stateful stat for next pass.
   */
  void tuningSgdFinished();

  /**
   * @brief output log in at the middle stage of training
   *
   * @note  flush log histroy and state at the end for async-sgd.
   *        it will log some performance log if some lagged node are found
   */
  void tuningAsyncsgdMidOutput();

  /**
   * @brief output log in at the end stage of training
   *
   * @note  flush log histroy and state at the end for async-sgd.
   */
  void tuningAsyncsgdFinished();
};

}  // namespace paddle
