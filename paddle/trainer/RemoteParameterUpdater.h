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

#include <functional>
#include <thread>
#include "ParameterUpdater.h"
#include "paddle/pserver/ParameterClient2.h"
#include "paddle/utils/Queue.h"
#include "paddle/utils/Util.h"

namespace paddle {

// TODO(yanfei):
// I think that the biggest feature of rdma is packet lossless control
// feature instead of high bandwiths, zero copy and gpu-direct rdma in
// theroy.
// But zero-copy and gpu-direct rdma features can help to reduce latency
// caused by os system.
// So, for some specified cluster, such as high density gpu cluster,
// gpu-direct and zero copy could help to improve cluster communication
// performance.
//

/**
 * Normal remote parameter updater for dense parameters.
 *
 * It first packs all parameters for all pservers using ParameterClient
 * module, then wait for merged parameters data from all pservers.
 * The synchronization pattern specified by sync-sgd or async-sgd is
 * achieved by all pservers with the help of the controller within this
 * remote parameter updater.
 * This module indeedly bridges the gradient machines and parameter servers.
 * It helps to transfer the parameters from acceleration device to cpu end
 * for network. It contains additional parameters copy buffers for
 * acceleration devices at cpu end, such as gpu, otherwise it will
 * directly use original parameters data to update pservers.
 *
 * This remote parameter updater does not use pipeline mechanism to hide
 * copy latency from gpu to cpu buffer. In addition the overlapped between
 * backward and communication is not supported.
 */
class RemoteParameterUpdater : public ParameterUpdater {
 public:
  RemoteParameterUpdater(
      const OptimizationConfig& config,
      int expectedPassCount,
      std::unique_ptr<ParameterUpdater>&& localUpdater = nullptr);
  ~RemoteParameterUpdater() {
    if (controllerThread_) {
      controllerThread_->join();
    }
  }

  /**
   * initialize the internal parameter client and itself.
   */
  virtual void init(const std::vector<ParameterPtr>& parameters);
  /**
   * @brief start batch
   *
   * @note  one batch training exhibits stateful feature to help
   *        to do performance tuning, sgd optimization if necessary.
   */
  virtual PassType startBatch(int64_t batchSize) {
    if (localUpdater_) {
      localUpdater_->startBatch(batchSize);
    }
    batchSize_ = batchSize;
    batchStatus_ = BATCH_START;
    return PASS_TRAIN;
  }

  /**
   * send parameters to pservers and get returned parameters
   * from all pservers if necessary. it will implictly
   * cooperate with controller thread for sync-sgd.
   */
  virtual void finishBatch(real cost);
  virtual void startPass();
  virtual bool finishPass();

#ifndef PADDLE_DISABLE_TIMER
  virtual void setForwardbackwardTime(uint64_t delta) {
    parameterClient_->setForwardbackwardTime(delta);
  }
#endif

  virtual void apply();
  virtual void restore();

 protected:
  /**
   * control all pservers with all trainers for sync-sgd
   */
  virtual void controller();

  /**
   * work need to do after finishBatch
   */
  virtual void updateImpl(Parameter* para);

  void startController();

  /**
   * @brief copy parameters from cpu host to device, such as gpu.
   *
   * @note  return if all data are transfered.
   */
  void copyParametersToDevice(ParameterType parameterType);

  /**
   * @brief copy parameters from device to cpu host
   *
   * @note  return if all data are transfered
   */
  void copyParametersFromDevice(ParameterType parameterType);

 protected:
  /// Optimization config used to guide initialization and finishBatch
  OptimizationConfig config_;
  /// internal parameter client object for exchanging data with pserver
  std::unique_ptr<ParameterClient2> parameterClient_;
  /// internal shadow buffer at cpu host end, use original parameters_
  /// if no acceleration devices are used.
  std::vector<ParameterPtr> cpuParameters_;
  /// local updater for aggregating multi-batches local delta
  std::unique_ptr<ParameterUpdater> localUpdater_;
  /// the size of mini-batch
  int64_t batchSize_;
  /// batches passed
  int64_t numBatches_;
  /// for stateful control
  BatchStatus batchStatus_;
  /// controller thread for sync-sgd
  std::unique_ptr<std::thread> controllerThread_;
  /// passed already finished
  int64_t passCount_;
  /// expected passes to finished
  int64_t expectedPassCount_;
  /// use normal synchronization communication if True
  bool separateSendAndRecv_;
  /// true if it's first pass
  bool isFirstPass_;
  bool useApplyInPserver_;

  static const std::string kAverage;
  static const std::string kElasticAverage;
};

// TODO(yanfei):
// do parameters level synchronization Optimization at pserver end with
// ConcurrentRemoteParameterUpdater to get more parallelization, at last
// to really hide pserver latency in backward computation.
//
/**
 * This updater add additional optimization for overlapping synchronization
 * from pservers with backward computation.
 *
 * Parameter can be sent to pservers when related backward stage is finished.
 * This concurrent udpater does data copy from acceleration device to host
 * memory aynchronously. In addition internal parameter client reads data in
 * host memory and send them to all pservers in next stage. So this class
 * help to pipeline device-to-host copy and host-to-network to hide network
 * latency in backward stage.
 * It contains separate send and recv thread for pipeline usage.
 */
class ConcurrentRemoteParameterUpdater : public RemoteParameterUpdater {
 public:
  ConcurrentRemoteParameterUpdater(
      OptimizationConfig config,
      int expectedPassCount,
      std::unique_ptr<ParameterUpdater>&& localUpdater);
  ~ConcurrentRemoteParameterUpdater();

  /**
   * @brief send paraemeters to all pservers
   *
   * @note  it just signal the end signal to internal parameter client
   *        to finished the aynchronous send action. In addition it also
   *        do synchronization for all asynchronous host-to-device copy.
   */
  virtual void finishBatch(real cost);

 protected:
  virtual void updateImpl(Parameter* para);
  /// internal thread called in send thread
  void send(Parameter* para);  // para == NULL indicate end of a minibatch
  /// internal function called in recv thread
  void recv(Parameter* para);
  /**
   * @brief send thread for relaying data from gradient to parameter client
   *
   * @note  just pipe data to internal parameter client for pipeline
   */
  void send();
  /**
   * @brief recv thread for relaying data from internal parameter client to
   *        host memory
   *
   * @note  it contains the asynchronous data copy form host to device
   */
  void recv();
  /// copy specified parameter from host to device
  void copySingleParaToDevice(Parameter* para, ParameterType parameterType);
  /// copy specified parameter from device to host
  void copySingleParaFromDevice(Parameter* para, ParameterType parameterType);
  bool needToUpdateRemotely() {
    return (numBatches_ + 1) % config_.num_batches_per_send_parameter() == 0;
  }

 private:
  /// send thread used for overlapping
  std::unique_ptr<std::thread> sendThread_;
  /// recv thread used for overlapping
  std::unique_ptr<std::thread> recvThread_;
  /// buffer queue for overlapping
  Queue<int> sendQueue_;
  /// buffer queue for overlapping
  Queue<int> recvQueue_;
  /// flags indicating to stop
  bool stopping_;
  /// conditional variable for threads synchronization between the
  /// thread calling finishBatch and internal recv thread
  LockedCondition finishBatchCond_;
  bool oneBatchFinished_;
};

// TODO(yanfei):
// merge sparse updater with dense updater, and could help to reduce
// the synchronization between sparse and dense udpater. it could also
// reduce the threads for managing all connections.
/**
 * This class is specified for updating sparse parameters.
 *
 * It allows part of parameter to be exchanged with all pservers.
 * If sparse input assigned, part gradients of first hidden layer
 * could remained zero which can not need to be exchanged within
 * all pservers. This is the key optimization point for this updater
 *
 * For updating sparse parameters, all latest parameters are stored
 * in pservers instead of keeping full copy at train end, so need to
 * prefetch parameters weight value which can be changed in next-batch
 * before doing next forwardbackward. Also, with above fact that the
 * parameters can be stored in pserver instead of trainer, we can
 * fetch specified parmeters if necessary, and can support huge
 * parameters which is larger enough than  the RAM size in single
 * node.
 *
 * Internally, this updater will direct internal parameter client
 * to encapsulate sparse specified message for all pservers.
 */
class SparseRemoteParameterUpdater : public ParameterUpdater {
 public:
  SparseRemoteParameterUpdater(const OptimizationConfig& config,
                               int expectedPassCount,
                               bool testing);
  ~SparseRemoteParameterUpdater() {
    if (controllerThread_) {
      controllerThread_->join();
    }
  }

  /// initialization
  virtual void init(const std::vector<ParameterPtr>& parameters);

  /// stateful batch control
  virtual PassType startBatch(int64_t batchSize);
  /// send all sparse related parameters to all pservers
  virtual void finishBatch(real cost);
  virtual void startPass();
  virtual bool finishPass();

  virtual void apply();
  virtual void restore();

  /// load parameters from pservers
  virtual void loadParametersRemote(const std::string& dirName);
  /// save parameters to pservers
  virtual void saveParametersRemote(const std::string& dirName);
  /**
   * @brief get latest sparse parameters value from all pservers
   *
   * @note  call it before next mini-batch
   */
  virtual void getParametersRemote(bool fullSize, bool apply);
  virtual void randParametersRemote();
#ifndef PADDLE_DISABLE_TIMER
  virtual void setForwardbackwardTime(uint64_t delta) {
    parameterClient_->setForwardbackwardTime(delta);
  }
#endif

 protected:
  /// update implimentation, not implemented
  virtual void updateImpl(Parameter* para) {}

  /// internal controller routine for controller thread
  virtual void controller();

  /// start controller thread
  void startController();

 protected:
  /// optimization config
  OptimizationConfig config_;
  /// internal parameter client
  std::unique_ptr<ParameterClient2> parameterClient_;
  int64_t batchSize_;
  std::unique_ptr<std::thread> controllerThread_;
  int64_t passCount_;
  int64_t expectedPassCount_;
  bool testing_;
  bool useApplyInPserver_;
};

/**
 * Class for supporting normal updater and sparse updater
 *
 * Not all parts of one model are sparse, so it exists dense updater
 * for normal layers while sparse updater is for sparse layers.
 *
 * it directly call internal dense and sparse udpater individually.
 */
class SparseRemoteParameterUpdaterComposite : public ParameterUpdaterComposite {
 public:
  enum {
    UPDATER_SPARSE_REMOTE = 0,  // execute in sync thread pool(tid:0)
    UPDATER_NORMAL = 1,         // execute in Owner thread(tid:1)
    NUMBER_UPDATERS = 2,
  };
  /**
   * @brief create one dense updater and one sparse updater
   *
   * @note  use syncThreadPool to synchronize these two updaters
   */
  SparseRemoteParameterUpdaterComposite(
      const OptimizationConfig& config,
      int expectedPassCount,
      bool testing,
      std::unique_ptr<ParameterUpdater>&& normalUpdater) {
    updaters_.resize(NUMBER_UPDATERS);
    updaters_[UPDATER_SPARSE_REMOTE].reset(
        new SparseRemoteParameterUpdater(config, expectedPassCount, testing));
    updaters_[UPDATER_NORMAL] = std::move(normalUpdater);

    syncThreadPool_.reset(new SyncThreadPool(NUMBER_UPDATERS - 1));
  }

  /// initialization of dense and sparse updaters
  virtual void init(const std::vector<ParameterPtr>& parameters);
};

class ParameterUpdaterCreators {
 public:
  /**
   * @brief add a creator to create custom ParameterUpdater while training.
   *        The creator is a function with type (alogrithm, optConfig, isLocal,
   *        numPasses) -> ParameterUpdater*. Trainer will use this
   *        ParameterUpdater if creator can create a no nullptr
   *        ParameterUpdater. Return nullptr will use trainer's default
   *        updaters.
   *
   * @param creator method which can create ParameterUpdater.
   */
  static void addCreator(
      const std::function<ParameterUpdater*(
          const std::string&,         // algo
          const OptimizationConfig&,  // optConfig
          bool,                       // isLocal
          size_t                      // numPasses
          )>& creator) {  // NOLINT  explicit move closing ) in this line
                          // for readability
    constructors_.push_back(creator);
  }

  /**
   * @brief Try to create an updater by given algo, optConfig, isLocal,
   *        numPasses. Return nullptr if cannot create anyone.
   * @param algo algorithm string.
   * @param optConfig optimization config.
   * @param isLocal is in local mode or not.
   * @param numPasses total passes that trainer will train.
   * @return nullptr if fail, not nullptr if we can create an updater.
   */
  static ParameterUpdater* tryCreateUpdater(const std::string& algo,
                                            const OptimizationConfig& optConfig,
                                            bool isLocal,
                                            size_t numPasses) {
    for (auto& c : constructors_) {
      if (auto updater = c(algo, optConfig, isLocal, numPasses)) {
        return updater;
      }
    }
    return nullptr;
  }

 private:
  static std::vector<std::function<ParameterUpdater*(
      const std::string&, const OptimizationConfig&, bool, size_t)>>
      constructors_;
};

}  // namespace paddle
