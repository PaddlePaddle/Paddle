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

#include <atomic>

#include "GradientMachine.h"

#include "hl_gpu.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/Queue.h"

namespace paddle {

class TrainerThread;

typedef Queue<int> PidQueue;
typedef std::unique_ptr<TrainerThread> TrainerThreadPtr;

struct GradBuffer {
  /// GradBuffer is used for gathering gradient for GPU parameters
  int paramId;

  /// sem is used to notify that the local gradient merge of the current thread
  /// finished for the current thread.
  Semaphore sem;

  // bufs[mergeIndex]
  std::vector<VectorPtr> bufs;
};

/**
 *  A MultiGradientMachine is a synchronous GradientMachine which devides
 *  one data batch into several smaller batches and assign each one small batch
 *  to one computint thread for computation. After each thread finishes
 *  computation, it merges result (including output Argument and gradient during
 *  backward()). It basically is the same as single thread gradient machine,
 *  except that it uses multi-thread to do the computation.
 *
 *  It handles GPU and Cpu parameters differently.  In GPU, one computing thread
 *  generally corresponds to one GPU device. Thus, each thread keeps a separate
 *  copy of the parameter in its own device's memory. In CPU, we only need to
 keep
 *  one copy of the parameters in the main memory. After, each computing thread
 *  computes its own parameter gradient, the update process needs to accumulate
 *  the parameter gradients from all the computing threads, and update the
 *  accumulated parameter gradient to the corresponding parameter value.
 *
 *  Each GPU parameter is assigned to a thread called its main thread. For each
 *  parameter, the accumulation of its gradients and the update of its value
 *  happens in its main thread. The main thread first gather the parameter
 *  gradients from all the computing thread. Then, it performs parameter update.
 *  After a gradient is updated by the main thread, it is scattered to all the
 *  computing thread so that the parameters in all the computing threads are
 *  synchronized. The scatter and gather process are implemented by ring-style
 *  communication. Assume we have N computing threads, its thread ids will be
 *  0, 1, ..., N-1. For each parameter, the id of the main thread is specified
 in
 *  paraMainThread_[pid], where pid is the id of the parameter. Each thread i
 only
 *  sends data to its partner thread (i - 1) % N. For example, for a parameter
 *  gradient that is computed in thread 4, and its main thread is 2. Its
 *  traveling process would be 4, 5,..., N-1, 0, 1, 2. In each step, the
 gradient
 *  buffer is added to the local gradient, and the local gradient is then copied
 *  to the gradient buffer of the next thread. At last, its main thread 2 will
 *  get the accumulated parameter gradient. For the same parameter, after its
 *  value is updated, the value's traveling process would be 2, 1, 0, N-1, ...
 3.
 *  At the end, all the computing threads would have the updated parameter
 value.
 *
 *  A computing thread (TrainerThread) uses 4 threads to do different jobs:
 *
 *  1. computeThread(): performing forward(), backward(), prefetch().
 *
 *  2. valueDispatchThread(): copying parameter values to partner thread.
 *
 *  3. copyGradToBufferThread(): copying parameter gradient to partner thread.
 *
 *  4. gradCollectThread(): merging the gradient from step 3 with local gradient
 *     and call the callback supplied by the user to update parameter value.
 *
 *  CPU parameter value has only one copy. And their gradients are merged at the
 *  end of backward().
 *
 *  * Handling of sparse update
 *  Currently, sparse update is only supported for CPU parameters.

 *  Sparse updates refers to gradient caculation where the gradient is sparse.
 For
 *  example, if the input argument to a 'fc' layer is sparse, the gradient of
 the
 *  weight matrix of this layer will be sparse. It is usually more efficient to
 *  treat the gradient explicitly as sparse vector during the parameter update.

 *  There are two types of sparse updates called local sparse update and remote
 *  sparse update.

 *  For both types of sparse updates, there is one copy of parameter value and
 *  gradient called main parameter value and gradient, and there is a copy of
 *  parameter value and gradient for each computing thread called slave
 parameter
 *  value and gradient. The slave parameter values are always shared with the
 *  corresponding main parameter value. The slave parameter grad is a sparse row
 *  matrix. The sparse pattern for slave parameter grads are different, because
 *  the small batches for each computing thread might have different sparsity
 *  pattern.

 *  1. Local sparse update
 *
 *     Main parameter value type is MAT_NORMAL. It is a dense matrix.
 *
 *     Main parameter grad type is MAT_SPARSE_ROW_IDS (SparseRowIdsCpuMatrix)
 *     It is also a dense matrix, but the updated values are specified by IDS.
 *
 *     Slave parameter value shares with main parameter value.
 *
 *     Slave parameter grad type is MAT_SPARSE_ROW_AUTO_GROW
 *     (SparseAutoGrowRowCpuMatrix). It is a sparse row matrix.
 *
 *     During backward() of each TrainerThread, SparseAutoGrowRowCpuMatrix will
 *     gather all the non-zero gradient. And After backward(), they will be
 merged
 *     into main parameter grad (SparseRowIdsCpuMatrix), with indices indicating
 *     which rows have nonzero gradient.
 *
 *  2. Remote sparse update
 *
 *     Main parameter value type is MAT_SPARSE_ROW_PREFETCH(_FULL_SIZE)
 *     (SparsePrefetchRowCpuMatrix). MAT_SPARSE_ROW_PREFETCH is a sparse matrix.
 *     MAT_SPARSE_ROW_PREFETCH_FULL_SIZE is a dense matrix. However, only the
 *     parameter values that are prefetched is up-to-date.
 *
 *     Main parameter grad type is MAT_SPARSE_ROW (SparseRowCpuMatrix).
 *     And it shares sparse pattern with value by sharing indexDictHandle_,
 which
 *     is an internal data structure used by SparseRowCpuMatrixto specify the
 *     sparsity pattern of Slave parameter value shares with main parameter
 value.
 *
 *     Slave parameter grad type is MAT_SPARSE_ROW_AUTO_GROW
 *     (SparsePrefetchRowCpuMatrix). It is a sparse row matrix
 *
 *     During prefetch(), all the layers will indicates which rows of each
 *     parameter are needed. Then the framework will retrieve those rows from
 *     parameter server.
 *
 *     During backward() of each TrainerThread, SparseAutoGrowRowCpuMatrix will
 *     gather all the non-zero gradient. And After backward(), they will be
 merged
 *     into main parameter grad (SparseRowCpuMatrix). And the framework will
 send
 *     the merged gradient to parameter server.
 */
class MultiGradientMachine : public GradientMachine {
 public:
  enum TaskType {
    TASK_FORWARD_BACKWARD = 0,
    TASK_FORWARD = 1,
    TASK_BACKWARD = 2,
    TASK_COPY_IN_ARGS = 3,
  };

  explicit MultiGradientMachine(const ModelConfig& config, bool useGpu);

  virtual void start();

  virtual void finish();

  virtual void prefetch(const std::vector<Argument>& inArgs);

  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  void forwardBackward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType,
                       const UpdateCallback& callback);

  virtual Argument getLayerOutput(const std::string& layerName);

  virtual void onPassEnd();

  virtual Evaluator* makeEvaluator() const;

  virtual void eval(Evaluator* evaluator) const;

  bool useGpu() const { return useGpu_; }

  /// @return whether to pass the gradients in outArgs_ to each threads.
  bool isPassGrad() { return isPassGrad_; }

  /// @brief set whether to pass the gradient in outArgs_ to each threads.
  void setPassGrad(bool isPass) { isPassGrad_ = isPass; }

  /// Set the gradients of the outputs.
  /// The gradietns will be copied to each thread in the computing threads.
  virtual void setOutputGrad(const std::vector<Argument>& args);

 protected:
  friend class TrainerThread;

  std::vector<TrainerThreadPtr>& getAllThreads() { return threads_; }
  /// Calculate the real device id based on the logical device id and the
  /// thread id.
  int logicalDeviceId2RealDeviceId(int logicalId, int threadId = 0) const {
    if (logicalId == -1) {
      logicalId = 0;
    }
    return mod(logicalId + FLAGS_gpu_id + threadId * numLogicalDevices_,
               numDevices_);
  }

  /// Calculate the logical device id based on the real device id and the
  /// thread id.
  int realDeviceId2LogicalDeviceId(int realId, int threadId = 0) const {
    if (realId == -1) {
      return 0;
    } else {
      return mod(realId - FLAGS_gpu_id - threadId * numLogicalDevices_,
                 numDevices_);
    }
  }

  std::vector<const std::vector<ParameterPtr>*> getSlaveParameters();

  bool hasNonstaticCpuParamters() const { return hasNonstaticCpuParamters_; }

  /// Called TrainerThread to wait before merging CPU parameter gradients.
  void waitBeforeMerge() { trainerBarrier_.wait(); }

  /// called by MultiGradientMachine and TrainerThread to wait after merging
  /// CPU parameter graidents.
  void waitAfterMerge() { allBarrier_.wait(); }

  /// called by MultiGradientMachine and TrainerThread to wait for copyInArgs()
  /// finishing
  void waitForCopyInArgs() { allBarrier_.wait(); }

  TrainerThreadPtr& getThread(int threadId) { return threads_[threadId]; }

  std::vector<GradBuffer>& getGradBuf(int threadId) {
    return gradBufs_[threadId];
  }

  PassType getPassType() const { return passType_; }

  /// Called by TrainerThread to notify MultiGradientMachine that the gradient
  /// for paramId is ready
  void notifyGradientTransfer(int paramId);

  const std::vector<Argument>& getInArgs() { return inArgs_; }

  TaskType getTaskType() const { return taskType_; }

  const UpdateCallback& getBackwardCallback() const {
    return backwardCallback_;
  }

  int getNumDevices() const { return numDevices_; }

  int getNumLogicalDevices() const { return numLogicalDevices_; }

  int getNumThreads() const { return numThreads_; }

  int paraMainThread(int pid) const { return paraMainThread_[pid]; }

 protected:
  virtual void forwardImp(const std::vector<Argument>& inArgs,
                          std::vector<Argument>* outArgs,
                          PassType passType,
                          TaskType taskType);

  virtual void backwardImp(const UpdateCallback& callback = NULL);

  /// update all parameters
  void updateThreadParameters();

  void startTask(TaskType taskType);

  void getOutArgs(std::vector<Argument>* outArgs, PassType passType);

  void allocGradBufs();

 protected:
  bool useGpu_;

  bool hasNonstaticCpuParamters_;

  /// store main parameter only
  std::unique_ptr<GradientMachine> gradientMachine_;

  std::vector<TrainerThreadPtr> threads_;
  std::vector<int> paraMainThread_;
  std::vector<std::vector<GradBuffer>> gradBufs_;  // [threadId][deviceId]
  std::vector<size_t> bufferSizes_;

  PassType passType_;
  TaskType taskType_;
  PidQueue gradQueue_;
  std::vector<Argument> inArgs_;
  std::vector<Argument> outArgs_;
  hl_stream_t outArgStream_;

  Argument outLayerArgs_;

  /// ParameterType which needs to be merged from each GPU
  std::vector<ParameterType> mergeTypes_;
  int numDevices_;         /* number of gpu devices */
  int numLogicalDevices_;  // number of GPU used by one NN
  int numThreads_;         /* number of train threads */

  UpdateCallback backwardCallback_;

  /// barrrier for threads_
  ThreadBarrier trainerBarrier_;

  /// barrier for both MultiGradientMachine and threds_
  ThreadBarrier allBarrier_;

  /// indicate whether inArgs is copied before forward()
  bool inArgsCopied_;

  /// Whether to copy the gradient back from an external input.
  bool isPassGrad_;
};

class TrainerThread {
 public:
  TrainerThread(const ModelConfig& config,
                int threadId,
                MultiGradientMachine* multiMachine);

  ~TrainerThread();

  void start();

  void onPassEnd() { gradientMachine_->onPassEnd(); }

  void waitOutArgsReady() { outArgsReadySem_.wait(); }

  void notifyTaskReady() { taskReadySem_.post(); }

  int getDeviceId() const { return deviceId_; }

  GradientMachine* getGradientMachine() { return gradientMachine_.get(); }

  const std::vector<ParameterPtr>& getParameters() { return parameters_; }

  void stop();

  void notifyValueReady(int paramId);

  const VectorPtr& getValueBuf(int paramId) {
    return parameters_[paramId]->getBuf(PARAMETER_VALUE);
  }

  const std::vector<Argument>& getOutArgs() { return outArgs_; }

  void incUpdateCounter(int n = 1) {
    updateCounter_ += n;
    parameterUpdated_ = true;
  }

  void notifyGradientCollect(int paramId) { gradQueue_.enqueue(paramId); }

  void notifyCopyGradToBuffer(int paramId) { gradBufQueue_.enqueue(paramId); }

  void notifyValueDispatch(int paramId) { valueReadyQueue_.enqueue(paramId); }

  void prefetch();

  /// copy the output gradient from the main GradientMachine.
  void copyOutputGrad();

  /// Whether the thread has input data.
  bool hasInputData() { return batchSize_ != 0; }

 protected:
  void mergeCpuGradients();

  void mergeGradSparse(
      Parameter* para,
      std::vector<const std::vector<ParameterPtr>*>& slaveParameters);

  void mergeGradSparseRemote(
      Parameter* para,
      std::vector<const std::vector<ParameterPtr>*>& slaveParameters);

  void mergeGradDense(
      Parameter* para,
      std::vector<const std::vector<ParameterPtr>*>& slaveParameters);

  void computeThread();
  void valueDispatchThread();
  void copyGradToBufferThread();
  void gradCollectThread();

  int copyInArgs();
  void forward();
  void backward();
  void backwardCallback(Parameter* para);

  /// call the actuall callback supplied by the caller of
  /// GradientMachine::backward
  void doCallback(int pid);

 protected:
  MultiGradientMachine* multiMachine_;
  ModelConfig config_;
  /// whether the thread should stop
  bool stopping_;
  /// the threads form which to collect gradient
  int partnerId_;
  /// from 0 to threads-1
  int threadId_;
  int deviceId_;
  std::unique_ptr<GradientMachine> gradientMachine_;
  std::vector<ParameterPtr> parameters_;

  /// ParameterType which needs to be merged from each GPU
  std::vector<ParameterType> mergeTypes_;

  /// compute thread
  std::unique_ptr<std::thread> computeThread_;
  std::vector<Argument> inArgs_;
  std::vector<Argument> outArgs_;
  Semaphore taskReadySem_;
  Semaphore outArgsReadySem_;

  /// copy thread
  std::unique_ptr<std::thread> copyThread_;
  /// queue of gradient needs to be copied to partner
  PidQueue gradBufQueue_;
  hl_stream_t gradStream_;

  /// grad merge thread
  std::unique_ptr<std::thread> gradCollectThread_;
  /// queue of gradient needs to be merged with gradient coopied by
  /// copyGradToBufferThread
  PidQueue gradQueue_;
  UpdateCallback backwardCallback_;

  /// value dispatch thread
  std::unique_ptr<std::thread> valueDispatchThread_;
  /// queue of the parameter whose the vale are ready for copy
  PidQueue valueReadyQueue_;

  /// used to notify all the parameter values are ready
  LockedCondition valueReadyCond_;

  hl_stream_t valueStream_;
  /// how many parameters are updated
  std::atomic<int> updateCounter_;
  bool parameterUpdated_;

  /// indicate whether inArgs is copied before forward()
  bool inArgsCopied_;
  int batchSize_;
};

}  // namespace paddle
