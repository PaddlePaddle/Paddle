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

#include "MultiGradientMachine.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Stat.h"

#include "NeuralNetwork.h"
#include "ParallelNeuralNetwork.h"

DEFINE_bool(allow_only_one_model_on_one_gpu,
            true,
            "If true, do not allow multiple models on one GPU device");

namespace paddle {

// get types of the parameters which need to be merged after backward()
static void fillMergeTypes(PassType passType,
                           std::vector<ParameterType>* mergeTypes) {
  mergeTypes->clear();
  if (passType != PASS_TEST) {
    mergeTypes->push_back(PARAMETER_GRADIENT);
  }
}

MultiGradientMachine::MultiGradientMachine(const ModelConfig& config,
                                           bool useGpu)
    : useGpu_(useGpu),
      trainerBarrier_(FLAGS_trainer_count),
      allBarrier_(FLAGS_trainer_count + 1),
      inArgsCopied_(false) {
  isPassGrad_ = false;
  numThreads_ = FLAGS_trainer_count;
  if (useGpu) {
    //! TODO(yuyang18): When useGpu=false && paddle is not compiled with gpu,
    //! the hl_get_device_count will get an error result. It seems should return
    //! 0 when hppl is not compiled as gpu version.
    numDevices_ = hl_get_device_count();
  } else {
    numDevices_ = 0;
  }
  ParamInitCallback mainParamInitCb = [](int paramId, Parameter* para) {
    // only create buf for CPU parameters
    // GPU parameters will be created in each thread
    if (para->useGpu()) return;

    if (para->isSparseRemoteUpdate()) {
      para->enableType(PARAMETER_VALUE,
                       FLAGS_loadsave_parameters_in_pserver
                           ? Parameter::MAT_SPARSE_ROW_PREFETCH
                           : Parameter::MAT_SPARSE_ROW_PREFETCH_FULL_SIZE);
      para->enableType(PARAMETER_GRADIENT, Parameter::MAT_SPARSE_ROW);
    } else if (para->isGradSparseUpdate()) {
      para->enableType(PARAMETER_VALUE);
      para->enableType(PARAMETER_GRADIENT, Parameter::MAT_SPARSE_ROW_IDS);
      SparseRowIdsCpuMatrix* mat = dynamic_cast<SparseRowIdsCpuMatrix*>(
          para->getMat(PARAMETER_GRADIENT).get());
      mat->setNumOfThreads(FLAGS_trainer_count);
    } else if (para->isValueShared()) {
      para->enableType(PARAMETER_VALUE, Parameter::MAT_VALUE_SHARED);
      if (!para->isStatic()) {
        para->enableType(PARAMETER_GRADIENT);
      }
    } else {
      para->enableType(PARAMETER_VALUE);
      if (!para->isStatic()) {
        para->enableType(PARAMETER_GRADIENT);
      }
    }
  };

  NeuralNetwork* nn = NeuralNetwork::create(config);
  nn->init(config, mainParamInitCb);
  gradientMachine_.reset(nn);
  parameters_ = gradientMachine_->getParameters();

  numLogicalDevices_ = 0;
  if (useGpu_) {
    numLogicalDevices_ = 1;

    for (size_t pid = 0; pid < parameters_.size(); pid++) {
      if (parameters_[pid]->getConfig().device() + 1 > numLogicalDevices_) {
        numLogicalDevices_ = parameters_[pid]->getConfig().device() + 1;
      }
    }
    LOG(INFO) << "numLogicalDevices=" << numLogicalDevices_
              << " numThreads=" << numThreads_ << " numDevices=" << numDevices_;

    if (numLogicalDevices_ * numThreads_ > numDevices_ &&
        FLAGS_allow_only_one_model_on_one_gpu) {
      LOG(FATAL) << "trainer_count * num_devices_in_model "
                 << "(" << numThreads_ << "*" << numLogicalDevices_ << ")"
                 << "=" << numThreads_ * numLogicalDevices_
                 << " exceeds number of GPU devices(" << numDevices_ << ")";
    }
    numLogicalDevices_ = std::min(numLogicalDevices_, numDevices_);

    /* Enables direct access to memory allocations on a peer device */
    for (int i = 0; i < numThreads_; i++) {
      for (int d = 0; d < numLogicalDevices_; ++d) {
        enablePeerAccess(logicalDeviceId2RealDeviceId(d, i),
                         logicalDeviceId2RealDeviceId(d, i + 1));
        enablePeerAccess(logicalDeviceId2RealDeviceId(d, i),
                         logicalDeviceId2RealDeviceId(d, i - 1));
      }
    }
  }

  for (int i = 0; i < numThreads_; ++i) {
    threads_.emplace_back(new TrainerThread(config, i, this));
  }

  bufferSizes_.resize(numLogicalDevices_, 0);
  paraMainThread_.reserve(parameters_.size());
  int pid = 0;
  for (auto& para : parameters_) {
    if (para->isStatic() || !para->useGpu()) {
      paraMainThread_.push_back(0);
    } else {
      int end = pid++ % numThreads_;
      paraMainThread_.push_back(end);
      int paraDeviceId = para->getDeviceId();
      if (paraDeviceId == -1) paraDeviceId = 0;
      paraDeviceId = paraDeviceId % numLogicalDevices_;
      if (para->getSize() > bufferSizes_[paraDeviceId]) {
        bufferSizes_[paraDeviceId] = para->getSize();
        VLOG(1) << "bufferSize[" << paraDeviceId << "]" << para->getSize();
      }
    }
  }

  // TODO(xuwei06) Instead of using maximal buffer size, we may use a smaller
  // fixed buffer size and use pipeline to dispatch parameter value and merge
  // parameter gradient, which may be faster.

  // combination of all trainers mainPara into GradientMachine parameters
  hasNonstaticCpuParamters_ = false;
  for (size_t pid = 0; pid < parameters_.size(); pid++) {
    if (parameters_[pid]->useGpu()) {
      parameters_[pid] = threads_[paraMainThread_[pid]]->getParameters()[pid];
    } else if (!parameters_[pid]->isStatic()) {
      hasNonstaticCpuParamters_ = true;
    }
  }

  gradBufs_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    gradBufs_[i].resize(numLogicalDevices_);
    for (int d = 0; d < numLogicalDevices_; ++d) {
      gradBufs_[i][d].sem.post();
    }
  }

  outArgStream_ = HPPL_STREAM_1;

  start();
}

void MultiGradientMachine::start() {
  for (auto& thread : threads_) {
    thread->start();
  }
}

void MultiGradientMachine::finish() {
  for (auto& thread : threads_) {
    thread->stop();
  }
}

std::vector<const std::vector<ParameterPtr>*>
MultiGradientMachine::getSlaveParameters() {
  std::vector<const std::vector<ParameterPtr>*> vec;
  vec.reserve(threads_.size());
  for (auto& thread : threads_) {
    vec.push_back(&thread->getParameters());
  }
  return vec;
}

void MultiGradientMachine::notifyGradientTransfer(int paramId) {
  gradQueue_.enqueue(paramId);
}

void MultiGradientMachine::allocGradBufs() {
  if (numLogicalDevices_ == 0) return;
  if (gradBufs_[0][0].bufs.size() >= mergeTypes_.size()) return;

  for (int i = 0; i < numThreads_; i++) {
    for (int d = 0; d < numLogicalDevices_; ++d) {
      if (bufferSizes_[d] == 0) continue;
      SetDevice device(logicalDeviceId2RealDeviceId(d, i));
      for (size_t j = 0; j < mergeTypes_.size(); j++) {
        gradBufs_[i][d].bufs.push_back(
            Vector::create(bufferSizes_[d], /* useGpu= */ true));
      }
    }
  }
}

void MultiGradientMachine::prefetch(const std::vector<Argument>& inArgs) {
  // Each gradient machine in threads needs to do prefetch on its own
  // part of inArgs. So we need to first divide inArgs to each thread
  inArgs_ = inArgs;
  startTask(TASK_COPY_IN_ARGS);

  for (auto& para : parameters_) {
    if (para->isSparseRemoteUpdate()) {
      auto mat = dynamic_cast<SparsePrefetchRowCpuMatrix*>(
          para->getMat(PARAMETER_VALUE).get());
      mat->clearIndices();
    }
  }

  waitForCopyInArgs();

  // Because SparsePrefetchRowCpuMatrix can only be changed by ONE thread
  // at one time, we need to do prefetch sequentially
  for (auto& thread : threads_) {
    thread->prefetch();
  }

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

void MultiGradientMachine::forward(const std::vector<Argument>& inArgs,
                                   std::vector<Argument>* outArgs,
                                   PassType passType) {
  forwardImp(inArgs, outArgs, passType, TASK_FORWARD);
}

void MultiGradientMachine::forwardImp(const std::vector<Argument>& inArgs,
                                      std::vector<Argument>* outArgs,
                                      PassType passType,
                                      TaskType taskType) {
  updateThreadParameters();
  passType_ = passType;

  if (!inArgsCopied_) {
    inArgs_ = inArgs;
    inArgsCopied_ = false;
  }

  fillMergeTypes(passType, &mergeTypes_);
  allocGradBufs();
  startTask(taskType);

  getOutArgs(outArgs, passType);
}

void MultiGradientMachine::backward(const UpdateCallback& callback) {
  backwardCallback_ = callback;
  startTask(TASK_BACKWARD);
  backwardImp(callback);
}

void MultiGradientMachine::forwardBackward(const std::vector<Argument>& inArgs,
                                           std::vector<Argument>* outArgs,
                                           PassType passType,
                                           const UpdateCallback& callback) {
  backwardCallback_ = callback;
  forwardImp(inArgs, outArgs, passType, TASK_FORWARD_BACKWARD);
  backwardImp(callback);
}

Argument MultiGradientMachine::getLayerOutput(const std::string& layerName) {
  std::vector<Argument> args;
  args.reserve(threads_.size());

  for (auto& thread : threads_) {
    args.push_back(thread->getGradientMachine()->getLayerOutput(layerName));
  }
  outLayerArgs_.concat(args, false /* use_gpu */, outArgStream_, passType_);

  return outLayerArgs_;
}

void MultiGradientMachine::backwardImp(const UpdateCallback& callback) {
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (!parameters_[i]->useGpu() || parameters_[i]->isStatic()) continue;
    REGISTER_TIMER("controller_dequeue");
    gradQueue_.dequeue();
  }
  if (hasNonstaticCpuParamters()) {
    waitAfterMerge();
    if (backwardCallback_) {
      for (auto& para : parameters_) {
        if (!para->useGpu() && !para->isStatic()) {
          backwardCallback_(para.get());
        }
      }
    }
  }
}

void MultiGradientMachine::updateThreadParameters() {
  for (size_t pid = 0; pid < parameters_.size(); ++pid) {
    if (!parameters_[pid]->useGpu()) continue;
    if (!parameters_[pid]->isValueUpdated()) continue;
    parameters_[pid]->clearValueUpdated();
    for (int i = 0; i < (int)threads_.size(); i++) {
      threads_[i]->incUpdateCounter();
    }
    // NotifyValueReady should happen after that all threads' incUpdateCounter()
    // are called so that the counters are correct when notifyValueReady()
    // is called.
    threads_[paraMainThread_[pid]]->notifyValueReady(pid);
  }
}

void MultiGradientMachine::onPassEnd() {
  for (auto& thread : threads_) {
    thread->onPassEnd();
  }
}

Evaluator* MultiGradientMachine::makeEvaluator() const {
  return threads_[0]->getGradientMachine()->makeEvaluator();
}

void MultiGradientMachine::eval(Evaluator* evaluator) const {
  for (auto& thread : threads_) {
    SetDevice device(thread->getDeviceId());
    if (thread->hasInputData()) {
      thread->getGradientMachine()->eval(evaluator);
    }
  }
}

void MultiGradientMachine::getOutArgs(std::vector<Argument>* outArgs,
                                      PassType passType) {
  for (auto& thread : threads_) {
    REGISTER_TIMER("waitOutArgs");
    thread->waitOutArgsReady();
  }

  outArgs_.resize(threads_[threads_.size() - 1]->getOutArgs().size());

  REGISTER_TIMER("copyOutArgs");
  for (size_t i = 0; i < outArgs_.size(); ++i) {
    std::vector<Argument> args;
    args.reserve(threads_.size());
    for (auto& thread : threads_) {
      // If the thread input is empty, then the output is empty.
      auto tmp = thread->getOutArgs();
      if (tmp.size() > 0) {
        args.push_back(tmp[i]);
      }
    }
    outArgs_[i].concat(args, useGpu_, outArgStream_, passType);
  }

  if (useGpu_) {
    hl_stream_synchronize(outArgStream_);
  }

  *outArgs = outArgs_;
}

void MultiGradientMachine::setOutputGrad(const std::vector<Argument>& args) {
  CHECK_EQ(args.size(), outArgs_.size());
  for (size_t i = 0; i < args.size(); i++) {
    outArgs_[i].grad = args[i].grad;
  }
}

void MultiGradientMachine::startTask(TaskType taskType) {
  taskType_ = taskType;
  for (auto& thread : threads_) {
    thread->notifyTaskReady();
  }
}

TrainerThread::TrainerThread(const ModelConfig& config,
                             int threadId,
                             MultiGradientMachine* multiMachine)
    : multiMachine_(multiMachine),
      config_(config),
      threadId_(threadId),
      inArgsCopied_(false) {
  int numThreads = multiMachine->getNumThreads();

  auto& mainParas = multiMachine->getParameters();

  using std::placeholders::_1;
  using std::placeholders::_2;

  partnerId_ = mod(threadId_ - 1, numThreads);

  deviceId_ = !multiMachine_->useGpu()
                  ? -1
                  : multiMachine_->logicalDeviceId2RealDeviceId(0, threadId_);
  SetDevice gpuDevice(deviceId_);

  NeuralNetwork* nn = nullptr;
  if (!multiMachine->useGpu() || !FLAGS_parallel_nn) {
    nn = NeuralNetwork::create(config);
  } else {
    nn = new ParallelNeuralNetwork();
    for (auto& paraConfig : *config_.mutable_parameters()) {
      if (paraConfig.device() != -1) {
        paraConfig.set_device(multiMachine_->logicalDeviceId2RealDeviceId(
            paraConfig.device(), threadId_));
      }
    }
    for (auto& layerConfig : *config_.mutable_layers()) {
      if (layerConfig.device() != -1) {
        layerConfig.set_device(multiMachine_->logicalDeviceId2RealDeviceId(
            layerConfig.device(), threadId_));
      }
    }
  }
  // Only GPU do not share parameter values with main paramters.
  ParamInitCallback slaveParamInitCb =
      std::bind(parameterInitNN, _1, _2, &mainParas);
  nn->init(config_, slaveParamInitCb);
  gradientMachine_.reset(nn);
  parameters_ = gradientMachine_->getParameters();
  if (!FLAGS_parallel_nn) {
    for (auto& para : parameters_) {
      para->setDevice(deviceId_);
    }
  }

  backwardCallback_ =
      std::bind(&TrainerThread::backwardCallback, this, std::placeholders::_1);

  gradStream_ = HPPL_STREAM_2;
  valueStream_ = HPPL_STREAM_3;
  stopping_ = true;
  updateCounter_ = 0;
  parameterUpdated_ = false;
}

TrainerThread::~TrainerThread() { stop(); }

void TrainerThread::start() {
  if (!stopping_) return;

  stopping_ = false;

  gradientMachine_->start();

  computeThread_.reset(new std::thread([this]() { computeThread(); }));

  if (multiMachine_->useGpu()) {
    gradCollectThread_.reset(
        new std::thread([this]() { gradCollectThread(); }));

    valueDispatchThread_.reset(
        new std::thread([this]() { valueDispatchThread(); }));

    copyThread_.reset(new std::thread([this]() { copyGradToBufferThread(); }));
  }
}

void TrainerThread::stop() {
  if (stopping_) return;

  stopping_ = true;

  if (computeThread_) {
    taskReadySem_.post();
    computeThread_->join();
  }
  if (gradCollectThread_) {
    gradQueue_.enqueue(0);
    gradCollectThread_->join();
  }
  if (copyThread_) {
    gradBufQueue_.enqueue(0);
    copyThread_->join();
  }
  if (valueDispatchThread_) {
    valueReadyQueue_.enqueue(0);
    valueDispatchThread_->join();
  }
}

void TrainerThread::computeThread() {
  VLOG(1) << "gradComputeThread " << threadId_;

  if (deviceId_ >= 0) {
    hl_init(deviceId_);
  }

  while (true) {
    {
      REGISTER_TIMER("taskSem_wait");
      taskReadySem_.wait();
    }

    if (stopping_) break;

    switch (multiMachine_->getTaskType()) {
      case MultiGradientMachine::TASK_FORWARD_BACKWARD:
        forward();
        backward();
        break;
      case MultiGradientMachine::TASK_FORWARD:
        forward();
        break;
      case MultiGradientMachine::TASK_BACKWARD:
        backward();
        break;
      case MultiGradientMachine::TASK_COPY_IN_ARGS:
        batchSize_ = copyInArgs();
        inArgsCopied_ = true;
        multiMachine_->waitForCopyInArgs();
        break;
    }
  }
}

void TrainerThread::prefetch() {
  SetDevice setDevice(deviceId_);
  gradientMachine_->prefetch(inArgs_);
}

void TrainerThread::forward() {
  if (!inArgsCopied_) {
    REGISTER_TIMER("copyInArgs");
    batchSize_ = copyInArgs();
  } else {
    inArgsCopied_ = false;
  }

  if (multiMachine_->getPassType() != PASS_TEST) {
    REGISTER_TIMER("clearGradient");
    // For main parameter, the user of MultiGpuSyncMachine is responsible
    // for setting the gradient to zero
    for (size_t i = 0; i < parameters_.size(); i++) {
      if (parameters_[i]->useGpu()) {
        if (multiMachine_->paraMainThread(i) != threadId_) {
          SetDevice device(parameters_[i]->getDeviceId());
          parameters_[i]->clearGradient();
        }
      } else {
        parameters_[i]->clearGradient();
      }
    }
  }

  {
    REGISTER_TIMER("wait_value");
    valueReadyCond_.wait([this]() { return !parameterUpdated_; });
  }

  { fillMergeTypes(multiMachine_->getPassType(), &mergeTypes_); }

  {
    REGISTER_TIMER("thread_forward");
    if (batchSize_ > 0) {
      gradientMachine_->forward(
          inArgs_, &outArgs_, multiMachine_->getPassType());
    } else {
      outArgs_.clear();
    }
  }
  outArgsReadySem_.post();
}

void TrainerThread::backward() {
  REGISTER_TIMER("thread_backward");
  if (multiMachine_->isPassGrad()) {
    copyOutputGrad();
  }
  if (batchSize_ > 0) {
    gradientMachine_->backward(backwardCallback_);
  } else {
    for (size_t i = parameters_.size(); i > 0; i--) {
      backwardCallback(parameters_[i - 1].get());
    }
  }
  if (multiMachine_->hasNonstaticCpuParamters()) {
    mergeCpuGradients();
  }
}

void TrainerThread::backwardCallback(Parameter* para) {
  // CPU parameters are merged in the end
  if (!para->useGpu() || para->isStatic()) return;

  int paramId = para->getID();
  if (multiMachine_->getNumThreads() == 1) {
    // no need to do merge if there is only one thread
    doCallback(paramId);
  } else if (threadId_ == mod(multiMachine_->paraMainThread(paramId) - 1,
                              multiMachine_->getNumThreads())) {
    notifyCopyGradToBuffer(paramId);
  } else {
    notifyGradientCollect(paramId);
  }
}

void TrainerThread::copyGradToBufferThread() {
  VLOG(1) << "copyGradToBufferThread " << threadId_;

  if (deviceId_ >= 0) {
    hl_init(deviceId_);
  }
  auto& partnerThread = multiMachine_->getThread(partnerId_);
  auto& gradBufs = multiMachine_->getGradBuf(partnerId_);

  while (true) {
    int pid = gradBufQueue_.dequeue();
    if (stopping_) break;

    int pdeviceId = multiMachine_->realDeviceId2LogicalDeviceId(
        parameters_[pid]->getDeviceId(), threadId_);

    auto& gradBuf = gradBufs[pdeviceId];

    {
      REGISTER_TIMER("waitBufferReady");
      gradBuf.sem.wait();
    }

    {
      REGISTER_TIMER("copyGradToBuffer");
      SetDevice setDevice(parameters_[pid]->getDeviceId());
      for (size_t i = 0; i < mergeTypes_.size(); ++i) {
        gradBuf.bufs[i]->resize(
            parameters_[pid]->getBuf(mergeTypes_[i])->getSize());
        gradBuf.bufs[i]->copyFrom(*parameters_[pid]->getBuf(mergeTypes_[i]),
                                  gradStream_);
      }
      hl_stream_synchronize(gradStream_);
    }
    partnerThread->notifyGradientCollect(pid);
  }
}

void TrainerThread::gradCollectThread() {
  VLOG(1) << "gradCollectThread " << threadId_;

  if (deviceId_ >= 0) {
    hl_init(deviceId_);
  }

  std::vector<size_t> gradReadyCount(parameters_.size(), 0);

  auto& gradBufs = multiMachine_->getGradBuf(threadId_);

  while (true) {
    int pid = gradQueue_.dequeue();
    if (stopping_) break;

    if (++gradReadyCount[pid] < 2) continue;
    gradReadyCount[pid] = 0;
    int pdeviceId = multiMachine_->realDeviceId2LogicalDeviceId(
        parameters_[pid]->getDeviceId(), threadId_);

    auto& gradBuf = gradBufs[pdeviceId];

    {
      REGISTER_TIMER("mergeGrad");
      for (size_t i = 0; i < mergeTypes_.size(); ++i) {
        ParameterType type = mergeTypes_[i];
        const VectorPtr& localGrad = parameters_[pid]->getBuf(type);
        SetDevice setDevice(parameters_[pid]->getDeviceId());
        localGrad->add(*gradBuf.bufs[i]);
      }
    }

    gradBuf.sem.post();

    if (multiMachine_->paraMainThread(pid) == threadId_) {
      doCallback(pid);
    } else {
      notifyCopyGradToBuffer(pid);
    }
  }
}

void TrainerThread::doCallback(int pid) {
  REGISTER_TIMER("callback");
  auto& gpuThreads = multiMachine_->getAllThreads();
  if (multiMachine_->getBackwardCallback()) {
    // The callback supplied by the user of MultiGradientMachine may handle
    // the parameter update using the gradient.
    multiMachine_->getBackwardCallback()(parameters_[pid].get());
    if (parameters_[pid]->isValueUpdated()) {
      parameters_[pid]->clearValueUpdated();
      for (auto& thread : gpuThreads) {
        thread->incUpdateCounter();
      }
      notifyValueReady(pid);
    }
  }
  multiMachine_->notifyGradientTransfer(pid);
}

void TrainerThread::valueDispatchThread() {
  VLOG(1) << "valueDispatchThread " << threadId_;

  if (deviceId_ >= 0) {
    hl_init(deviceId_);
  }

  auto& thread = multiMachine_->getThread(partnerId_);

  while (true) {
    int pid;
    {
      REGISTER_TIMER("value_dequeue");
      pid = valueReadyQueue_.dequeue();
    }
    if (stopping_) break;

    if (multiMachine_->paraMainThread(pid) == partnerId_) continue;

    {
      REGISTER_TIMER("copyValue");
      SetDevice setDevice(parameters_[pid]->getDeviceId());
      thread->getValueBuf(pid)->copyFrom(*getValueBuf(pid), valueStream_);
      hl_stream_synchronize(valueStream_);
    }

    thread->notifyValueReady(pid);
  }
}

void TrainerThread::notifyValueReady(int paramId) {
  if (--updateCounter_ == 0) {
    valueReadyCond_.notify_all([this] { parameterUpdated_ = false; });
  }

  notifyValueDispatch(paramId);
}

int TrainerThread::copyInArgs() {
  const std::vector<Argument>& fullInArgs = multiMachine_->getInArgs();
  int numThreads = multiMachine_->getAllThreads().size();
  int32_t numSequences = fullInArgs[0].getNumSequences();
  int32_t startSeq = numSequences * threadId_ / numThreads;
  int32_t endSeq = numSequences * (threadId_ + 1) / numThreads;
  int32_t copySize = endSeq - startSeq;

  /**
   * For the first copy, need to allocate space here
   */
  if (inArgs_.size() == 0) {
    inArgs_.resize(fullInArgs.size());
  }

  if (copySize == 0) {
    return 0;
  }

  for (size_t i = 0; i < fullInArgs.size(); i++) {
    inArgs_[i].resizeAndCopyFrom(
        fullInArgs[i],
        startSeq,
        copySize,
        FLAGS_parallel_nn ? false : multiMachine_->useGpu());
  }
  return copySize;
}

void TrainerThread::mergeCpuGradients() {
  CHECK_EQ(mergeTypes_.size(), 1UL);
  CHECK_EQ(mergeTypes_[0], PARAMETER_GRADIENT);

  {
    REGISTER_TIMER("waitbeforeMerge");
    multiMachine_->waitBeforeMerge();
  }
  std::vector<const std::vector<ParameterPtr>*> slaveParameters =
      multiMachine_->getSlaveParameters();

  CHECK(slaveParameters.size());
  for (auto& para : multiMachine_->getNonStaticParameters()) {
    if (para->useGpu()) continue;
    if (para->isSparseRemoteUpdate()) {
      REGISTER_TIMER("mergeRemoteGradSparse");
      mergeGradSparseRemote(para.get(), slaveParameters);
    } else if (para->isGradSparseUpdate()) {
      REGISTER_TIMER("mergeGradSparse");
      mergeGradSparse(para.get(), slaveParameters);
    } else {
      REGISTER_TIMER("mergeGradDense");
      mergeGradDense(para.get(), slaveParameters);
    }
  }
  {
    REGISTER_TIMER("waitbeforeMerge");
    multiMachine_->waitAfterMerge();
  }
}

void TrainerThread::mergeGradSparse(
    Parameter* para,
    std::vector<const std::vector<ParameterPtr>*>& slaveParameters) {
  size_t pid = para->getID();
  SparseRowIdsCpuMatrix* mainMat = dynamic_cast<SparseRowIdsCpuMatrix*>(
      para->getMat(PARAMETER_GRADIENT).get());
  std::vector<uint32_t>& ids = mainMat->getIds(threadId_);

  for (auto slaveParams : slaveParameters) {
    SparseRowCpuMatrix* mat = dynamic_cast<SparseRowCpuMatrix*>(
        (*slaveParams)[pid]->getMat(PARAMETER_GRADIENT).get());
    mat->addTo(*mainMat, ids, threadId_, multiMachine_->getNumThreads());
    // we use a sample hash method(%) instead of range partition,
    // because range partition has balance issue sometimes,
    // when feature ids are not generated from hashcode.
  }
  uniqueIds(ids);
}

void TrainerThread::mergeGradSparseRemote(
    Parameter* para,
    std::vector<const std::vector<ParameterPtr>*>& slaveParameters) {
  size_t pid = para->getID();
  SparseRowCpuMatrix* mainMat =
      dynamic_cast<SparseRowCpuMatrix*>(para->getMat(PARAMETER_GRADIENT).get());

  mainMat->checkIndices();
  mainMat->zeroMemThread(threadId_, multiMachine_->getNumThreads());

  for (auto slaveParams : slaveParameters) {
    SparseRowCpuMatrix* mat = dynamic_cast<SparseRowCpuMatrix*>(
        (*slaveParams)[pid]->getMat(PARAMETER_GRADIENT).get());
    mat->addTo(*mainMat, threadId_, multiMachine_->getNumThreads());
  }
}

void TrainerThread::mergeGradDense(
    Parameter* para,
    std::vector<const std::vector<ParameterPtr>*>& slaveParameters) {
  size_t pid = para->getID();
  auto interval = calcSplitArrayInterval(para->getSize(),
                                         (size_t)threadId_,
                                         multiMachine_->getNumThreads(),
                                         8LU /*for avx*/);
  size_t startSeq = interval.first;
  size_t copySize = interval.second - interval.first;

  // setup sub bufs
  CpuVector destGrad(0, nullptr);
  destGrad.subVecFrom(*para->getBuf(PARAMETER_GRADIENT), startSeq, copySize);

  // merge
  CpuVector slaveGradSub(0, nullptr);
  for (auto slaveParams : slaveParameters) {
    slaveGradSub.subVecFrom(
        *(*slaveParams)[pid]->getBuf(PARAMETER_GRADIENT), startSeq, copySize);
    destGrad.add(slaveGradSub);
  }
}

void TrainerThread::copyOutputGrad() {
  const std::vector<Argument>& outputGradArgs = multiMachine_->outArgs_;
  int numThreads = multiMachine_->getAllThreads().size();
  int32_t numSequences = outputGradArgs[0].getNumSequences();
  int32_t startSeq = numSequences * threadId_ / numThreads;
  int32_t endSeq = numSequences * (threadId_ + 1) / numThreads;
  int32_t copySize = endSeq - startSeq;
  outArgs_.resize(outputGradArgs.size());
  for (size_t i = 0; i < outputGradArgs.size(); i++) {
    outArgs_[i].resizeAndCopyFrom(outputGradArgs[i],
                                  startSeq,
                                  copySize,
                                  multiMachine_->useGpu(),
                                  HPPL_STREAM_DEFAULT);
  }
  if (multiMachine_->useGpu()) {
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  }
  gradientMachine_->setOutputGrad(outArgs_);
}
}  // namespace paddle
