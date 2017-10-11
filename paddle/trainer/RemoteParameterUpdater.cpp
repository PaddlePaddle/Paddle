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

#include "RemoteParameterUpdater.h"
#include "Trainer.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/Stat.h"

DECLARE_int32(trainer_id);
DECLARE_string(save_dir);

namespace paddle {

static const hl_stream_t kDeviceToHostStream = HPPL_STREAM_1;
static const hl_stream_t kHostToDeviceStream = HPPL_STREAM_2;
static const int kFinishBatchPid = -1;

const std::string RemoteParameterUpdater::kAverage = "average";
const std::string RemoteParameterUpdater::kElasticAverage = "elastic_average";

RemoteParameterUpdater::RemoteParameterUpdater(
    const OptimizationConfig& config,
    int expectedPassCount,
    std::unique_ptr<ParameterUpdater>&& localUpdater)
    : config_(config),
      localUpdater_(std::move(localUpdater)),
      numBatches_(0),
      passCount_(0),
      expectedPassCount_(expectedPassCount),
      separateSendAndRecv_(false),
      isFirstPass_(true),
      useApplyInPserver_(false) {
  addParameterType(PARAMETER_MOMENTUM);
}

void RemoteParameterUpdater::init(const std::vector<ParameterPtr>& parameters) {
  ParameterUpdater::init(parameters);

  if (localUpdater_) {
    localUpdater_->init(parameters);

    for (auto& parameter : parameters) {
      parameter->enableType(PARAMETER_DELTA);
    }

    CHECK(config_.center_parameter_update_method() == kAverage ||
          config_.center_parameter_update_method() == kElasticAverage)
        << "unknown center_parameter_update_method";

    // modify delta_add_rate
    CHECK_GT(FLAGS_num_gradient_servers, 1)
        << "FLAGS_num_gradient_servers should be set in trainer args.";
    real delta_add_rate = config_.delta_add_rate() / FLAGS_num_gradient_servers;
    config_.set_delta_add_rate(delta_add_rate);
    LOG(INFO) << "center parameter in pserver,"
              << " modify delta_add_rate=" << delta_add_rate;
  }

  if (!FLAGS_use_gpu) {
    cpuParameters_ = parameters;
  } else {
    for (auto& parameter : parameters) {
      cpuParameters_.emplace_back(new Parameter(parameter->getConfig(),
                                                /* useGpu= */ false));
      cpuParameters_.back()->setID(parameter->getID());
      if (localUpdater_) {
        cpuParameters_.back()->enableType(PARAMETER_DELTA);
      }
    }
  }

  parameterClient_.reset(new ParameterClient2(separateSendAndRecv_));
  parameterClient_->init(cpuParameters_);
  parameterClient_->setTrainerId(FLAGS_trainer_id);

  if (FLAGS_trainer_id == 0) {
    parameterClient_->setConfig(config_);
    copyParametersFromDevice(PARAMETER_VALUE);
    parameterClient_->setParameter();
    parameterClient_->setStatus(PSERVER_STATUS_PARAMETER_READY);
  } else {
    parameterClient_->waitForStatus(PSERVER_STATUS_PARAMETER_READY);
    parameterClient_->getParameter();
    copyParametersToDevice(PARAMETER_VALUE);
  }
  if (FLAGS_trainer_id == 0 &&
      (config_.algorithm() != TrainAlgorithm::AsyncSGD)) {
    startController();
    useApplyInPserver_ = useApplyInPserver(config_);
  }
}

void RemoteParameterUpdater::startController() {
  controllerThread_.reset(new std::thread([this]() { this->controller(); }));
}

void RemoteParameterUpdater::controller() {
  ParameterClient2 client(false);
  client.init(cpuParameters_);
  while (true) {
    /*start pass*/ {
      client.waitPassStart();

      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_START_PASS);
      client.doOperation(ops,
                         /* waitForGradient= */ false,
                         /* sendBackarameter= */ false,
                         /* releasePass= */ false);
    }

    while (true) {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_SGD);
      client.doOperation(ops,
                         /* waitForGradient= */ true,
                         /* sendBackarameter= */ true,
                         /* releasePass= */ false);
      if (client.isPassFinish()) {
        break;
      }
    }

    /*finish pass*/ {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_FINISH_PASS);
      client.doOperation(ops,
                         /* waitForGradient= */ true,
                         /* sendBackarameter= */ true,
                         /* releasePass= */ true);
    }

    passCount_++;
    if (passCount_ == expectedPassCount_) {
      break;
    }
  }
}

void RemoteParameterUpdater::copyParametersToDevice(
    ParameterType parameterType) {
  if (!FLAGS_use_gpu) {
    return;
  }
  int numParameters = cpuParameters_.size();
  for (int i = 0; i < numParameters; ++i) {
    parameters_[i]
        ->getBuf(parameterType)
        ->copyFrom(*cpuParameters_[i]->getBuf(parameterType));
    if (parameterType == PARAMETER_VALUE) {
      parameters_[i]->setValueUpdated();
    }
  }
}

void RemoteParameterUpdater::copyParametersFromDevice(
    ParameterType parameterType) {
  if (!FLAGS_use_gpu) {
    return;
  }
  int numParameters = cpuParameters_.size();
  for (int i = 0; i < numParameters; ++i) {
    cpuParameters_[i]
        ->getBuf(parameterType)
        ->copyFrom(*parameters_[i]->getBuf(parameterType));
  }
}

void RemoteParameterUpdater::updateImpl(Parameter* para) {
  REGISTER_TIMER("update");
  if (localUpdater_) {
    localUpdater_->update(para);
  }
}

void RemoteParameterUpdater::finishBatch(real cost) {
  if (localUpdater_) {
    localUpdater_->finishBatch(cost);
  }

  const std::string& algorithm = config_.algorithm();
  ParameterUpdateMode mode;
  if (algorithm == TrainAlgorithm::AsyncSGD) {
    mode = PSERVER_UPDATE_MODE_ASYNC_SGD;
  } else if (algorithm == TrainAlgorithm::SGD) {
    mode = PSERVER_UPDATE_MODE_ADD_GRADIENT;
  } else {
    LOG(FATAL) << "Unknown algorithm: " << algorithm;
  }

  ParameterType sendType;
  bool sendBackParameter = true;
  if (localUpdater_) {
    ++numBatches_;
    if (numBatches_ % config_.num_batches_per_send_parameter() != 0) {
      return;
    }

    if (config_.center_parameter_update_method() == kElasticAverage) {
      parameterClient_->getParameter(PARAMETER_DELTA);
      copyParametersToDevice(PARAMETER_DELTA);
      sendBackParameter = false;  // no need send back after send

      // calc delta
      for (auto& para : parameters_) {
        // DELTA = LOCAL_VALUE - CENTER_VALUE/*store in DELTA*/
        para->getBuf(PARAMETER_DELTA)
            ->add(*para->getBuf(PARAMETER_VALUE), -1.0f, 1.0f);

        // when delta send to pserver, pserver will do:
        // CENTER_VALUE += alpha * (LOCAL_VALUE - CENTER_VALUE)
      }
    } else {
      // calc delta
      for (auto& para : parameters_) {
        // DELTA = NEW_VALUE - OLD_VALUE/*store in DELTA*/
        para->getBuf(PARAMETER_DELTA)
            ->add(*para->getBuf(PARAMETER_VALUE), -1.0f, 1.0f);
      }
    }

    sendType = PARAMETER_DELTA;

  } else {
    // In this case, we perform SGD on pserver.
    sendType = PARAMETER_GRADIENT;
  }

  copyParametersFromDevice(sendType);

  {
    REGISTER_TIMER("sendAndRecv_dense");
    parameterClient_->sendAndReceiveParameter(mode,
                                              sendType,
                                              batchSize_,
                                              0,  // cost = 0
                                              sendBackParameter);
  }

  if (sendBackParameter) {
    copyParametersToDevice(PARAMETER_VALUE);
  }

  if (localUpdater_) {
    if (config_.center_parameter_update_method() == kElasticAverage) {
      for (auto& para : parameters_) {
        SetDevice device(para->getDeviceId());
        // LOCAL_VALUE += -alpha * (LOCAL_VALUE - CENTER_VALUE)
        para->getBuf(PARAMETER_VALUE)
            ->add(*para->getBuf(PARAMETER_DELTA), -config_.delta_add_rate());
      }

    } else {  // average
      // copy value to delta
      for (auto& para : parameters_) {
        SetDevice device(para->getDeviceId());
        para->getBuf(PARAMETER_DELTA)->copyFrom(*para->getBuf(PARAMETER_VALUE));
      }
    }
  } else {
    for (auto& para : parameters_) {
      SetDevice device(para->getDeviceId());
      para->getBuf(sendType)->zeroMem();
    }
  }
}

void RemoteParameterUpdater::startPass() {
  if (config_.algorithm() == TrainAlgorithm::SGD) {
    parameterClient_->waitPassStart();
  } else {
    // sync could benifits reducing lagged trainer for async-sgd
    // even if sync could not remove all lagged trainer for the
    // sake of file loading, buffer etc.
    parameterClient_->asyncStartPass();
  }

  if (localUpdater_) {
    localUpdater_->startPass();
    numBatches_ = 0;

    if (config_.center_parameter_update_method() == kElasticAverage) {
      if (!isFirstPass_) {
        // restore local value from delta
        for (auto& para : parameters_) {
          SetDevice device(para->getDeviceId());
          para->getBuf(PARAMETER_VALUE)
              ->copyFrom(*para->getBuf(PARAMETER_DELTA));
        }
      }
    } else {  // average
      // copy value to delta
      for (auto& para : parameters_) {
        SetDevice device(para->getDeviceId());
        para->getBuf(PARAMETER_DELTA)->copyFrom(*para->getBuf(PARAMETER_VALUE));
      }
    }
  }
}

bool RemoteParameterUpdater::finishPass() {
  if (localUpdater_) {
    localUpdater_->finishPass();
  }

  if (config_.algorithm() == TrainAlgorithm::SGD) {
    parameterClient_->waitPassFinish();
  } else {
    parameterClient_->asyncFinishPass();
  }
  if (localUpdater_) {
    if (config_.center_parameter_update_method() == kElasticAverage) {
      // backup local value to delta as we will get
      // the remote parameter for saving/testing
      for (auto& para : parameters_) {
        SetDevice device(para->getDeviceId());
        para->getBuf(PARAMETER_DELTA)->copyFrom(*para->getBuf(PARAMETER_VALUE));
      }
    }
  }
  parameterClient_->getParameter();
  copyParametersToDevice(PARAMETER_VALUE);

  isFirstPass_ = false;
  return true;
}

void RemoteParameterUpdater::apply() {
  if (useApplyInPserver_) {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_APPLY);
    parameterClient_->doOperation(ops,
                                  /* waitForGradient= */ false,
                                  /* sendBackarameter= */ false);
    parameterClient_->getParameter(
        /* recvParameterType= */ PARAMETER_VALUE,
        /* sendBackParameterType= */ PARAMETER_APPLY);
    copyParametersToDevice(PARAMETER_VALUE);
  }
}

void RemoteParameterUpdater::restore() {
  if (useApplyInPserver_) {
    parameterClient_->getParameter();
    copyParametersToDevice(PARAMETER_VALUE);
  }
}

ConcurrentRemoteParameterUpdater::ConcurrentRemoteParameterUpdater(
    OptimizationConfig config,
    int passCount,
    std::unique_ptr<ParameterUpdater>&& localUpdater)
    : RemoteParameterUpdater(config, passCount, std::move(localUpdater)) {
  sendThread_.reset(new std::thread([this]() { this->send(); }));
  recvThread_.reset(new std::thread([this]() { this->recv(); }));

  stopping_ = false;
  oneBatchFinished_ = false;
  separateSendAndRecv_ = true;
}

ConcurrentRemoteParameterUpdater::~ConcurrentRemoteParameterUpdater() {
  stopping_ = true;
  sendQueue_.enqueue(0);
  sendThread_->join();
  recvQueue_.enqueue(0);
  recvThread_->join();
}

void ConcurrentRemoteParameterUpdater::finishBatch(real cost) {
  if (localUpdater_) {
    localUpdater_->finishBatch(cost);

    if (!needToUpdateRemotely()) {
      ++numBatches_;
      return;
    }
  }

  sendQueue_.enqueue(kFinishBatchPid);

  finishBatchCond_.wait([this]() { return oneBatchFinished_; });
  oneBatchFinished_ = false;
  {
    REGISTER_TIMER("sync_hostToDeviceStream");
    for (auto& para : parameters_) {
      SetDevice device(para->getDeviceId());
      hl_stream_synchronize(kHostToDeviceStream);
    }
  }

  if (localUpdater_) {
    ++numBatches_;
  }
}

// Use para=NULL to signal the end of one batch
void ConcurrentRemoteParameterUpdater::send(Parameter* para) {
  const std::string& algorithm = config_.algorithm();
  ParameterUpdateMode mode;
  if (algorithm == TrainAlgorithm::AsyncSGD) {
    mode = PSERVER_UPDATE_MODE_ASYNC_SGD;
  } else if (algorithm == TrainAlgorithm::SGD) {
    mode = PSERVER_UPDATE_MODE_ADD_GRADIENT;
  } else {
    LOG(FATAL) << "Unknown algorithm: " << algorithm;
  }
  ParameterType sendType;
  if (localUpdater_) {
    sendType = PARAMETER_DELTA;
  } else {
    // In this case, we perform SGD on pserver.
    sendType = PARAMETER_GRADIENT;
  }
  std::vector<ParameterSegments> paraSegment;
  if (para == NULL) {
    parameterClient_->sendParameter(
        mode,
        sendType,
        paraSegment,
        batchSize_,
        0,              // cost=0
        true,           // sendBackParameter = true
        batchStatus_);  // batchStatus_ = BATCH_FINISH

  } else {
    ParameterSegments paraSegTemp;
    paraSegment.reserve(1);
    paraSegTemp.name = para->getName();
    paraSegTemp.id = para->getID();
    paraSegment.push_back(paraSegTemp);
    {
      SetDevice device(para->getDeviceId());
      REGISTER_TIMER("copySingleParaFromDevice");
      copySingleParaFromDevice(para, sendType);
      hl_stream_synchronize(kDeviceToHostStream);
    }
    parameterClient_->sendParameter(mode,
                                    sendType,
                                    paraSegment,
                                    batchSize_,
                                    0,     // cost=0
                                    true,  // sendBackParameter = true
                                    batchStatus_);
    if (batchStatus_ == BATCH_START) batchStatus_ = BATCH_ON;
  }
}
void ConcurrentRemoteParameterUpdater::recv(Parameter* para) {
  parameterClient_->recvParameter();
  if (para != NULL) {
    REGISTER_TIMER("copySingleParaToDevice");
    SetDevice device(para->getDeviceId());
    copySingleParaToDevice(para, PARAMETER_VALUE);

    if (localUpdater_) {
      para->getBuf(PARAMETER_DELTA)->copyFrom(*para->getBuf(PARAMETER_VALUE));
    } else {
      // if cpu, parameter should not changes until recvParameter().
      // if gpu, zero mem when send finish
      if (!FLAGS_use_gpu) {
        para->getBuf(PARAMETER_GRADIENT)->zeroMem();
      }
    }
  }
}

void ConcurrentRemoteParameterUpdater::recv() {
  if (FLAGS_use_gpu) hl_set_device(FLAGS_gpu_id);
  StatPtr stat = getStat("recv");
  FOR_TIMING(Timer timer);
  while (true) {
    int pid;
    {
      REGISTER_TIMER("recv_dequeue");
      pid = recvQueue_.dequeue();
    }
    if (pid == kFinishBatchPid) {
      Parameter* para = NULL;
      FOR_TIMING(timer.start());
      recv(para);
      FOR_TIMING(timer.stop());
      FOR_TIMING(stat->addSample(timer.get()));
      FOR_TIMING(timer.reset());
      finishBatchCond_.notify_all([this] { oneBatchFinished_ = true; });
    } else {
      if (stopping_) break;
      Parameter* para = parameters_[pid].get();
      FOR_TIMING(timer.start());
      recv(para);
      FOR_TIMING(timer.stop());
      oneBatchFinished_ = false;
    }
  }
}

void ConcurrentRemoteParameterUpdater::send() {
  if (FLAGS_use_gpu) hl_set_device(FLAGS_gpu_id);
  StatPtr stat = getStat("send");
  FOR_TIMING(Timer timer);
  while (true) {
    int pid;
    {
      REGISTER_TIMER("send_dequeue");
      pid = sendQueue_.dequeue();
    }
    if (pid == kFinishBatchPid) {
      batchStatus_ = BATCH_FINISH;
      if (!localUpdater_) {
        // if cpu, parameter should not changes until recvParameter().
        // if gpu, zeroMem() at the end of batch so that it won't
        // interfere with computation.
        if (FLAGS_use_gpu) {
          REGISTER_TIMER("para_zeroMem");
          for (auto& para : parameters_) {
            SetDevice device(para->getDeviceId());
            para->getBuf(PARAMETER_GRADIENT)->zeroMem();
          }
        }
      }
      Parameter* para = NULL;
      FOR_TIMING(timer.start());
      send(para);
      FOR_TIMING(timer.stop());
      FOR_TIMING(stat->addSample(timer.get()));
      FOR_TIMING(timer.reset());
      recvQueue_.enqueue(pid);
    } else {
      if (stopping_) break;
      Parameter* para = parameters_[pid].get();
      if (localUpdater_) {
        // DELTA = NEW_VALUE - OLD_VALUE/*store in DELTA*/
        para->getBuf(PARAMETER_DELTA)
            ->add(*para->getBuf(PARAMETER_VALUE), -1.0f, 1.0f);
      }
      FOR_TIMING(timer.start());
      send(para);
      FOR_TIMING(timer.stop());
      recvQueue_.enqueue(nonStaticParaIDMap_[para->getID()]);
    }
  }
}

void ConcurrentRemoteParameterUpdater::updateImpl(Parameter* para) {
  REGISTER_TIMER("update");
  if (localUpdater_) {
    localUpdater_->update(para);
    if (!needToUpdateRemotely()) {
      return;
    }
  }
  sendQueue_.enqueue(nonStaticParaIDMap_[para->getID()]);
}

void ConcurrentRemoteParameterUpdater::copySingleParaToDevice(
    Parameter* para, ParameterType parameterType) {
  if (!FLAGS_use_gpu) {
    return;
  }
  int i = nonStaticParaIDMap_[para->getID()];
  para->getBuf(parameterType)
      ->copyFrom(*cpuParameters_[i]->getBuf(parameterType),
                 kHostToDeviceStream);
  if (parameterType == PARAMETER_VALUE) {
    para->setValueUpdated();
  }
}

void ConcurrentRemoteParameterUpdater::copySingleParaFromDevice(
    Parameter* para, ParameterType parameterType) {
  if (!FLAGS_use_gpu) {
    return;
  }
  int i = nonStaticParaIDMap_[para->getID()];
  cpuParameters_[i]
      ->getBuf(parameterType)
      ->copyFrom(*para->getBuf(parameterType), kDeviceToHostStream);
}

SparseRemoteParameterUpdater::SparseRemoteParameterUpdater(
    const OptimizationConfig& config, int expectedPassCount, bool testing)
    : config_(config),
      passCount_(0),
      expectedPassCount_(expectedPassCount),
      testing_(testing),
      useApplyInPserver_(false) {}

void SparseRemoteParameterUpdater::init(
    const std::vector<ParameterPtr>& parameters) {
  ParameterUpdater::init(parameters);

  parameterClient_.reset(new ParameterClient2(
      false, FLAGS_port + FLAGS_ports_num, FLAGS_ports_num_for_sparse));
  parameterClient_->init(parameters_);
  parameterClient_->setTrainerId(FLAGS_trainer_id);

  if (FLAGS_trainer_id == 0) {
    parameterClient_->setConfig(
        config_, FLAGS_save_dir, true /*is_sparse_server*/);
    if (parameters[0]->isFullSize()) {
      parameterClient_->setParameter();
    } else {  // init in pserver
      parameterClient_->setParameterZero();
    }
  }
  if (FLAGS_trainer_id == 0 && !testing_ &&
      config_.algorithm() == TrainAlgorithm::SGD) {
    startController();
    useApplyInPserver_ = useApplyInPserver(config_);
  }
}

void SparseRemoteParameterUpdater::startController() {
  controllerThread_.reset(new std::thread([this]() { this->controller(); }));
}

void SparseRemoteParameterUpdater::controller() {
  ParameterClient2 client(
      false, FLAGS_port + FLAGS_ports_num, FLAGS_ports_num_for_sparse);
  client.init(parameters_);

  while (true) {
    /*start pass*/ {
      client.waitPassStart();

      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_START_PASS);
      client.doOperation(ops,
                         /* waitForGradient= */ false,
                         /* sendBackarameter= */ false,
                         /* releasePass= */ false);
    }

    while (true) {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_SGD);
      client.doOperation(ops,
                         /* waitForGradient= */ true,
                         /* sendBackarameter= */ true,
                         /* releasePass= */ false);
      if (client.isPassFinish()) {
        break;
      }
    }

    /*finish pass*/ {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_FINISH_PASS);
      client.doOperation(ops,
                         /* waitForGradient= */ true,
                         /* sendBackarameter= */ true,
                         /* releasePass= */ true);
    }

    passCount_++;
    if (passCount_ == expectedPassCount_) {
      break;
    }
  }
}

PassType SparseRemoteParameterUpdater::startBatch(int64_t batchSize) {
  batchSize_ = batchSize;
  return PASS_TRAIN;
}

void SparseRemoteParameterUpdater::finishBatch(real cost) {
  const std::string& algorithm = config_.algorithm();
  ParameterUpdateMode mode;
  if (algorithm == TrainAlgorithm::AsyncSGD) {
    mode = PSERVER_UPDATE_MODE_ASYNC_SGD;
  } else if (algorithm == TrainAlgorithm::SGD) {
    mode = PSERVER_UPDATE_MODE_ADD_GRADIENT;
  } else {
    LOG(FATAL) << "Unknown algorithm: " << algorithm;
  }

  ParameterType sendType = PARAMETER_GRADIENT;

  REGISTER_TIMER("sendSparseParam");
  parameterClient_->sendAndReceiveParameter(mode,
                                            sendType,
                                            batchSize_,
                                            0,       // cost = 0
                                            false);  // sendBackParameter

  // grad zero move to sgd grad machine, before merge grad sparse remote
}

void SparseRemoteParameterUpdater::startPass() {
  if (config_.algorithm() == TrainAlgorithm::SGD) {
    parameterClient_->waitPassStart();
  } else {
    if (FLAGS_trainer_id == 0) {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_START_PASS);
      parameterClient_->doOperation(ops,
                                    /* waitForGradient= */ false,
                                    /* sendBackarameter= */ false);
    }
    parameterClient_->asyncStartPass();
  }
}

bool SparseRemoteParameterUpdater::finishPass() {
  if (config_.algorithm() == TrainAlgorithm::SGD) {
    parameterClient_->waitPassFinish();
  } else {
    if (FLAGS_trainer_id == 0) {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_FINISH_PASS);
      parameterClient_->doOperation(ops,
                                    /* waitForGradient= */ false,
                                    /* sendBackarameter= */ false);
    }
    parameterClient_->asyncFinishPass();
  }

  return true;
}

// Trainer will call getParametersRemote at batch start or before save,
// so we do not get values in apply() and restore().
void SparseRemoteParameterUpdater::apply() {
  if (useApplyInPserver_) {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_APPLY);
    parameterClient_->doOperation(ops,
                                  /* waitForGradient= */ false,
                                  /* sendBackarameter= */ false);
  }
}

void SparseRemoteParameterUpdater::restore() {}

void SparseRemoteParameterUpdater::getParametersRemote(bool fullSize,
                                                       bool apply) {
  ParameterType sendBackParameterType =
      (useApplyInPserver_ && apply) ? PARAMETER_APPLY : PARAMETER_VALUE;
  std::function<void()> getParams;
  std::function<void(Parameter&, real)> applyL1;
  if (fullSize) {
    getParams = [&] {
      parameterClient_->getParameter(
          /* recvParameterType= */ PARAMETER_VALUE, sendBackParameterType);
    };
    applyL1 = [](Parameter& para, real decayRate) {
      para.getBuf(PARAMETER_VALUE)->applyL1(/*lr=*/1.0f, decayRate);
    };
  } else {
    getParams = [&] {
      parameterClient_->getParameterSparse(
          /* recvParameterType= */ PARAMETER_VALUE, sendBackParameterType);
    };
    applyL1 = [](Parameter& para, real decayRate) {
      para.getMat(PARAMETER_VALUE)->applyL1(/*lr=*/1.0f, decayRate);
    };
  }
  {
    REGISTER_TIMER("getParamDenseAndSparse");
    getParams();
    if (config_.shrink_parameter_value() > 0) {
      for (auto& para : parameters_) {
        if (para->getConfig().decay_rate_l1() > 0) {
          applyL1(*para, config_.shrink_parameter_value());
        }
      }
    }
  }
}

void SparseRemoteParameterUpdater::randParametersRemote() {
  CHECK_EQ(FLAGS_trainer_id, 0);

  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_RANDOMIZE);
  parameterClient_->doOperation(ops,
                                /* waitForGradient= */ false,
                                /* sendBackarameter= */ false);
}

void SparseRemoteParameterUpdater::loadParametersRemote(
    const std::string& dirName) {
  if (FLAGS_trainer_id == 0) {
    parameterClient_->loadValueVector(dirName);
  }

  if (testing_) {
    // we do not use synchronize() here,
    // because test mode may run only one tester
    if (FLAGS_trainer_id == 0) {
      parameterClient_->setStatus(PSERVER_STATUS_PARAMETER_READY);
    } else {
      parameterClient_->waitForStatus(PSERVER_STATUS_PARAMETER_READY);
    }
  }
}

void SparseRemoteParameterUpdater::saveParametersRemote(
    const std::string& dirName) {
  if (FLAGS_trainer_id == 0) {
    parameterClient_->saveValueVector(dirName);
  }
}

void SparseRemoteParameterUpdaterComposite::init(
    const std::vector<ParameterPtr>& parameters) {
  parameters_ = parameters;

  std::vector<ParameterPtr> parametersArray[NUMBER_UPDATERS];

  for (auto& para : parameters_) {
    if (para->isSparseRemoteUpdate()) {
      parametersArray[UPDATER_SPARSE_REMOTE].push_back(para);
    } else {
      parametersArray[UPDATER_NORMAL].push_back(para);
    }
  }
  CHECK(!parametersArray[UPDATER_SPARSE_REMOTE].empty());
  CHECK(!parametersArray[UPDATER_NORMAL].empty());

  syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
    updaters_[tid]->init(parametersArray[tid]);
  });

  parameterTypes_ = updaters_[UPDATER_NORMAL]->getParameterTypes();
}

std::vector<std::function<ParameterUpdater*(
    const std::string&, const OptimizationConfig&, bool, size_t)>>
    ParameterUpdaterCreators::constructors_;

}  // namespace paddle
