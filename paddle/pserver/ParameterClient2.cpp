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

#include <unistd.h>

#include "ParameterClient2.h"
#include "paddle/math/SparseRowMatrix.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"

DEFINE_string(pservers, "127.0.0.1", "Comma separated addresses of pservers");
DEFINE_int32(parallel_thread_num, 1, "Thread number for parameter send");

namespace paddle {

template <typename T1, typename T2>
void copyToRepeatedField(google::protobuf::RepeatedField<T1>* dest,
                         const T2* src,
                         size_t size) {
  dest->Clear();
  dest->Reserve(size);
  for (size_t i = 0; i < size; ++i) {
    dest->AddAlreadyReserved(src[i]);
  }
}

ParameterClient2::ParameterClient2(bool separate, int port, int numPorts)
    : BaseClient(separate, numPorts), port_(port) {
#ifndef PADDLE_DISABLE_TIMER
  forwardbackwordTime_ = 0;
#endif
}

int ParameterClient2::calcParameterBlockSize(
    const std::vector<ParameterPtr>& parameters, size_t serviceNum) {
  size_t totalSize = 0;
  for (auto& para : parameters) {
    totalSize += para->getSize();
  }
  size_t perServerSize = totalSize / serviceNum;

  int sizeBits = 64 - __builtin_clzl(perServerSize);

  /// 2^10 is min block size
  /// 2^7 will be max number of blocks in one pserver
  int blockSizeBits = std::max((sizeBits - 7), 10);
  return 1 << blockSizeBits;
}

void ParameterClient2::initThreads() {
  threadNum_ = serviceNum_;
  if (FLAGS_parallel_thread_num > 1) {
    LOG(INFO) << "parallel_thread_num dosent need to set";
  }
  syncThreadPool_.reset(new SyncThreadPool(threadNum_));

  startThreads();
}

bool ParameterClient2::init(const std::vector<ParameterPtr>& parameters) {
  destroy();

  std::vector<std::string> hosts;
  str::split(FLAGS_pservers, ',', &hosts);
  serviceNum_ = hosts.size() * numPorts_;
  uint64_t denseBlockSize = calcParameterBlockSize(parameters, serviceNum_);

  /// setup prefetch matrix if exists
  for (auto& para : parameters) {
    /// set block size for each parameter
    para->getConfig().set_parameter_block_size(
        para->getConfig().sparse_remote_update() ? para->getConfig().dims(1)
                                                 : denseBlockSize);
  }

  for (auto& para : parameters) {
    CHECK_NE(-1UL, para->getID()) << "id in parameter is not initialized";
    parameterMap_[para->getID()] = para;
  }

  allSegments_.reserve(parameters.size());

  for (auto& para : parameters) {
    ParameterSegments segments;
    segments.name = para->getName();
    segments.id = para->getID();
    allSegments_.push_back(segments);
    if (para->getConfig().sparse_remote_update()) {
      CHECK_EQ(para->getConfig().parameter_block_size(),
               para->getConfig().dims(1))
          << "For sparse remote update parameter,"
          << " block size is the width of each row.";
    }
  }

  /// init clients
  clients_.reserve(serviceNum_);
  recvDataMems_.resize(serviceNum_);

  for (size_t i = 0; i < hosts.size(); ++i) {
    for (int j = 0; j < numPorts_; ++j) {
      LOG(INFO) << "pserver " << i * numPorts_ + j << " " << hosts[i] << ":"
                << port_ + j;
      if (FLAGS_rdma_tcp == "rdma") {
        clients_.emplace_back(hosts[i], port_ + j, F_RDMA);
      } else {
        clients_.emplace_back(hosts[i], port_ + j, F_TCP);
      }
    }
  }

  sparseDistribution_.reset(new SparseParameterDistribution(serviceNum_));

  sleep(2);

  initThreads();

  return true;
}

ParameterClient2::~ParameterClient2() { destroy(); }

void ParameterClient2::destroy() {
  if (clients_.empty()) {
    /// this means not initialized.
    return;
  }
  finishThreads();

  parameterMap_.clear();
  allSegments_.clear();
  clients_.clear();
}

void ParameterClient2::sendParallel(int tid,
                                    size_t numThreads,
                                    ParameterType recvParameterType) {
  int numMyClients = divup(serviceNum_ - tid, numThreads);

  for (int j = 0; j < numMyClients; ++j) {
    REGISTER_TIMER("client_sendAndRecv_send");
    int i = numThreads * j + tid;
    /// Try to make different clients to send data to different pservers
    /// at the same time so that they will not flood data to the same
    /// pserver.
    i = calcClientId(i, serviceNum_);
    clients_[i].send("sendParameter",
                     sendJob_.parallelRequests[i],
                     sendJob_.parallelInputIovs[i]);

    /// clear large structure
    sendJob_.parallelRequests[i].Clear();
    sendJob_.parallelInputIovs[i].clear();
  }

  std::vector<void*> bufs;
  SendParameterResponse response;
  for (int j = 0; j < numMyClients; ++j) {
    REGISTER_TIMER("client_sendAndRecv_recv");
    int i = numThreads * j + tid;
    i = calcClientId(i, serviceNum_);
    auto msgReader = clients_[i].recv(&response);
    CHECK_EQ(msgReader->getNumBlocks(), (size_t)response.blocks_size());
    bufs.clear();
    bufs.reserve(response.blocks_size());
    for (auto& block : response.blocks()) {
      auto it = parameterMap_.find(block.para_id());
      CHECK(it != parameterMap_.end());
      Parameter* parameter = it->second.get();
      real* buf = nullptr;
      if (parameter->getBuf(recvParameterType)) {
        buf = parameter->getBuf(recvParameterType)->getPoint(block.begin_pos());
      } else {
        auto recvMat = dynamic_cast<SparseRowCpuMatrix*>(
            parameter->getMat(recvParameterType).get());
        CHECK(recvMat);
        size_t width = parameter->getConfig().dims(1);
        buf = recvMat->getLocalRow(block.begin_pos() / width);
      }
      /// sparse_id is not useful while receiving data since sparse data
      /// storage is continuous, do commit recieved data as that of dense.
      bufs.push_back(buf);
    }
    msgReader->readBlocks(bufs);
  }
}

void ParameterClient2::prepareSendData(
    ParameterUpdateMode updateMode,
    ParameterType parameterType,
    const std::vector<ParameterSegments>& parameterSegments,
    int64_t numSamples,
    real cost,
    bool sendBackParameter,
    ParameterType sendBackParameterType,
    BatchStatus batchStatus,
    SendJob* sendJob) {
  sendJob->parallelRequests.resize(serviceNum_);
  sendJob->parallelInputIovs.resize(serviceNum_);

  for (auto& request : sendJob->parallelRequests) {
#ifndef PADDLE_DISABLE_TIMER
    if (updateMode == PSERVER_UPDATE_MODE_ADD_GRADIENT) {
      request.set_forwardbackward_time(forwardbackwordTime_);
    }
#endif
    request.set_trainer_id(trainerId_);
    request.set_update_mode(updateMode);
    request.set_send_back_parameter(sendBackParameter);
    request.set_send_back_parameter_type(sendBackParameterType);
    request.set_num_samples(numSamples);
    request.set_cost(cost);
    request.set_batch_status(batchStatus);
    CHECK_EQ(request.blocks_size(), 0);
  }
  for (const auto& segments : parameterSegments) {
    const auto it = parameterMap_.find(segments.id);
    CHECK(it != parameterMap_.end());
    Parameter* parameter = it->second.get();
    CHECK(parameter != nullptr) << "parameter is nullptr";
    int64_t nameHash = std::hash<std::string>()(segments.name);
    bool sendingPara = !(updateMode == PSERVER_UPDATE_MODE_GET_PARAM ||
                         updateMode == PSERVER_UPDATE_MODE_GET_PARAM_SPARSE ||
                         updateMode == PSERVER_UPDATE_MODE_SET_PARAM_ZERO);
    bool sparseUpdate = parameter->getConfig().sparse_remote_update() &&
                        (updateMode == PSERVER_UPDATE_MODE_ADD_GRADIENT ||
                         updateMode == PSERVER_UPDATE_MODE_ASYNC_SGD ||
                         updateMode == PSERVER_UPDATE_MODE_GET_PARAM_SPARSE);

    const auto blockSize = parameter->getConfig().parameter_block_size();
    CHECK_GE(blockSize, 1LU) << "blockSize should > 0 " << blockSize;
    const auto paraSize = parameter->getSize();
    if (sparseUpdate) {
      auto prefetchMat = std::dynamic_pointer_cast<SparsePrefetchRowCpuMatrix>(
          parameter->getMat(PARAMETER_VALUE));
      CHECK(prefetchMat != nullptr) << "prefetchMat is nullptr";
      auto sendMat = dynamic_cast<SparseRowCpuMatrix*>(
          parameter->getMat(parameterType).get());
      CHECK(sendMat != nullptr) << "sendMat is nullptr";

      syncThreadPool_->exec([&](int tid, size_t numThreads) {
        const auto& localIndices = prefetchMat->getLocalIndices();
        /// num of sparse rows
        size_t nLocalBlocks = localIndices.size();
        uint64_t beginDim = 0;
        uint64_t endDim = 0;
        for (size_t row = 0; row < nLocalBlocks; ++row) {
          int64_t blockId = localIndices[row];  // local row -> sparse row
          int serverId = std::abs((blockId + nameHash) % serviceNum_);
          if (serverId % numThreads != (size_t)tid) {
            continue;
          }

          beginDim = blockId * blockSize;
          endDim = std::min<int64_t>(beginDim + blockSize, paraSize);

          auto& request = sendJob->parallelRequests[serverId];
          ParameterBlock* block = request.add_blocks();
          block->set_para_id(segments.id);
          /// global sparse row id
          block->set_block_id(blockId);
          /// local row offset
          block->set_begin_pos(row * blockSize);
          /// block len
          block->set_block_size(endDim - beginDim);

          if (sendingPara) {
            sendJob->parallelInputIovs[serverId].push_back(
                {sendMat->getLocalRow(row), sizeof(real) * (size_t)blockSize});
            /// detect sparse parameter distribution
            sparseDistribution_->probeDistribution(serverId,
                                                   sizeof(real) * blockSize);
          }
        }
      });

    } else {  /// parameter set for dense and sparse
      real* buf =
          sendingPara ? parameter->getBuf(parameterType)->getPoint(0) : nullptr;
      uint64_t endDim = 0;
      for (uint64_t beginDim = 0; beginDim < paraSize; beginDim = endDim) {
        endDim = std::min<int64_t>(beginDim + blockSize, paraSize);
        int64_t blockId = beginDim / blockSize;
        int serverId = std::abs((blockId + nameHash) % serviceNum_);

        auto& request = sendJob->parallelRequests[serverId];
        ParameterBlock* block = request.add_blocks();
        block->set_para_id(segments.id);
        block->set_block_id(blockId);
        block->set_begin_pos(beginDim);
        block->set_block_size(endDim - beginDim);
        if (buf) {
          sendJob->parallelInputIovs[serverId].push_back(
              {buf + beginDim, sizeof(real) * ((size_t)(endDim - beginDim))});
        }
      }
    }
  }  // parameterSegments

  sparseDistribution_->checkAndResetDistribution();
}

void ParameterClient2::sendAndReceiveParameter(
    ParameterUpdateMode updateMode,
    ParameterType parameterType,
    const std::vector<ParameterSegments>& parameterSegments,
    int64_t numSamples,
    real cost,
    bool sendBackParameter,
    ParameterType sendBackParameterType,
    ParameterType recvParameterType) {
  prepareSendData(updateMode,
                  parameterType,
                  parameterSegments,
                  numSamples,
                  cost,
                  sendBackParameter,
                  sendBackParameterType,
                  /*batchStatus = */ BATCH_START_AND_FINISH,
                  &sendJob_);

  syncThreadPool_->exec([&](int tid, size_t numThreads) {
    this->sendParallel(tid, numThreads, recvParameterType);
  });
}

void ParameterClient2::sendParameter(
    ParameterUpdateMode updateMode,
    ParameterType parameterType,
    const std::vector<ParameterSegments>& parameterSegments,
    int64_t numSamples,
    real cost,
    bool sendBackParameter,
    BatchStatus batchStatus) {
  SendJobPtr sendJob = std::make_shared<SendJob>();
  prepareSendData(updateMode,
                  parameterType,
                  parameterSegments,
                  numSamples,
                  cost,
                  sendBackParameter,
                  PARAMETER_VALUE,
                  batchStatus,
                  sendJob.get());

  for (int i = 0; i < threadNum_; i++) {
    sendJobQueue_[i]->enqueue(sendJob);
  }
}

void ParameterClient2::recvParameter() { recvSyncBarrier_->wait(); }

void ParameterClient2::send(int threadId) {
  int index = threadId;
  LOG(INFO) << "send thread " << threadId << " started";
  int numMyClients = divup(serviceNum_ - index, threadNum_);
  while (true) {
    SendJobPtr recvJob = sendJobQueue_[index]->dequeue();
    if (stopping_) {
      recvJobQueue_[index]->enqueue(recvJob);
      break;
    }
    for (int j = 0; j < numMyClients; ++j) {
      REGISTER_TIMER("client_send");
      int i = threadNum_ * j + index;
      /// Try to make different clients to send data to different pservers
      /// at the same time so that they will not flood data to the same
      /// pserver.
      i = calcClientId(i, serviceNum_);
      if (recvJob->parallelRequests.size()) {
        clients_[i].send("sendParameter",
                         recvJob->parallelRequests[i],
                         recvJob->parallelInputIovs[i]);
      } else {
        clients_[i].send("sendData",
                         recvJob->parallelDataRequests[i],
                         recvJob->parallelInputIovs[i]);
      }
    }
    recvJobQueue_[index]->enqueue(recvJob);
  }
}

void ParameterClient2::recv(int threadId) {
  LOG(INFO) << "recv thread " << threadId << " started";
  int index = threadId;
  int numMyClients = divup(serviceNum_ - index, threadNum_);
  while (true) {
    std::vector<void*> bufs;
    SendParameterResponse response;
    SendDataResponse dataResponse;
    SendJobPtr recvJob = recvJobQueue_[index]->dequeue();
    if (stopping_) break;
    for (int j = 0; j < numMyClients; ++j) {
      REGISTER_TIMER("client_recv");
      int i = threadNum_ * j + index;
      i = calcClientId(i, serviceNum_);
      if (recvJob->parallelRequests.size()) {
        auto msgReader = clients_[i].recv(&response);
        CHECK_EQ(msgReader->getNumBlocks(), (size_t)response.blocks_size());
        bufs.clear();
        bufs.reserve(response.blocks_size());
        for (auto& block : response.blocks()) {
          auto it = parameterMap_.find(block.para_id());
          CHECK(it != parameterMap_.end());
          Parameter* parameter = it->second.get();
          real* buf =
              parameter->getBuf(PARAMETER_VALUE)->getPoint(block.begin_pos());
          CHECK_EQ(msgReader->getBlockLength(bufs.size()),
                   sizeof(real) * (block.block_size()));
          bufs.push_back(buf);
        }
        msgReader->readBlocks(bufs);
      } else {
        auto msgReader = clients_[i].recv(&dataResponse);
        CHECK_EQ(msgReader->getNumBlocks(), (size_t)dataResponse.blocks_size());
        size_t totalLen = msgReader->getTotalLength();
        if (0 == totalLen) {
          continue;
        }
        auto& recvMem = recvDataMems_[dataResponse.server_id()];
        CHECK_EQ(dataResponse.blocks_size(), 1)
            << "Only one block currently support now!";
        auto& block = dataResponse.blocks(0);
        CHECK_EQ(totalLen % sizeof(block.data_size()), 0U);
        recvMem = std::make_shared<CpuMemoryHandle>(totalLen);
        msgReader->readNextBlock(recvMem.get()->getBuf());
      }
    }
    recvSyncBarrier_->wait();
  }
}

void ParameterClient2::waitPassStart() {
  WaitPassStartRequest request;
  std::vector<WaitPassStartResponse> responses;
  multiCall(__func__, request, &responses);
}

void ParameterClient2::waitPassFinish() {
  WaitPassFinishRequest request;
  std::vector<WaitPassFinishResponse> responses;
  multiCall(__func__, request, &responses);
}

void ParameterClient2::synchronize(SyncObject syncObjectId) {
  SynchronizeRequest request;
  request.set_sync_object_id(syncObjectId);
  std::vector<SynchronizeResponse> responses;
  multiCall(__func__, request, &responses);
}

void ParameterClient2::asyncFinishPass(SyncObject syncObjectId) {
  SynchronizeRequest request;
  request.set_sync_object_id(syncObjectId);
  request.set_trainer_id(trainerId_);
  std::vector<SynchronizeResponse> responses;
  multiCall(__func__, request, &responses);
}

void ParameterClient2::setConfig(const OptimizationConfig& optConfig,
                                 const std::string& saveDir,
                                 bool isSparseServer) {
  SetConfigRequest request;
  std::vector<SetConfigResponse> responses;

  for (auto& nameAndPara : parameterMap_) {
    *request.add_param_configs() = nameAndPara.second->getConfig();
  }

  *request.mutable_opt_config() = optConfig;
  request.set_save_dir(saveDir);
  request.set_is_sparse_server(isSparseServer);

  std::vector<SetConfigRequest> requests;
  requests.resize(clients_.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].CopyFrom(request);
    requests[i].set_server_id(i);
  }

  responses.resize(clients_.size());
  size_t numClients = clients_.size();
  for (size_t i = 0; i < numClients; ++i) {
    clients_[i].send(__func__, requests[i]);
  }
  for (size_t i = 0; i < numClients; ++i) {
    clients_[i].recv(&responses[i]);
  }
}

bool ParameterClient2::inStatus(PServerStatus status) {
  GetStatusRequest request;
  std::vector<GetStatusResponse> responses;

  bool ok = true;
  multiCall("getStatus", request, &responses);
  for (auto& response : responses) {
    if (response.status() != status) {
      ok = false;
    }
  }

  return ok;
}

void ParameterClient2::setStatus(PServerStatus status) {
  SetStatusRequest request;
  request.set_status(status);
  std::vector<SetStatusResponse> responses;
  multiCall(__func__, request, &responses);
}

void ParameterClient2::waitForStatus(PServerStatus status) {
  while (!inStatus(status)) {
    sleep(1);
  }
}

template <typename Proto>
static void validateResponses(const std::vector<Proto>& responses) {
  for (auto& response : responses) {
    CHECK(response.return_message().empty())
        << "client" << &response - &responses[0]
        << " error:" << response.return_message();
  }
}

PServerVector ParameterClient2::createVector() {
  CreateVectorRequest request;
  std::vector<CreateVectorResponse> responses;
  int64_t handle = -1;

  multiCall(__func__, request, &responses);
  validateResponses(responses);

  for (auto& response : responses) {
    if (handle == -1) {
      handle = response.handle();
    } else {
      CHECK_EQ(handle, response.handle()) << "Inconsistent handle from client"
                                          << &response - &responses[0] << " "
                                          << handle << " " << response.handle();
    }
  }
  return PServerVector{handle};
}

void ParameterClient2::releaseVector(PServerVector handle) {
  ReleaseVectorRequest request;
  std::vector<ReleaseVectorResponse> responses;

  request.set_handle(handle.handle);
  multiCall(__func__, request, &responses);
  validateResponses(responses);
}

PServerMatrix ParameterClient2::createMatrix(int32_t numCols) {
  CreateMatrixRequest request;
  std::vector<CreateMatrixResponse> responses;
  int64_t handle = -1;

  request.set_num_cols(numCols);
  multiCall(__func__, request, &responses);
  validateResponses(responses);

  for (auto& response : responses) {
    if (handle == -1) {
      handle = response.handle();
    } else {
      CHECK_EQ(handle, response.handle()) << "Inconsistent handle from client"
                                          << &response - &responses[0] << " "
                                          << handle << " " << response.handle();
    }
  }
  return PServerMatrix{handle};
}

void ParameterClient2::releaseMatrix(PServerMatrix handle) {
  ReleaseMatrixRequest request;
  std::vector<ReleaseMatrixResponse> responses;

  request.set_handle(handle.handle);
  multiCall(__func__, request, &responses);
  validateResponses(responses);
}

void PreparedOperations::addOperationHelper(Operation* op, CpuVectorPtr vec) {
  ProtoVector& pvec = *op->add_vectors();
  size_t dim = vec->getSize();
  pvec.set_dim(dim);
  copyToRepeatedField(pvec.mutable_values(), vec->getData(), vec->getSize());
}

void PreparedOperations::addOperationHelper(Operation* op, CpuMatrixPtr mat) {
  ProtoMatrix& pmat = *op->add_matrices();
  pmat.set_num_cols(mat->getWidth());
  pmat.set_num_rows(mat->getHeight());
  copyToRepeatedField(
      pmat.mutable_values(), mat->getData(), pmat.num_cols() * pmat.num_rows());
}

static inline real addTwo(real a, double b) { return a + b; }

void ParameterClient2::doOperation(PreparedOperations& ops,
                                   bool waitForGradient,
                                   bool sendBackGradient,
                                   bool releasePass) {
  std::vector<DoOperationResponse> responses;
  ops.request_.set_wait_for_gradient(waitForGradient);
  ops.request_.set_send_back_parameter(sendBackGradient);
  ops.request_.set_release_pass(releasePass);
  multiCall(__func__, ops.request_, &responses);
  validateResponses(responses);
  size_t numPassFinishServers = 0;

  size_t numOps = ops.request_.operations_size();
  for (auto& response : responses) {
    numPassFinishServers += response.pass_finish();
    CHECK_EQ(numOps, (size_t)response.results_size());
    for (size_t opId = 0; opId < numOps; ++opId) {
      const OperationResult& result = response.results(opId);
      std::vector<real*>& resultScalars = ops.localResults_[opId].resultScalars;
      std::vector<CpuVectorPtr>& resultVectors =
          ops.localResults_[opId].resultVectors;
      std::vector<CpuMatrixPtr>& resultMatrices =
          ops.localResults_[opId].resultMatrices;

      if (&response == &responses[0]) {
        /// Initialize results to zero

        resultScalars.resize(result.scalars_size());
        for (auto p : resultScalars) {
          if (!p) continue;
          *p = 0;
        }
        size_t numVectors = result.vectors_size();
        resultVectors.resize(numVectors);
        for (size_t i = 0; i < numVectors; ++i) {
          if (!resultVectors[i]) continue;
          resultVectors[i]->resize(result.vectors(i).dim());
          resultVectors[i]->zeroMem();
        }
        size_t numMatrices = result.matrices_size();
        resultMatrices.resize(numMatrices);
        for (size_t i = 0; i < numMatrices; ++i) {
          if (!resultMatrices[i]) continue;
          resultMatrices[i]->resize(result.matrices(i).num_rows(),
                                    result.matrices(i).num_cols());
          resultMatrices[i]->zeroMem();
        }
      }

      // aggregate results from each pserver to results

      CHECK_EQ(resultScalars.size(), (size_t)result.scalars_size());
      for (ssize_t i = 0; i < result.scalars_size(); ++i) {
        real* rscalar = resultScalars[i];
        if (!rscalar) continue;
        *rscalar += result.scalars(i);
      }

      CHECK_EQ(resultVectors.size(), (size_t)result.vectors_size());
      for (auto& vec : result.vectors()) {
        int i = &vec - &result.vectors(0);
        CpuVectorPtr rvec = resultVectors[i];
        if (!rvec) continue;
        CHECK_EQ(rvec->getSize(), (size_t)vec.dim());
        std::transform(rvec->getData(),
                       rvec->getData() + rvec->getSize(),
                       vec.values().data(),
                       rvec->getData(),
                       addTwo);
      }

      CHECK_EQ(resultMatrices.size(), (size_t)result.matrices_size());
      for (auto& mat : result.matrices()) {
        int i = &mat - &result.matrices(0);
        CpuMatrixPtr rmat = resultMatrices[i];
        if (!rmat) continue;
        CHECK_EQ(rmat->getHeight(), (size_t)mat.num_rows());
        CHECK_EQ(rmat->getWidth(), (size_t)mat.num_cols());

        std::transform(rmat->getData(),
                       rmat->getData() + rmat->getElementCnt(),
                       mat.values().data(),
                       rmat->getData(),
                       addTwo);
      }
    }
  }
  passFinish_ = numPassFinishServers == clients_.size();
}

real ParameterClient2::vectorDotProduct(PServerVector u, PServerVector v) {
  real result = 0.0;
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_utv, u, v)(&result);
  doOperation(ops, false, false);
  return result;
}

void ParameterClient2::vectorScale(PServerVector u, real a) {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_au, u, a);
  doOperation(ops, false, false);
}

void ParameterClient2::vectorCopy(PServerVector src, PServerVector dst) {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_COPY, src, dst);
  doOperation(ops, false, false);
}

void ParameterClient2::vectorAddMult(PServerVector u, PServerVector v, real a) {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_au_bv, v, u, a, (real)1);
  doOperation(ops, false, false);
}

void ParameterClient2::vectorAddMultInto(PServerVector u,
                                         PServerVector v,
                                         PServerVector w,
                                         real a) {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_au_bv_cw, v, w, u, (real)1, a, (real)0);
  doOperation(ops, false, false);
}

void ParameterClient2::vectorScaleInto(PServerVector u,
                                       PServerVector v,
                                       real a) {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_au_bv, v, u, a, (real)0);
  doOperation(ops, false, false);
}

void ParameterClient2::loadValueVector(const std::string& dirName) {
  LoadValueRequest request;
  request.set_dir_name(dirName);
  std::vector<LoadValueResponse> responses;

  multiCall(__func__, request, &responses);
  validateResponses(responses);
}

void ParameterClient2::saveValueVector(const std::string& dirName) {
  SaveValueRequest request;
  request.set_dir_name(dirName);
  std::vector<SaveValueResponse> responses;

  multiCall(__func__, request, &responses);
  validateResponses(responses);
}

}  // namespace paddle
