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

#include "ParameterService.pb.h"
#include "paddle/math/Matrix.h"
#include "paddle/pserver/ProtoServer.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Queue.h"

namespace paddle {

/**
 * it manages all connections to pservers.
 * it exists two modes to manage connections to all pservers. Firstly, one
 * connection owns two threads that separately manage to send and receive
 * data. Secondly, each thread uses one connection for all activation in it.
 * the first solution arms with sendThreads_/recvThreads_ and sendJobQueue_/
 * recvJobQueue_. the second solution use some shared thread pool to manage
 * connections.
 */
class BaseClient {
protected:
  typedef std::unique_ptr<std::thread> ThreadPtr;
  typedef std::vector<std::vector<iovec>> InputIovs;
  typedef std::vector<SendParameterRequest> SendRequest;
  typedef std::vector<SendDataRequest> SendDataRequestVec;

  // TODO(yanfei):
  // refine data structure to unify parameter and features communication
  struct SendJob {
    /// store parameters related blocks data
    InputIovs parallelInputIovs;
    /// store protobuf request
    SendRequest parallelRequests;
    /// store data, such as features for metric learning
    SendDataRequestVec parallelDataRequests;
  };

public:
  explicit BaseClient(bool separate = false, int numPorts = FLAGS_ports_num);

  virtual ~BaseClient();

  typedef std::shared_ptr<SendJob> SendJobPtr;
  typedef Queue<SendJobPtr> SendQueue;

  /// send data to server, support only synchronize
  template <class DataType>
  void putData(int clientId,
               SendDataType type,
               DataType* datas,
               size_t size,
               DataUpdateMode mode) {
    synchronize(SYNC_DATA);
    sendData(clientId, type, mode, datas, size);
    recvData();
    synchronize(SYNC_DATA);
  }

  template <class DataType>
  void putOwnData(int clientId,
                  SendDataType type,
                  DataType* datas,
                  size_t size) {
    putData(clientId, type, datas, size, DATA_UPDATE_MODE_SET_OWN);
  }

  template <class DataType>
  void getAllData(int clientId,
                  SendDataType type,
                  DataType* datas,
                  size_t size) {
    sendData(clientId,
             type,
             DATA_UPDATE_MODE_GET_ALL,
             reinterpret_cast<DataType*>(NULL),
             0);
    recvData();
    size_t dataOffset = 0;
    for (auto& recvMem : recvDataMems_) {
      CHECK_LE(dataOffset, size);
      size_t memSize = std::min(recvMem.get()->getSize(),
                                sizeof(DataType) * (size - dataOffset));
      CHECK_EQ(memSize % sizeof(DataType), size_t(0));
      memcpy(datas + dataOffset, recvMem.get()->getBuf(), memSize);
      dataOffset += memSize / sizeof(DataType);
    }
    CHECK_EQ(dataOffset, size);
  }

  /**
   * Reduces values on all clients.
   * This reduce just support SUM.
   * The results are saved in recvBuf of rootId client
   */
  template <class DataType>
  void reduce(DataType* sendBuf,
              DataType* recvBuf,
              size_t size,
              int clientId,
              int rootId) {
    putOwnData(clientId, DATA_REDUCE_SUM, sendBuf, size);
    if (rootId == clientId) {
      getAllData(clientId, DATA_REDUCE_SUM, recvBuf, size);
    }
  }

  /**
   * return trans data type according to the input type
   */
  virtual TransDataType getTransDtype(const std::type_info& info) {
    TransDataType dataType;
    if (typeid(int*) == info) {  // NOLINT
      dataType = TRANS_INT32;
    } else if (typeid(uint32_t*) == info) {  // NOLINT
      dataType = TRANS_UINT32_T;
    } else if (typeid(int64_t*) == info) {  // NOLINT
      dataType = TRANS_INT64_T;
    } else if (typeid(uint64_t*) == info) {  // NOLINT
      dataType = TRANS_UINT64_T;
    } else if (typeid(float*) == info) {  // NOLINT
      dataType = TRANS_FLOAT;
    } else if (typeid(double*) == info) {  // NOLINT
      dataType = TRANS_DOUBLE;
    } else {
      LOG(FATAL) << "not supported";
    }
    return dataType;
  }

protected:
  /// for a > 0, b > 0:
  /// return the smallest x s.t. b*x >= a
  static int divup(int a, int b) { return (a + b - 1) / b; }

  int calcClientId(int i, int serviceNum) {
    return (i + FLAGS_trainer_id * numPorts_) % serviceNum;
  }

  /// start threads in sendThreads_ and recvThreads_
  void startThreads();

  /// finish threads in sendThreads_ and recvThreads_
  void finishThreads();

  template <class DataType>
  void prepareData(int clientId,
                   SendDataType type,
                   DataUpdateMode updateMode,
                   DataType* datas,
                   size_t size,
                   SendJob* sendJob) {
    sendJob->parallelDataRequests.resize(serviceNum_);
    sendJob->parallelInputIovs.resize(serviceNum_);
    for (int i = 0; i < serviceNum_; ++i) {
      auto& request = sendJob->parallelDataRequests[i];
      request.set_update_mode(updateMode);
      request.set_type(type);
      request.set_client_id(clientId);
      request.set_server_id(i);
    }

    /// split datas which need send to Server into serviceNum_ pieces
    if (!datas) {
      CHECK(!size) << "ownSize should be zero since datas is nullptr";
    }
    size_t baseSize = size / serviceNum_;
    size_t dataOffset = 0;
    for (int i = 0; i < serviceNum_; ++i) {
      auto& request = sendJob->parallelDataRequests[i];
      DataBlock* block = request.add_blocks();
      size_t ownSize = size_t(i) < size % serviceNum_ ? baseSize + 1 : baseSize;
      size_t realSize = datas ? std::max(ownSize, size_t(1)) : 0;
      block->set_total_size(realSize * sizeof(DataType));
      block->set_data_size(sizeof(DataType));
      // TODO(yuyang18): The getTransDtype can be rewritten as template method
      //                 to reduce runtime overhead.
      block->set_data_type(getTransDtype(typeid(DataType*)));  // NOLINT
      if (datas) {
        sendJob->parallelInputIovs[i].push_back(
            {datas + dataOffset, realSize * sizeof(DataType)});
      }
      dataOffset += ownSize;
    }
    CHECK_EQ(dataOffset, size);
  }

  /**
   * @brief send data to all data servers
   *
   * @note  each trainer sends all its data to all data servers
   *        it's for broadcast data synchronization, such as features
   *        synchronization in metric learning.
   */
  template <class DataType>
  void sendData(int clientId,
                SendDataType type,
                DataUpdateMode updateMode,
                DataType* datas,
                size_t size) {
    SendJobPtr sendJob = std::make_shared<SendJob>();
    prepareData(clientId, type, updateMode, datas, size, sendJob.get());
    for (int i = 0; i < threadNum_; ++i) {
      sendJobQueue_[i]->enqueue(sendJob);
    }
  }

  /**
   * @brief recv data from all data servers
   *
   * @note  synchronize all recv threads
   */
  void recvData();

  /// send request, and recv responses
  template <typename ProtoIn, typename ProtoOut>
  void multiCall(const char* funcName,
                 const ProtoIn& request,
                 std::vector<ProtoOut>* responses) {
    responses->resize(clients_.size());
    size_t numClients = clients_.size();
    for (size_t i = 0; i < numClients; ++i) {
      clients_[i].send(funcName, request);
    }
    for (size_t i = 0; i < numClients; ++i) {
      clients_[i].recv(&(*responses)[i]);
    }
  }

  /**
   * @brief synchronize all trainers and pservers
   *
   * @note  used to ensure that data of all trainers have been received
   */
  void synchronize(SyncObject syncObjectId = SYNC_DEFAULT);

  /**
   * @brief use multithread to separately send data
   *
   * @note  each thread should read its own JobQueue to handle requests
   *        each thread should calcClientId() to retrieve connections
   *        managed by himself.
   *        send and recv are implemented in child class.
   */
  virtual void send(int threadId) = 0;

  /**
   * @brief use multithread to separately receive data
   *
   * @note  almost same as send()
   */
  virtual void recv(int threadId) = 0;

protected:
  bool stopping_;
  /// nodes * ports that means the number of real pservers
  int serviceNum_;
  /**
   * threads num for managing all services. Normally the
   * number of pservers are relatively less than several
   * hundreds so that using thread-based parallelization
   * can benifit traffic performance and pserver's sgd
   * optimization performance.
   */
  int threadNum_;
  /// the connection manager at client end
  std::vector<ProtoClient> clients_;
  /// send threads for parallelization
  std::vector<ThreadPtr> sendThreads_;
  /// recv threads for parallelization
  std::vector<ThreadPtr> recvThreads_;
  std::unique_ptr<ThreadBarrier> recvSyncBarrier_;

  // TODO(yanfei):
  // current pserver's will return value until all parameters'
  // optimization are finished so that recv are not overlapped
  // in reality. More robust implimentation should be to pipeline
  // all send/recv action based on parameter unit level, and
  // it will benifits deep and larger model training in future,
  // especially local node compution power surpasses inter-connection
  // such as GPU cluster, even with BOX GPU cluster.
  // queue for buffering send request
  /**
   * send/recv queue cooperates with each other to accomplish
   * overlapping communication with forwardBackward action.
   */
  std::vector<std::unique_ptr<SendQueue>> sendJobQueue_;
  /// queue for buffering recv request
  std::vector<std::unique_ptr<SendQueue>> recvJobQueue_;
  /// specific for dserver
  SendJob sendJob_;
  /// port num for each node
  int numPorts_;
  /// if set, overlapped optimization is disabled
  bool separateSendAndRecv_;
  std::vector<CpuMemHandlePtr> recvDataMems_;
};
}  // namespace paddle
