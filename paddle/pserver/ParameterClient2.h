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
#include <mutex>
#include <unordered_map>
#include <vector>

#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/pserver/BaseClient.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/Queue.h"
#include "paddle/utils/Util.h"

#include "ParameterService.pb.h"

#include "ProtoServer.h"
#include "SparseParameterDistribution.h"

DECLARE_int32(parallel_thread_num);

namespace paddle {

struct PServerMatrix {
  int64_t handle;
};

struct PServerVector {
  int64_t handle;
};

/**
 * @brief A class to help to prepare server-side operations.
 */
class PreparedOperations {
protected:
  class ResultsAdder;
  struct LocalOperationResult;

public:
  /**
   * Offers an easy way to prepare operations that will be performed on
   * server-side.
   *
   * Usage:
   * @code
   *   addOperation(optype, arguments...)(results...)
   * @endcode
   *
   * Examples:
   * 1. set pserver vector to 1:
   * @code
   *   PServerVector u = parameterClient.createVector();
   *   addOperation(PSERVER_OP_RESET, u, (real)1);
   * @endcode
   *
   * 2. Compute inner product of to pserver vectors.
   * @code
   *   PServerVector u = parameterClient.createVector();
   *   PServerVector v = parameterClient.createVector();
   *   real result;
   *   addOperation(PSERVER_OP_utv, u, v)(&result)
   * @endcode
   *
   * @param[in] operation The operation that pserver will perform.
   * @param[in] args Argument list of the operation
   * @return A ResultsAdder object initialized with the last element of
   *         localResults_.
   */
  template <typename... Args>
  ResultsAdder addOperation(MatrixVectorOperation operation, Args... args) {
    Operation* op = request_.add_operations();
    op->set_operation(operation);
    localResults_.emplace_back();
    addOperationHelper(op, args...);
    return ResultsAdder(&localResults_.back());
  }

protected:
  void addOperationHelper(Operation* op) {}

  /**
   * @brief Helper function to add an new operation that takes a PServerVector
   *        as an operand.
   */
  void addOperationHelper(Operation* op, PServerVector arg) {
    op->add_pvectors(arg.handle);
  }

  /**
   * @brief Helper function to add an new operation that takes a PServerMatrix
   *        as an operand.
   */
  void addOperationHelper(Operation* op, PServerMatrix arg) {
    op->add_pmatrices(arg.handle);
  }

  /**
   * @brief Helper function to add an new operation that takes a real valued
   *        scalar as an operand.
   */
  void addOperationHelper(Operation* op, real arg) { op->add_scalars(arg); }

  /**
   * @brief Helper function to add an new operation that takes a CpuVectorPtr
   *        as an operand.
   * @note The array of CpuVectors that arg points to will be copied to
   *       op's vectors field.
   */
  void addOperationHelper(Operation* op, CpuVectorPtr arg);

  /**
   * @brief Helper function to add an new operation that takes a CpuMatrixPtr
   *        as an operand.
   * @note The array of CpuMatrixs that arg points to will be copied to
   *       op's matrices field.
   */
  void addOperationHelper(Operation* op, CpuMatrixPtr arg);

  /**
   * @brief Helper function to add an new operation and prepare the operands.
   *
   * @tparam Arg An operand of the operation.
   * @tparam Args A list of rest operands of the operation.
   * @param op Pointer to an Operation object.
   */
  template <typename Arg, typename... Args>
  void addOperationHelper(Operation* op, Arg arg, Args... args) {
    addOperationHelper(op, arg);
    addOperationHelper(op, args...);
  }

  /**
   * @brief ResultsAdder offers easy ways to quickly store operation results.
   */
  class ResultsAdder {
  public:
    explicit ResultsAdder(LocalOperationResult* localResult)
        : localResult_(localResult) {}
    template <typename... Args>
    void operator()(Args... args) {
      addResult(args...);
    }
    void addResult() {}
    void addResult(real* arg) { localResult_->resultScalars.push_back(arg); }
    void AddResult(CpuVectorPtr arg) {
      localResult_->resultVectors.push_back(arg);
    }
    void AddResult(CpuMatrixPtr arg) {
      localResult_->resultMatrices.push_back(arg);
    }
    template <typename Arg, typename... Args>
    void addResult(Arg arg, Args... args) {
      addResult(arg);
      addResult(args...);
    }

  protected:
    LocalOperationResult* localResult_;
  };

protected:
  DoOperationRequest request_;
  std::vector<iovec> inputIovs_;
  struct LocalOperationResult {
    std::vector<real*> resultScalars;
    std::vector<CpuVectorPtr> resultVectors;
    std::vector<CpuMatrixPtr> resultMatrices;
  };
  std::vector<LocalOperationResult> localResults_;
  friend class ParameterClient2;
};

struct ParameterSegments {
  std::string name;  // name of the parameter
  size_t id;         // id of the parameter
};

/**
 * The client interface for parameter server. ParameterClient2 supports 2 modes
 * for managing connections to parameter servers, in the 1st mode one connection
 * is shared by 2 threads that are separately responsible for sending and
 * recieving activities, in the 2nd mode one connection is owned by only one
 * thread, and all the sending and recieving activities run in that single
 * thread.
 * TODO(yanfei):
 * Additional core idea to further optimizate pserver performance is
 * to do sync-sgd based parameter level instead of pserver level.
 * full-parallelization based parameter level for sync-sgd also can
 * sense forwardbackward computation layer-by-layer for more deeper layer
 * model.
 * Firstly, pserver can do full-parallelization on all computation based
 * parameter level instead of waiting for all gradients are finished and
 * start to send back parameters value immediately if parameter is ready
 * instead of waiting for all parameters value are ready
 * Secondly, parameter client can write back parameters to GPU instead of
 * waiting until all parameters are received to CPU host end.
 */
class ParameterClient2 : public BaseClient {
public:
  /** Constructor.
   * @param separate True if sending and recieving activities are separated
   *                 into 2 threads, otherwise false.
   * @param port Port number that parameter client runs on.
   * @param numPorts Number of ports parameter clients occupies,
   *                 numPorts * pserver number is the total number of
   *                 connections the parameter client maintains.
   */
  ParameterClient2(bool separate = false,
                   int port = FLAGS_port,
                   int numPorts = FLAGS_ports_num);

  ~ParameterClient2();

  static int calcParameterBlockSize(const std::vector<ParameterPtr>& parameters,
                                    size_t serviceNum);

public:
  bool init(const std::vector<ParameterPtr>& parameters);

  /// service functions

  /**
   * @brief Sends the segments in parameter to parameter servers, then receives
   *        the response from the servers.
   * @param[in] updateMode Indicates how parameters should be updated on the
   *            server side.
   * @param[in] parameterType Type of parameter that will be sent.
   * @param[in] segments Segments in the parameter that will be sent.
   * @param[in] numSamples Number of samples this update is based on.
   * @param[in] cost Cost of the batch, will be used to calculate global object
   *            value.
   * @param[in] sendBackParameter True if the updated parameters should be sent
   *            back, otherwise false.
   * @param[in] sendBackParameterType Send back parameter type on pserver,
   *            PARAMETER_VALUE by default
   * @param[in] recvParameterType pserver[sendBackParameterType] will be copy to
   *            client[recvParameterType]
   * @note Only parameterType will be sent.
   */
  void sendAndReceiveParameter(ParameterUpdateMode updateMode,
                               ParameterType parameterType,
                               const std::vector<ParameterSegments>& segments,
                               int64_t numSamples,
                               real cost,
                               bool sendBackParameter,
                               ParameterType sendBackParameterType,
                               ParameterType recvParameterType);

  /**
   * @brief Sends all parameters to parameter servers, and receives the response
   *        from the servers.
   */
  void sendAndReceiveParameter(
      ParameterUpdateMode updateMode,
      ParameterType parameterType,
      int64_t numSamples,
      real cost,
      bool sendBackParameter,
      ParameterType sendBackParameterType = PARAMETER_VALUE,
      ParameterType recvParameterType = PARAMETER_VALUE) {
    sendAndReceiveParameter(updateMode,
                            parameterType,
                            allSegments_,
                            numSamples,
                            cost,
                            sendBackParameter,
                            sendBackParameterType,
                            recvParameterType);
  }

  /**
   * @brief Sends the segments in parameter to parameter servers. Each
   *        sendParameter() must be paired with a recvParameter() in the future.
   *        Only parameterType will be sent.
   *
   * @param[in] updateMode Indicates how parameters should be updated on the
   *            server side.
   * @param[in] parameterType Type of parameter that will be sent.
   * @param[in] segments Segments in the parameter that will be sent.
   * @param[in] numSamples Number of samples this update is based on.
   * @param[in] cost Cost of the batch, will be used to calculate global object
   *            value.
   * @param[in] sendBackParameter True if the updated parameters should be sent
   *            back, otherwise false.
   * @param[in] batchStatus Status of the batch.
   * @note This function is non-blocking. This means that parameter should
   *       not change between this call and recvParameter()
   */
  void sendParameter(ParameterUpdateMode updateMode,
                     ParameterType parameterType,
                     const std::vector<ParameterSegments>& segments,
                     int64_t numSamples,
                     real cost,
                     bool sendBackParameter,
                     BatchStatus batchStatus);

  void recvParameter();

  /**
   * Sends all parameters to parameter servers, recvParameter() have to be
   * invoked
   * afterwards.
   *
   * @note This function is non-blocking. This means that if parameter should
   *       not changes between this call and recvParameter()
   */
  void sendParameter(ParameterUpdateMode updateMode,
                     ParameterType parameterType,
                     int64_t numSamples,
                     real cost,
                     bool sendBackParameter,
                     BatchStatus batchStatus) {
    sendParameter(updateMode,
                  parameterType,
                  allSegments_,
                  numSamples,
                  cost,
                  sendBackParameter,
                  batchStatus);
  }

  /// Get all parameters from parameter servers
  void getParameter(ParameterType recvParameterType = PARAMETER_VALUE,
                    ParameterType sendBackParameterType = PARAMETER_VALUE) {
    sendAndReceiveParameter(PSERVER_UPDATE_MODE_GET_PARAM,
                            PARAMETER_VALUE,
                            0,     // numSamples = 0
                            0,     // cost = 0
                            true,  // sendBackParameter = true
                            sendBackParameterType,
                            recvParameterType);
  }

  /// Get parameters by sparse row ids from parameter servers
  void getParameterSparse(
      ParameterType recvParameterType = PARAMETER_VALUE,
      ParameterType sendBackParameterType = PARAMETER_VALUE) {
    sendAndReceiveParameter(PSERVER_UPDATE_MODE_GET_PARAM_SPARSE,
                            PARAMETER_VALUE,
                            0,     // numSamples = 0
                            0,     // cost = 0
                            true,  // sendBackParameter = true
                            sendBackParameterType,
                            recvParameterType);
  }

  /// Set all parameters on parameter servers using the local parameters
  void setParameter() {
    sendAndReceiveParameter(PSERVER_UPDATE_MODE_SET_PARAM,
                            PARAMETER_VALUE,
                            0,       // numSamples = 0
                            0,       // cost = 0
                            false);  // sendBackParameter = false
  }
  /**
   * Set all parameters on parameter servers, values will be zero
   * means do not sending local parameters
   */
  void setParameterZero() {
    sendAndReceiveParameter(PSERVER_UPDATE_MODE_SET_PARAM_ZERO,
                            PARAMETER_VALUE,
                            0,       // numSamples = 0
                            0,       // cost = 0
                            false);  // sendBackParameter = false
  }

  /**
   * @brief Wait until all gradient servers start one pass.
   *
   * @note This is now only used by the gradient servers for "sgd"
   *       algorithm. Calling this function means that the calling gradient
   *       server is ready to start a new pass.
   */
  void waitPassStart();

  /**
   * @brief Wait until all gradient servers finish one pass.
   *
   * @note This is now only used by the gradient servers for "sgd" algorithm.
   *       Calling this function means that the calling gradient server
   *       finishes one pass.
   */
  void waitPassFinish();

  /// Wait until all gradient servers call this function.
  void synchronize(SyncObject syncObjectId = SYNC_DEFAULT);

  /// Called when async-sgd finish pass.
  void asyncFinishPass(SyncObject syncObjectId = SYNC_DEFAULT);

  void asyncStartPass(SyncObject syncObjectId = SYNC_DEFAULT) {
    return synchronize(syncObjectId);
  }

  /**
   * @brief Execute the prepared operations on pservers, fetch the results and
   *        aggregate results from different pservers.
   * @param[in] ops Prepared operations that will be executed on pservers.
   * @param[in] waitForGradient If true, wait for gradient to be ready before
   *            starting the operations.
   * @param[in] sendBackParameter If true, send back the parameter to clients
   *            after the operations are finished.
   * @param[in] If true, and if all clients call waitPassFinish, signal all
   *            clients finish the pass.
   */
  void doOperation(PreparedOperations& ops,
                   bool waitForGradient,
                   bool sendBackParameter,
                   bool releasePass = true);

  /**
   * Set the configuration of pserver, including parameter config and
   * optimization config
   */
  void setConfig(const OptimizationConfig& optConfig,
                 const std::string& saveDir = "",
                 bool isSparseServer = false);

  /// Return true if all pservers are in the given status
  bool inStatus(PServerStatus status);
  bool isPassFinish() { return passFinish_; }

  /// Set pserver status
  void setStatus(PServerStatus status);

  /**
   * @brief Wait until all pservers are at status
   * @note This function is not suitable for frequent use,
   *       because it sleeps 1 second each time when condition is satisfied.
   */
  void waitForStatus(PServerStatus status);

  /// Create a column vector. The size is the dimension of parameter.
  PServerVector createVector();

  /// Release the PServerVector given handle.
  void releaseVector(PServerVector handle);

  /**
   * Create a column major matrix. The number of rows is the dimension of
   * parameter. The number of columns is specifed by numCols.
   */
  PServerMatrix createMatrix(int32_t numCols);

  /// Release the PServerMatrix given handle.
  void releaseMatrix(PServerMatrix handle);

  // Some basic algebra functions
  /// Calculate the dot product of u and v
  real vectorDotProduct(PServerVector u, PServerVector v);

  /// Scale u by a
  void vectorScale(PServerVector u, real a);

  /// Copy from src to dest
  void vectorCopy(PServerVector src, PServerVector dst);

  /// u += v * a
  void vectorAddMult(PServerVector u, PServerVector v, real a);

  /// u = v + w * a
  void vectorAddMultInto(PServerVector u,
                         PServerVector v,
                         PServerVector w,
                         real a);
  /// u = v * a
  void vectorScaleInto(PServerVector u, PServerVector v, real a);

  /// Return pserver parameter value.
  PServerVector getPServerParameterValue() {
    PServerVector vec;
    vec.handle = PARAMETER_VALUE;
    return vec;
  }

  /// Return pserver parameter gradient.
  PServerVector getPServerParameterGradient() {
    PServerVector vec;
    vec.handle = PARAMETER_GRADIENT;
    return vec;
  }

  /**
   * Tell pservers to load value vector from file.
   *
   * @param[in] dirName The directory that contains the value vector file.
   */
  void loadValueVector(const std::string& dirName);

  /// Tell pservers to save value vector to file.
  void saveValueVector(const std::string& dirName);

  void setTrainerId(int trainerId) { trainerId_ = trainerId; }

#ifndef PADDLE_DISABLE_TIMER
  void setForwardbackwardTime(uint64_t delta) { forwardbackwordTime_ = delta; }
#endif

protected:
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

private:
  void destroy();

  /**
   * @brief management function for parallelizing send/recv all connections
   *        to all pservers. it is called under one SyncThreadPool. it
   *        supports to use N thread to control M connections. the receiving
   *        actions can be started until all sending action to all connections
   *        owned by current thread are finished. Different connections
   * controlled
   *        by different threads can transfer data asynchronously.
   */
  void sendParallel(int tid,
                    size_t numThreads,
                    ParameterType recvParameterType);
  /// sending thread routine for asynchronously send data
  void send(int threadId);
  /// receiving thread routing for asynchronously receive data
  void recv(int threadId);

  /**
   * @brief main routine to build data for pserver
   *
   * @note  it can prepare different kinds of parameter type data. it can
   *        be regarded as layer for bridging real parameters data and
   *        protobuf data for communication.
   *        TODO(yanfei):
   *        can abstract additional layer to encode and decode data to/from
   *        protobuf data.
   */
  void prepareSendData(
      ParameterUpdateMode updateMode,
      ParameterType parameterType,  // client send type
      const std::vector<ParameterSegments>& parameterSegments,
      int64_t numSamples,
      real cost,
      bool sendBackParameter,
      ParameterType sendBackParameterType,  // send back type in pserver
      BatchStatus batchStatus,
      SendJob* sendJob);

  /// start necessary threads for threadPool
  void initThreads();

protected:
  /// start port number of pserver
  /// it deduce all ports for dense and sparse with some rules
  int port_;
  /// identify the trainer id using this client
  int trainerId_;

#ifndef PADDLE_DISABLE_TIMER
  uint64_t forwardbackwordTime_;
#endif

  /// map id to parameter used for decoding protobuf data
  std::unordered_map<size_t, ParameterPtr> parameterMap_;
  /// segments for all parameters that needed to sync
  std::vector<ParameterSegments> allSegments_;

  /// module for sensing sparse parameters distribution on all pservers
  std::unique_ptr<SparseParameterDistribution> sparseDistribution_;

  /// thread pool for parallelizing all connections to pservers
  std::unique_ptr<SyncThreadPool> syncThreadPool_;

  bool passFinish_;
};

}  // namespace paddle
