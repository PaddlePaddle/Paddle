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

#include "LightNetwork.h"

#include <map>

#include <google/protobuf/message_lite.h>

namespace paddle {

/**
 *
 * It implements the rpc framework, which launchs one thread for each
 * connection. Here define one parameter server as single TCP server
 * binding on single port. All connections share single tcp ProtoServer
 * object, each connection handles all requests from specified trainer
 * within single worker thread.
 * to accelerate bandwidth efficiency and harness multicore for pserver
 * optimization to reduce pserver latency, you could launch more port
 * for single NIC hardward with --port=N(N>1) for small cluster job.
 */
class ProtoServer : public SocketServer {
public:
  /// rdmaCpu controls the cpu affinity of RDMA server daemon,
  /// which could benifit performance. rdmaCpu = -1 means TCP
  /// is used instead of RDMA transport.
  ProtoServer(const std::string& addr, int port, int rdmaCpu = -1)
      : SocketServer(addr, port, rdmaCpu) {}

  typedef std::function<void(const google::protobuf::MessageLite& protoOut,
                             const std::vector<iovec>& outputIovs)>
      ProtoResponseCallbackEx;

  typedef std::function<void(const google::protobuf::MessageLite& protoOut)>
      ProtoResponseCallback;

  /**
   * Register a service function for this server
   * void(const ProtoIn& request,
   *      ProtoResponseCallback callback)
   * The service function process the request and call the callback
   * after it finishes the request.

   * Use macro REGISTER_SERVICE_FUNCTION as a helper
   * to simplify the use.
   */
  template <class ProtoIn>
  void registerServiceFunction(
      const std::string& funcName,
      std::function<void(const ProtoIn& request,
                         ProtoResponseCallback callback)> func);

  /**
   * Register a service function for this server
   * The signature of the service function is
   * void(const ProtoIn&,
   *      std::unique_ptr<MsgReader> msgReader,
   *      ProtoResponseCallbackEx callback)
   * The service function process the request and call the callback
   * after it finishes the request.
   * The extended service function can take extra input blocks from
   * the communication channel by reading msgReader. It can also
   * send extra blocks to the communication channel by providing
   * outputIovs as the argument for the callback function.

   * Use macro REGISTER_SERVICE_FUNCTION_EX as a helper
   * to simplify the use.
   */
  template <class ProtoIn>
  void registerServiceFunctionEx(
      const std::string& funcName,
      std::function<void(const ProtoIn&,
                         std::unique_ptr<MsgReader> msgReader,
                         ProtoResponseCallbackEx callback)> func);

protected:
  /**
   * @brief handle rpc request
   * @param[in] msgReader  Message reader for reading data from connection
   * @param[in] callback   equal to channel->writeMessage
   *
   * @note  it lookups rpc function mapping table to find function pointer,
   *        then call this function with further reading data from connection
   */
  virtual void handleRequest(std::unique_ptr<MsgReader> msgReader,
                             ResponseCallback callback);

  typedef std::function<void(std::unique_ptr<MsgReader> msgReader,
                             ResponseCallback callback)>
      ServiceFunction;

  /**
   * @brief register one RPC function in function mapping
   * @param[in] funcName  function name string
   * @param[in] func      rpc function wrapped with reading and writing data
   */
  void registerServiceFunctionImp(const std::string& funcName,
                                  ServiceFunction func);

protected:
  /// Tuning bare network overhead: the beginning of receiving request
  ThreadLocal<struct timeval> handleRequestBegin_;

  /// mapping to find rpc function while handling request
  std::map<std::string, ServiceFunction> nameToFuncMap_;
};

class ProtoClient : public SocketClient {
public:
  ProtoClient(const std::string& serverAddr,
              int serverPort,
              enum ChannelType channelType = F_TCP)
      : SocketClient(serverAddr, serverPort, channelType) {}

  /**
   * @brief Make a request to the server.
   * @param[in] funcName  request rpc function name string
   * @param[in] proto     protobuf data for sending to pserver
   * @param[in] iov       additional iov data for sending to pserver
   *
   * @note  iov provides additional blocks which need to be written to the
   *        communication channel
   */
  void send(const char* funcName,
            const google::protobuf::MessageLite& proto,
            const std::vector<iovec>& iov = std::vector<iovec>());

  /**
   * @brief receive the response from the server.
   * @param[in] proto     proto binary buffer
   *
   * @note  this must be paired with a corresponding send() call. The
   *        returned MsgReader allows the caller to receive additional
   *        blocks from the communication channel.
   */
  std::unique_ptr<MsgReader> recv(google::protobuf::MessageLite* proto);

  /// combines send() and recv()
  std::unique_ptr<MsgReader> sendAndRecv(
      const char* funcName,
      const google::protobuf::MessageLite& protoIn,
      google::protobuf::MessageLite* protoOut) {
    send(funcName, protoIn);
    return recv(protoOut);
  }

  /// combines send() and recv()
  std::unique_ptr<MsgReader> sendAndRecv(
      const char* funcName,
      const google::protobuf::MessageLite& protoIn,
      const std::vector<iovec>& iov,
      google::protobuf::MessageLite* protoOut) {
    send(funcName, protoIn, iov);
    return recv(protoOut);
  }
};

template <class>
struct service_arg_type;
/// helper class for obtaining the argument type of a service function
template <class R, class C, class Arg1, class Arg2>
struct service_arg_type<R (C::*)(const Arg1&, Arg2)> {
  typedef Arg1 _1;
};

template <class R, class C, class Arg1, class Arg2>
struct service_arg_type<R (C::*)(  // NOLINT
    const Arg1&,
    std::unique_ptr<MsgReader>,
    Arg2)> {
  typedef Arg1 _1;
};

/// register a service function to the ProtoServer
/// This should only be used within a member function of className
#define REGISTER_SERVICE_FUNCTION(className, funcName)       \
  registerServiceFunction<                                   \
      service_arg_type<decltype(&className::funcName)>::_1>( \
      #funcName,                                             \
      std::bind(&className::funcName,                        \
                this,                                        \
                std::placeholders::_1,                       \
                std::placeholders::_2))

/// register a service function to the ProtoServer
/// This should only be used within a member function of className
#define REGISTER_SERVICE_FUNCTION_EX(className, funcName)    \
  registerServiceFunctionEx<                                 \
      service_arg_type<decltype(&className::funcName)>::_1>( \
      #funcName,                                             \
      std::bind(&className::funcName,                        \
                this,                                        \
                std::placeholders::_1,                       \
                std::placeholders::_2,                       \
                std::placeholders::_3))

/// create wrapper function for parameter server high level function and
/// register the wrapper function into function mapping.
template <class ProtoIn>
void ProtoServer::registerServiceFunctionEx(
    const std::string& funcName,
    std::function<void(const ProtoIn&,
                       std::unique_ptr<MsgReader> msgReader,
                       ProtoResponseCallbackEx callback)> func) {
  auto f = [func](std::unique_ptr<MsgReader> msgReader,
                  ResponseCallback callback) {
    ProtoIn request;
    std::string str(msgReader->getNextBlockLength(), 0);
    msgReader->readNextBlock(&str[0]);
    CHECK(request.ParseFromString(str));
    auto pcob = [callback](const google::protobuf::MessageLite& response,
                           const std::vector<iovec>& outputIovs) {
      std::string out;
      CHECK(response.SerializeToString(&out));
      std::vector<iovec> iovs;
      iovs.push_back({&out[0], out.size()});
      iovs.insert(iovs.end(), outputIovs.begin(), outputIovs.end());
      callback(iovs);
    };

    func(request, std::move(msgReader), pcob);
  };

  registerServiceFunctionImp(funcName, f);
}

template <class ProtoIn>
void ProtoServer::registerServiceFunction(
    const std::string& funcName,
    std::function<void(const ProtoIn&, ProtoResponseCallback callback)> func) {
  auto f = [func](std::unique_ptr<MsgReader> msgReader,
                  ResponseCallback callback) {
    ProtoIn request;
    std::string str(msgReader->getNextBlockLength(), 0);
    msgReader->readNextBlock(&str[0]);
    CHECK(request.ParseFromString(str));
    msgReader.reset();

    auto pcob = [callback](const google::protobuf::MessageLite& response) {
      std::string out;
      CHECK(response.SerializeToString(&out));
      std::vector<iovec> iovs;
      iovs.push_back({&out[0], out.size()});
      callback(iovs);
    };

    func(request, pcob);
  };

  registerServiceFunctionImp(funcName, f);
}

}  // namespace paddle
