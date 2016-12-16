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

#include "SocketChannel.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "paddle/utils/Thread.h"

struct sxi_socket;

namespace paddle {

class SocketWorker;

/**
 * @brief class for holding all parameters processing for current port
 *
 * @note  each parameter server inherits from one socket server, each
 *        server contains serveral woker threads which are to parallelize
 *        the processing of computation, but share some common datas stored
 *        in child class of socketserver.
 */
class SocketServer : public Thread {
  // rdmaCpu controls the cpu affinity of RDMA server daemon,
  // which could benifit performance. rdmaCpu = -1 means TCP
  // is used instead of RDMA transport.
public:
  SocketServer(const std::string& addr, int port, int rdmaCpu);
  ~SocketServer();

  virtual void run();

  typedef std::function<void(const std::vector<iovec>& outputIovs)>
      ResponseCallback;

protected:
  //
  // The derived class needs to implement this function
  // to handle the request received by SocketWorker
  // The request is encapsulated by MsgReader, which contains
  // a set of blocks.
  virtual void handleRequest(std::unique_ptr<MsgReader> msgReader,
                             ResponseCallback callback) = 0;

  std::unique_ptr<SocketChannel> createChannel(int sock,
                                               const std::string& peerName) {
    return std::unique_ptr<SocketChannel>(new SocketChannel(sock, peerName));
  }
  std::unique_ptr<SocketChannel> createChannel(struct sxi_sock* sock,
                                               const std::string& peerName) {
    return std::unique_ptr<SocketChannel>(new SocketChannel(sock, peerName));
  }

  friend class SocketWorker;

private:
  void rdmaServer();
  void tcpServer();

  void detach() {}  // detach accept thread is forbidden

protected:
  enum ChannelType tcpRdma_;
  // for rdma
  int rdmaCpu_;
  std::string rdmaUri_;
  sxi_socket* rdmaSocket_;
  // for tcp
  int port_;
  std::string addr_;
  int socket_;
  int maxPendingConnections_;
  bool stopping_;
};

/**
 * @brief class for holding one connection from one trainer
 *
 * @note  all parameter processing will run in the context of this worker
 */
class SocketWorker : public Thread {
public:
  SocketWorker(std::unique_ptr<SocketChannel>&& channel, SocketServer* server)
      : channel_(std::move(channel)), server_(server) {}

  virtual ~SocketWorker() {}

  virtual void run();

protected:
  std::unique_ptr<SocketChannel> channel_;
  SocketServer* server_;
  enum ChannelType tcpRdma_;
};

/**
 * @brief class for providing rdma client deamon thread
 *
 * @note  the deamons are required by sock like rdam library. Here
 *        use singleton model for daemons. Each deamon hosts in
 *        single cpu core for better load balance performance
 */
class RdmaClientDaemons {
private:
  RdmaClientDaemons();

  static std::unique_ptr<RdmaClientDaemons> daemons_;

public:
  static RdmaClientDaemons* get() {
    std::call_once(RdmaClientDaemons::initDataFlag_,
                   &RdmaClientDaemons::getInstance);

    return daemons_.get();
  }

  struct sxi_socket* selectDaemon() {
    int cpu = curCpu_;
    curCpu_ = (curCpu_ + 1) % onlineCpus_;

    LOG(INFO) << "select daemon " << cpu << "onlineCpus_ " << onlineCpus_;
    return rdmaClientSocket_[cpu];
  }

  ~RdmaClientDaemons();

public:
  friend class SocketClient;

private:
  static std::once_flag initDataFlag_;
  static void getInstance() {
    if (!daemons_.get()) daemons_.reset(new RdmaClientDaemons());
  }

  std::vector<struct sxi_socket*> rdmaClientSocket_;
  std::atomic<int> curCpu_;
  int onlineCpus_;
};

/**
 * @brief management for client connection which are from trainers
 *
 * @note  it contains one channel descriptor which used to write and
 *        read data
 */
class SocketClient {
public:
  SocketClient(const std::string& serverAddr,
               int serverPort,
               enum ChannelType channelType);

  SocketChannel* getChannel() { return channel_.get(); }

protected:
  std::unique_ptr<SocketChannel> channel_;
  struct sxi_socket* socketDaemon_;
  enum ChannelType tcpRdma_;

private:
  void RdmaClient(const std::string& serverAddr, int serverPort);
  void TcpClient(const std::string& serverAddr, int serverPort);
};

std::string getIpAddr(std::string& device);
void setOption(int sockfd);

}  // namespace paddle
