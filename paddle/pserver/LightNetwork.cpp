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

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <chrono>

#include <arpa/inet.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <sys/ioctl.h>
#include <sstream>

#include "LightNetwork.h"
#include "RDMANetwork.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

/// quick ack can reduce the latency of small message
DEFINE_bool(small_messages,
            false,
            "if message size is small, recommend set it True to enable quick "
            "ack and no delay");

/// reasonable sock_send_buf_size can control the traffic injected into switch
/// network. Injecting too many data into traffic could cause packets loss which
/// cause long latency and degrade the efficiency of communication.
DEFINE_int32(sock_send_buf_size,
             1024 * 1024 * 40,
             "restrict sock send buff size, can reduce network congestion if "
             "set carefully");

/// reasonable size can hold bursted packets and reduce packets loss
DEFINE_int32(sock_recv_buf_size,
             1024 * 1024 * 40,
             "restrict sock recv buff size");

namespace paddle {

/**
 * @brief get ip address from interface name
 *
 * @param[in] device device interface name
 */
std::string getIpAddr(std::string &device) {
  int sock;
  struct sockaddr_in sin;
  struct ifreq ifr;

  sock = socket(AF_INET, SOCK_DGRAM, 0);
  CHECK(sock >= 0) << "Create socket error.";

  strncpy(ifr.ifr_name, device.c_str(), IFNAMSIZ);
  ifr.ifr_name[IFNAMSIZ - 1] = 0;

  CHECK_GE(ioctl(sock, SIOCGIFADDR, &ifr), 0);
  memcpy(&sin, &ifr.ifr_addr, sizeof(sin));
  close(sock);
  return std::string(inet_ntoa(sin.sin_addr));
}

/**
 * @brief set sock option
 *
 * @param[in] sockfd sock file descriptor
 *
 * @note adjust some default sock option for better performance
 */
void setOption(int sockfd) {
#if !defined(__APPLE__) && !defined(__OSX__)
  int sendSize = FLAGS_sock_send_buf_size;
  int recvSize = FLAGS_sock_recv_buf_size;
  CHECK_GE(
      setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recvSize, sizeof(recvSize)),
      0);
  CHECK_GE(
      setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sendSize, sizeof(sendSize)),
      0);
#endif

  if (FLAGS_small_messages) {
    int optval = 1;
    CHECK_GE(
        setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)),
        0);
#ifdef TCP_QUICKACK
    optval = 1;
    CHECK_GE(
        setsockopt(sockfd, IPPROTO_TCP, TCP_QUICKACK, &optval, sizeof(optval)),
        0);
#endif
  }
  int reuse = 1;
  CHECK_GE(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)),
           0);
}

/**
 * @brief class constructor for SocketServer
 * @param[in] addr sock bind address
 * @param[in] port sock bind port
 * @param[in] rdmaCpu rdma sock bind cpu core
 *
 * @note start one socket server which hosts parameter server process.
 *       rdmaCpu is passed to rdma deamon for better performance, and
 *       start tcp socket instead of rdma socket if rdmaCpu is equal
 *       to -1. Each trainer process starts one connection to one socket
 *       server, and use --ports_num to build more connections to harness
 *       fat communication channel if necessary.
 *       each connection is controlled by single thread with blocking
 *       read and write.
 */
SocketServer::SocketServer(const std::string &addr, int port, int rdmaCpu)
    : port_(port), addr_(addr), stopping_(false) {
  if (rdmaCpu == -1) {
    tcpRdma_ = F_TCP;
    socket_ = 0;
    maxPendingConnections_ = 100;
  } else {
    tcpRdma_ = F_RDMA;
    rdmaCpu_ = rdmaCpu;
    rdmaSocket_ = 0;

    std::stringstream ss;
    ss << port;
    rdmaUri_ = "rdma://" + addr + ":" + ss.str();
  }

  /// trigger to initialize RDMA lib
  PCHECK(RdmaClientDaemons::get()) << "initilizate RDMA failed\n";
}

SocketServer::~SocketServer() {
  stopping_ = true;
  /// trigger accept thread to stop
  {
    SocketClient trigger(addr_.empty() ? "127.0.0.1" : addr_, port_, tcpRdma_);
  }
  this->join();
}

/**
 * @brief start one tcp server which hosts parameter server
 *
 * @note do tcp socket bind and listen. it will spawn one thread
 *       for each connection
 */
void SocketServer::tcpServer() {
  int newsockfd;
  socklen_t clilen;
  struct sockaddr_in serv_addr, cli_addr;
  struct hostent *server;

  /// First call to socket() function
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(socket_ >= 0) << "ERROR opening socket";

  /// Initialize socket structure
  bzero((char *)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port_);
  if (!addr_.empty()) {
    server = gethostbyname(addr_.c_str());
    PCHECK(server) << "ERROR, no such host: " << addr_;
    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
  } else {
    serv_addr.sin_addr.s_addr = INADDR_ANY;
  }

  setOption(socket_);

  /// Now bind the host address using bind() call.
  PCHECK(bind(socket_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >= 0)
      << "ERROR on binding " << addr_;

  /// Now start listening for the clients, here process will
  /// go in sleep mode and will wait for the incoming connection
  listen(socket_, maxPendingConnections_);
  clilen = sizeof(cli_addr);

  while (true) {
    /// Accept actual connection from the client
    newsockfd = accept(socket_, (struct sockaddr *)&cli_addr, &clilen);
    if (stopping_) {
      break;
    }
    PCHECK(newsockfd >= 0) << "ERROR on accept";
    constexpr int kPeerNameLen = 128;
    char peerName[kPeerNameLen];
    CHECK(inet_ntop(AF_INET, &cli_addr.sin_addr, peerName, kPeerNameLen));

    SocketWorker *worker =
        new SocketWorker(createChannel(newsockfd, std::string(peerName)), this);
    worker->start();
    worker->detach();
  }
  close(socket_);
  LOG(INFO) << "pserver accept thread finish, addr=" << addr_
            << " port=" << port_;
}

/**
 * @brief start one rdma server which hosts parameter server
 *
 * @note do rdma bind and listen, which calling self-defined socket
 *       like rdma library. it will spawn one thread for each connection
 */
void SocketServer::rdmaServer() {
  struct sxi_sock *newsock;

  /// First call to socket() function
  rdmaSocket_ = rdma::ssocket(rdmaCpu_);
  PCHECK(rdmaSocket_) << "ERROR opening RDMA socket";

  PCHECK(rdma::bind(rdmaSocket_, rdmaUri_.c_str()) == 0)
      << "ERROR bind RDMA socket";

  /// Now start listening for the clients, here process will
  /// go in sleep mode and will wait for the incoming connection
  PCHECK(rdma::listen(rdmaSocket_) == 0) << "ERROR listen RDMA socket";

  while (true) {
    /// Accept actual connection from the client
    newsock = rdma::accept(rdmaSocket_);
    if (stopping_) {
      break;
    }
    PCHECK(newsock) << "ERROR on accept";

    constexpr int kPeerNameLen = 128;
    char peerName[kPeerNameLen];

    struct sockaddr_in *saddr = rdma::getSourceAddress(newsock);
    CHECK(inet_ntop(AF_INET, &saddr->sin_addr, peerName, kPeerNameLen));

    SocketWorker *worker =
        new SocketWorker(createChannel(newsock, std::string(peerName)), this);
    worker->start();
    worker->detach();
  }
  rdma::close(rdmaSocket_);
  LOG(INFO) << "pserver accept thread finish, rdma uri=" << rdmaUri_;
}

/**
 * @brief start a socket server
 *
 * @note framework for starting socket server
 */
void SocketServer::run() {
  if (tcpRdma_ == F_TCP) {
    LOG(INFO) << "tcp server start ";
    tcpServer();
  } else if (tcpRdma_ == F_RDMA) {
    LOG(INFO) << "rdma server start ";
    rdmaServer();
  }
}

/**
 * @brief class constructor for rdma client deamons
 *
 * @note  automatically start several client deamons for better performance
 */
std::unique_ptr<RdmaClientDaemons> RdmaClientDaemons::daemons_ = nullptr;
std::once_flag RdmaClientDaemons::initDataFlag_;

RdmaClientDaemons::RdmaClientDaemons() {
  if (FLAGS_rdma_tcp == "rdma") {
    rdma::init();

    struct sxi_socket *socket;
    onlineCpus_ = rdma::numCpus();
    for (auto i = 0; i < onlineCpus_; i++) {
      socket = rdma::csocket(i);
      PCHECK(socket) << "ERROR open client socket daemon";

      rdmaClientSocket_.push_back(socket);
    }
    LOG(INFO) << "RDMA client daemons started, onlineCpus_:" << onlineCpus_;
    /// round robin scheduler for new connection
    curCpu_ = 0;
    /// wait daemons to start completely.
    sleep(2);
  }
}

RdmaClientDaemons::~RdmaClientDaemons() {
  if (FLAGS_rdma_tcp == "rdma") {
    for (auto i = 0; i < onlineCpus_; i++) {
      rdma::close(rdmaClientSocket_[i]);
    }
    LOG(INFO) << "RDMA client daemons is destoryed, onlineCpus_ "
              << onlineCpus_;
  }
}

/**
 * @brief worker thread main context
 *
 * @note  each connection from client(trainer) is controlled by single worker
 *        thread, which is for handling all parameter server requests
 */
void SocketWorker::run() {
  LOG(INFO) << "worker started, peer = " << channel_->getPeerName();

  std::vector<iovec> inputIovs;

  while (true) {
    std::unique_ptr<MsgReader> msgReader = channel_->readMessage();
    if (!msgReader) {
      break;
    }

    auto callback = [this](const std::vector<iovec> &outputIovs) {
      channel_->writeMessage(outputIovs);
    };

    server_->handleRequest(std::move(msgReader), callback);
  }

  LOG(INFO) << "worker begin to finish, peer = " << channel_->getPeerName();
  delete this;
}

/**
 * @brief start one tcp connection to tcp server
 * @param[in] serverAddr  tcp server ip
 * @param[in] serverPort  tcp server port
 *
 * @note each object contains one channel which accept byte stream
 */
void SocketClient::TcpClient(const std::string &serverAddr, int serverPort) {
  struct sockaddr_in serv_addr;
  struct hostent *server;

  int errRet;  // temp for gethostbyname_r

  /// Create a socket point
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(sockfd >= 0) << "ERROR opening socket";

#if defined(__OSX__) || defined(__APPLE__)
  server = getipnodebyname(serverAddr.c_str(), AF_INET, AI_DEFAULT, &errRet);
  CHECK_NE(HOST_NOT_FOUND, errRet) << "ERROR, no such host: " << serverAddr
                                   << " ret = " << errRet;
  CHECK(server) << "getipnodebyname error!";
#else
  struct hostent hostinfo;
  char buf[1024];  // temp for gethostbyname_r
  CHECK_EQ(
      0,
      gethostbyname_r(
          serverAddr.c_str(), &hostinfo, buf, sizeof(buf), &server, &errRet))
      << "ERROR, no such host: " << serverAddr << " ret = " << errRet;
  CHECK(server) << "gethostbyname_r error!";
#endif

  bzero((char *)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char *)server->h_addr,
        (char *)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(serverPort);

  setOption(sockfd);

  /// Now connect to the server
  int retry_second = 0;
  int error = 0;
  do {
    error = connect(sockfd, (sockaddr *)&serv_addr, sizeof(serv_addr));
    if (error == ECONNREFUSED) {
      LOG(WARNING) << "connection refused by pserver, try again!";
      if (retry_second++ >= 7) {
        LOG(FATAL) << "connection refused by pserver, maybe pserver failed!";
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } else {
      PCHECK(error >= 0) << "ERROR connecting to " << serverAddr;
    }
  } while (error == ECONNREFUSED);

  channel_.reset(new SocketChannel(sockfd, serverAddr));
  tcpRdma_ = F_TCP;
}

/**
 * @brief start one RDMA connection to rdma server
 * @param[in] serverAddr  rdma server ip
 * @param[in] serverPort  rdma server port
 *
 * @note  each object contains one channel which accept byte stream
 *        for rdma, low level sock also provide byte stream api.
 */
void SocketClient::RdmaClient(const std::string &serverAddr, int serverPort) {
  struct sxi_sock *sock;

  std::stringstream ss;
  ss << serverPort;

  std::string rdmaUri = "rdma://" + serverAddr + ":" + ss.str();

  RdmaClientDaemons *daemons = RdmaClientDaemons::daemons_->get();
  socketDaemon_ = daemons->selectDaemon();

  /// connect to server with socket daemon
  sock = rdma::connect(socketDaemon_, rdmaUri.c_str());
  PCHECK(sock) << "ERROR connect to server" << rdmaUri;

  std::vector<std::string> seg;
  str::split(rdmaUri, '/', &seg);
  std::string server = seg.at(seg.size() - 1);
  channel_.reset(new SocketChannel(sock, server));
  tcpRdma_ = F_RDMA;
}

/**
 * @brief class constructor
 * @param[in] serverAddr pserver ip address
 * @param[in] serverPort pserver port
 * @param[in] ChannelType F_TCP or F_RDMA
 *
 * @note  responsible for building one connection to specified pserver port
 */
SocketClient::SocketClient(const std::string &serverAddr,
                           int serverPort,
                           enum ChannelType channelType) {
  if (channelType == F_RDMA)
    RdmaClient(serverAddr, serverPort);
  else
    TcpClient(serverAddr, serverPort);
}

}  // namespace paddle
