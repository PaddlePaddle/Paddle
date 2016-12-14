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

#include "paddle/utils/Util.h"

#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <thread>

#include "paddle/math/Vector.h"
#include "paddle/utils/Logging.h"

struct MessageHeader {
  int64_t dataLength;
};

class Thread {
public:
  void start();
  virtual void run() = 0;
  virtual ~Thread() {}

protected:
  std::unique_ptr<std::thread> thread_;
};

void Thread::start() {
  thread_.reset(new std::thread([this]() { this->run(); }));
}

class SocketChannel {
public:
  explicit SocketChannel(int socket) : socket_(socket) {}
  int getSocketFd() const { return socket_; }
  uint64_t readAll(void* buf, size_t size);
  uint64_t writeAll(const void* buf, size_t size);

protected:
  int socket_;
};

uint64_t SocketChannel::readAll(void* buf, size_t size) {
  uint64_t total = 0;
  while (total < size) {
    int64_t len = read(socket_, (char*)buf + total, size - total);
    if (len <= 0) {
      return total;
    }
    total += len;
  }
  return total;
}

uint64_t SocketChannel::writeAll(const void* buf, size_t size) {
  uint64_t total = 0;
  while (total < size) {
    int64_t len = write(socket_, (const char*)buf + total, size - total);
    if (len <= 0) {
      return total;
    }
    total += len;
  }
  return total;
}

class SocketWorker : public Thread {
public:
  explicit SocketWorker(int socket) : channel_(socket) {}
  virtual void run();

  // read n bytes.
  int64_t readAll(char* buf, size_t n);

  // write n bytes

protected:
  SocketChannel channel_;
  std::string buffer_;
};

class SocketServer : public Thread {
public:
  explicit SocketServer(int port)
      : port_(port), socket_(0), maxPendingConnections_(100) {}

  virtual void run();

protected:
  int port_;
  int socket_;
  int maxPendingConnections_;
};

void SocketServer::run() {
  int newsockfd;
  socklen_t clilen;
  struct sockaddr_in serv_addr, cli_addr;

  /* First call to socket() function */
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(socket_ >= 0) << "ERROR opening socket";

  /* Initialize socket structure */
  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(port_);

  /* Now bind the host address using bind() call.*/
  PCHECK(bind(socket_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0)
      << "ERROR on binding";

  /* Now start listening for the clients, here process will
   * go in sleep mode and will wait for the incoming connection
   */
  listen(socket_, maxPendingConnections_);
  clilen = sizeof(cli_addr);

  while (true) {
    /* Accept actual connection from the client */
    newsockfd = accept(socket_, (struct sockaddr*)&cli_addr, &clilen);
    PCHECK(newsockfd >= 0) << "ERROR on accept";

    SocketWorker* worker = new SocketWorker(newsockfd);
    worker->start();
  }
}

void SocketWorker::run() {
  MessageHeader header;

  while (true) {
    int64_t n = channel_.readAll(&header, sizeof(header));
    PCHECK(n == sizeof(header)) << "ERROR reading from socket";

    buffer_.resize(header.dataLength);
    n = channel_.readAll(&buffer_[0], header.dataLength);
    PCHECK(n == header.dataLength) << "ERROR reading from socket";

    /* Write a response to the client */
    n = channel_.writeAll(&header, sizeof(header));
    PCHECK(n == sizeof(header)) << "ERROR reading from socket";
    n = channel_.writeAll(buffer_.data(), buffer_.size());
    PCHECK(n == header.dataLength) << "ERROR writing to socket";
  }
}

class SocketClient {
public:
  SocketClient(const std::string& serverAddr, int serverPort);
  SocketChannel* getChannel() const { return channel_.get(); }

protected:
  std::unique_ptr<SocketChannel> channel_;
};

SocketClient::SocketClient(const std::string& serverAddr, int serverPort) {
  struct sockaddr_in serv_addr;
  struct hostent* server;

  // char buffer[256];

  /* Create a socket point */
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  PCHECK(sockfd >= 0) << "ERROR opening socket";
  server = gethostbyname(serverAddr.c_str());
  PCHECK(server) << "ERROR, no such host: " << serverAddr;

  bzero((char*)&serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char*)server->h_addr,
        (char*)&serv_addr.sin_addr.s_addr,
        server->h_length);
  serv_addr.sin_port = htons(serverPort);

  /* Now connect to the server */
  PCHECK(connect(sockfd, (sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0)
      << "ERROR connecting";

  channel_.reset(new SocketChannel(sockfd));
}

DEFINE_string(server_addr, "127.0.0.1", "Server address");
DEFINE_int64(dim, 10000000, "Data size");
DEFINE_int32(loop_time, 100000, "test loop time");

using namespace paddle;  // NOLINT

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  SocketServer server(FLAGS_port);
  server.start();
  sleep(1);

  SocketClient client(FLAGS_server_addr, FLAGS_port);

  SocketChannel* channel = client.getChannel();

  MessageHeader header;

  uint64_t dataSize = FLAGS_dim * sizeof(real);

#ifndef PADDLE_ONLY_CPU
  GpuVector gpuParam(FLAGS_dim);
  GpuVector gpuGrad(FLAGS_dim);
#else
  CpuVector gpuParam(FLAGS_dim);
  CpuVector gpuGrad(FLAGS_dim);
#endif
  CpuVector cpuParam(FLAGS_dim);
  CpuVector cpuGrad(FLAGS_dim);

  gpuParam.rand();
  gpuGrad.rand();
  cpuParam.rand();
  cpuGrad.rand();

  for (int i = 0; i < FLAGS_loop_time; ++i) {
    cpuGrad.copyFrom(gpuGrad);

    header.dataLength = dataSize;
    PCHECK(channel->writeAll(&header, sizeof(header)) == sizeof(header))
        << "Client write header error";

    PCHECK(channel->writeAll(cpuGrad.getData(), dataSize) == dataSize)
        << "Client write data error";

    /* Now read server response */
    PCHECK(channel->readAll(&header, sizeof(header)) == sizeof(header))
        << "Client read header error";

    CHECK_EQ((uint64_t)header.dataLength, dataSize);
    PCHECK(channel->readAll(cpuParam.getData(), dataSize) == dataSize)
        << "Client read data error";

    gpuParam.copyFrom(cpuParam);

    LOG_EVERY_N(INFO, 100) << "i=" << i;
  }
  exit(0);
}
