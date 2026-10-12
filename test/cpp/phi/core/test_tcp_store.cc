// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "glog/logging.h"
#endif

namespace phi {
namespace distributed {

TEST(MasterDaemon, init) {
  int socket = tcputils::tcp_listen("", std::to_string(0), AF_INET);
  std::unique_ptr<detail::MasterDaemon> d =
      detail::MasterDaemon::createDaemon(socket, 1, 100);
  d->start();
  printf("started to sleep 2s\n");
#ifdef _WIN32
  Sleep(2 * 1000);
#else
  usleep(2 * 1000 * 1000);
#endif
  printf("end to reset\n");

  d.reset();
}

/* now for only c compile test
TEST(TCPStore, init) {
  TCPStore store("127.0.0.1", 6170, true, 1);
  store.add("my", 3);
  auto ret1 = store.get("my");
  store.add("my", 3);
  auto ret2 = store.get("my");
  PADDLE_ENFORCE_EQ(ret1[0] + 3, ret2[0],
    paddle::errors::Fatal("result of add is not right"));
}
*/

#ifndef _WIN32
// The libuv master daemon polls its uv loop on a background thread that can
// outlive the destruction of glog's static state, so the callbacks reached
// while a connection is torn down must not use glog. They report through a
// stderr helper gated on FLAGS_v instead. These tests pin down both halves of
// that contract: the diagnostics still appear when the verbosity is raised,
// and nothing is printed at the default verbosity.
namespace {

// Captures whatever is written to fd 2 for the lifetime of the object.
class StderrCapture {
 public:
  StderrCapture() {
    snprintf(path_, sizeof(path_), "/tmp/paddle_libuv_stderr_XXXXXX");
    int fd = mkstemp(path_);
    fflush(stderr);
    saved_fd_ = dup(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
  }

  ~StderrCapture() {
    if (saved_fd_ >= 0) {
      Release();
    }
    unlink(path_);
  }

  std::string Read() {
    fflush(stderr);
    FILE* fp = fopen(path_, "rb");
    if (fp == nullptr) {
      return std::string();
    }
    std::string out;
    char buf[4096];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
      out.append(buf, n);
    }
    fclose(fp);
    return out;
  }

 private:
  void Release() {
    fflush(stderr);
    dup2(saved_fd_, STDERR_FILENO);
    close(saved_fd_);
    saved_fd_ = -1;
  }

  char path_[64] = {};
  int saved_fd_ = -1;
};

// Sets FLAGS_v for the duration of the scope.
class ScopedVerbosity {
 public:
  explicit ScopedVerbosity(int level) : saved_(FLAGS_v) { FLAGS_v = level; }
  ~ScopedVerbosity() { FLAGS_v = saved_; }

 private:
  int saved_;
};

std::uint16_t PickFreePort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  EXPECT_EQ(bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)),
            0);
  socklen_t len = sizeof(addr);
  EXPECT_EQ(getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &len),
            0);
  std::uint16_t port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

int ConnectTo(std::uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  for (int i = 0; i < 100; ++i) {
    if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) ==
        0) {
      return fd;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  close(fd);
  return -1;
}

// Waits until `needle` shows up in the captured stderr, or the budget runs out.
bool WaitFor(StderrCapture* capture, const std::string& needle, int tries) {
  for (int i = 0; i < tries; ++i) {
    if (capture->Read().find(needle) != std::string::npos) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

}  // namespace

// A peer closing the connection cleanly (FIN) reaches the UV_EOF branch.
TEST(LibUVMasterDaemon, ReportsPeerDisconnectWhenVerbose) {
  ScopedVerbosity verbosity(5);
  std::uint16_t port = PickFreePort();
  auto daemon = detail::create_libuv_tcpstore(port);
  daemon->start();

  StderrCapture capture;
  int fd = ConnectTo(port);
  ASSERT_GE(fd, 0);
  close(fd);
  EXPECT_TRUE(WaitFor(&capture, "remote peer closed the connection", 60));

  daemon.reset();
}

// Same path, default verbosity: the diagnostics must stay silent so that
// replacing VLOG with the stderr helper is not a behaviour change.
TEST(LibUVMasterDaemon, SilentOnPeerDisconnectByDefault) {
  ScopedVerbosity verbosity(0);
  std::uint16_t port = PickFreePort();
  auto daemon = detail::create_libuv_tcpstore(port);
  daemon->start();

  StderrCapture capture;
  int fd = ConnectTo(port);
  ASSERT_GE(fd, 0);
  close(fd);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::string out = capture.Read();
  EXPECT_EQ(out.find("remote peer closed the connection"), std::string::npos);
  EXPECT_EQ(out.find("read callback handler exception"), std::string::npos);

  daemon.reset();
}

// A peer aborting the connection (RST) reaches the nread < 0 branch, which
// reports the libuv error name alongside the code.
TEST(LibUVMasterDaemon, ReportsReadErrorWhenVerbose) {
  ScopedVerbosity verbosity(5);
  std::uint16_t port = PickFreePort();
  auto daemon = detail::create_libuv_tcpstore(port);
  daemon->start();

  StderrCapture capture;
  int fd = ConnectTo(port);
  ASSERT_GE(fd, 0);
  struct linger abort_on_close = {1, 0};
  setsockopt(
      fd, SOL_SOCKET, SO_LINGER, &abort_on_close, sizeof(abort_on_close));
  // Give the server a chance to arm its read callback before the RST lands.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  close(fd);
  EXPECT_TRUE(WaitFor(&capture, "read callback handler exception", 60));

  daemon.reset();
}

// An over-long key makes SegmentedDataStream::readKey throw, which is caught
// in readCallback and reported through the same helper at verbosity 2.
TEST(LibUVMasterDaemon, ReportsMalformedMessageWhenVerbose) {
  ScopedVerbosity verbosity(2);
  std::uint16_t port = PickFreePort();
  auto daemon = detail::create_libuv_tcpstore(port);
  daemon->start();

  StderrCapture capture;
  int fd = ConnectTo(port);
  ASSERT_GE(fd, 0);
  // SET command followed by a key length well past MAX_KEY_LEN (16 * 1024).
  std::vector<char> payload(sizeof(std::uint32_t) + sizeof(std::uint64_t));
  auto command = static_cast<std::uint32_t>(Command::SET);
  std::uint64_t key_len = 1024 * 1024;
  memcpy(payload.data(), &command, sizeof(command));
  memcpy(payload.data() + sizeof(command), &key_len, sizeof(key_len));
  ASSERT_EQ(write(fd, payload.data(), payload.size()),
            static_cast<ssize_t>(payload.size()));
  EXPECT_TRUE(
      WaitFor(&capture, "failed to process incoming client message", 60));

  close(fd);
  daemon.reset();
}
#endif

}  // namespace distributed
}  // namespace phi
