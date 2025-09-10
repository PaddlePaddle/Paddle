// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <uv.h>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <exception>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/common/macros.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"

namespace phi::distributed::detail {
auto constexpr MAX_KEY_LEN = 16 * 1024;
auto constexpr MAX_CONTENT_LEN = 16 * 1024 * 1024;
auto constexpr MAX_BUFFER_SIZE = size_t(4096);

class PADDLE_API SegmentedDataStream {
  std::deque<uv_buf_t> _buffers;
  size_t _buff_idx{0};
  size_t _buff_offset{0};
  size_t capacity{0};
  size_t _buff_offset_commit{0};
  size_t _read_offset{0};

 public:
  SegmentedDataStream() = default;
  void append(uv_buf_t buf);
  bool readMany(char* dest, size_t size);
  template <typename T>
  bool readValue(T& value);  // NOLINT(runtime/references)

  bool readKey(std::string& str);                // NOLINT(runtime/references)
  bool readContent(std::vector<uint8_t>& data);  // NOLINT(runtime/references)
  size_t available();
  void commit();
  void reset();
};

class PADDLE_API LibUVHandle
    : public std::enable_shared_from_this<LibUVHandle> {
 public:
  ~LibUVHandle() = default;
  std::shared_ptr<LibUVHandle> ptr();
  virtual uv_handle_t* getRawHandle() = 0;
  void close();

 protected:
  void handleAvailable();
  virtual void onClose() = 0;

 private:
  static void handleClose(uv_handle_t* uv_handle);
};

class PADDLE_API LibUVTCPSocket : public LibUVHandle {
 public:
  explicit LibUVTCPSocket(uv_loop_t* loop);
  uv_handle_t* getRawHandle() override;
  std::shared_ptr<LibUVTCPSocket> ptr();
  static std::shared_ptr<LibUVTCPSocket> getTCPSocket(uv_stream_t* handle);
  virtual void doProcess(const uv_buf_t* buf, size_t nread) {
    PADDLE_THROW(
        common::errors::Fatal("Socket subclass does not implement doProcess"));
  }
  uv_tcp_t client{};

 protected:
  void onClose() override {}
};

class PADDLE_API LibUVTCPServer : public LibUVTCPSocket {
 public:
  typedef std::function<void(int)> LibUVCallback;
  explicit LibUVTCPServer(uv_loop_t* loop)
      : LibUVTCPSocket(loop), _on_connect_callback(defaultOnConnect) {}
  void setCallback(LibUVCallback&& callback);
  static std::shared_ptr<LibUVTCPServer> createServer(uv_loop_t* loop,
                                                      std::uint16_t port,
                                                      bool useIpv6);
  std::uint16_t port() const { return _port; }
  void accept(const std::shared_ptr<LibUVTCPSocket>& socket);

 protected:
  uv_tcp_t* getRawSocket() { return &client; }
  uv_stream_t* getRawStream() {
    return reinterpret_cast<uv_stream_t*>(&client);
  }

 private:
  LibUVCallback _on_connect_callback;
  std::uint16_t _port{};

  void setSocketPort();
  static void defaultOnConnect(int status) {
    PADDLE_THROW(common::errors::Fatal(
        "Socket accepted, but onConnect callback is undefined"));
  }
  static void onNewConnection(uv_stream_t* server, int status);
};

class PADDLE_API LibUVMasterDaemon : public DaemonThread {
 public:
  explicit LibUVMasterDaemon(int port);
  // Disable copy constructor
  LibUVMasterDaemon(const LibUVMasterDaemon& other) = delete;
  // Disable move constructor
  LibUVMasterDaemon(LibUVMasterDaemon&& other) = delete;
  // Disable copy assignment operator
  LibUVMasterDaemon& operator=(const LibUVMasterDaemon& other) = delete;
  // Disable move assignment operator
  LibUVMasterDaemon& operator=(LibUVMasterDaemon&& other) = delete;
  ~LibUVMasterDaemon() override;
  void init(const std::uint16_t& port);
  // operator for key
  void set(const std::string& key, const std::vector<uint8_t>& value);
  const std::vector<uint8_t>& get(const std::string& key);
  int64_t add(const std::string& key, int64_t addVal);
  bool waitKey(const std::string& key,
               const std::shared_ptr<LibUVHandle>& client);
  bool checkKeys(const std::vector<std::string>& keys);
  // client
  void addClient(const std::shared_ptr<LibUVHandle>& client);
  void removeClient(const std::shared_ptr<LibUVHandle>& client);
  void clearWaitState(const std::shared_ptr<LibUVHandle>& client);

 protected:
  void run() override;
  void stop() override;

 private:
  uv_loop_t loop_{};
  uv_async_t _exit_handle{};
  // tcp server
  std::shared_ptr<LibUVTCPServer> _tcp_server;
  // tcp store
  std::unordered_map<std::string, std::vector<uint8_t>> _tcp_store;
  // the list of LibUVClient waiting on the key
  std::unordered_map<std::string, std::vector<std::shared_ptr<LibUVHandle>>>
      _waiting_sockets;
  // number of keys awaited
  std::unordered_map<std::shared_ptr<LibUVHandle>, size_t> _awaited_keys;
  std::unordered_set<std::shared_ptr<LibUVHandle>> _clients;
  int port_;

  static LibUVMasterDaemon& UVMasterDaemon(uv_handle_t* stream) {
    return *reinterpret_cast<LibUVMasterDaemon*>(uv_handle_get_data(stream));
  }
  static void on_new_connection(uv_stream_t* server, int status) {
    UVMasterDaemon(reinterpret_cast<uv_handle_t*>(server)).onConnect(status);
  }
  static void on_exit_request(uv_async_t* handle) {
    UVMasterDaemon(reinterpret_cast<uv_handle_t*>(handle)).onExitRequest();
  }
  void onConnect(int status);
  void onExitRequest();
  void notifyWaitingClients(const std::string& key);
};

class PADDLE_API WriteUVContent
    : public std::enable_shared_from_this<WriteUVContent> {
  std::shared_ptr<WriteUVContent> ptr() { return shared_from_this(); }
  static void writeDone(uv_write_t* req, int status);
  struct RequestData {
    std::shared_ptr<WriteUVContent> strong_self;
  };
  std::vector<uint8_t> data;
  uv_write_t req = {};
  uv_buf_t buf = {};
  std::shared_ptr<LibUVHandle> handle;

 public:
  WriteUVContent(std::vector<uint8_t>&& in_data,
                 std::shared_ptr<LibUVHandle> handle);
  ~WriteUVContent();
  void send();
};

class PADDLE_API UVWriter {
  std::vector<uint8_t> data;
  std::shared_ptr<LibUVHandle> handle;
  void* operator new(size_t);

 public:
  explicit UVWriter(std::shared_ptr<LibUVHandle> handle)
      : handle(std::move(handle)) {}
  template <typename T>
  void writeValue(T val);
  void writeVector(const std::vector<uint8_t>& val);
  void writeString(const std::string& val);
  void send();
};

class PADDLE_API LibUVClient : public LibUVTCPSocket {
  SegmentedDataStream stream;
  LibUVMasterDaemon* store;
  std::string _address{"null"};
  const std::string& address() const { return _address; }
  static void allocBuffer(uv_handle_t* handle, size_t buf_size, uv_buf_t* buf);
  static void readCallback(uv_stream_t* client,
                           ssize_t nread,
                           const uv_buf_t* buf);

 protected:
  void doProcess(const uv_buf_t* buf, size_t nread) override;
  bool doSetCommand();
  bool doGetCommand();
  bool doAddCommand();
  bool doCheckCommand();
  bool doWaitCommand();
  void onClose() override;

 public:
  explicit LibUVClient(uv_loop_t* loop, LibUVMasterDaemon* store)
      : LibUVTCPSocket(loop), store(store) {}
  void readStart();
  static std::shared_ptr<LibUVClient> make(uv_loop_t* loop,
                                           LibUVMasterDaemon* store);
  std::shared_ptr<LibUVClient> ptr();
};
}  // namespace phi::distributed::detail
