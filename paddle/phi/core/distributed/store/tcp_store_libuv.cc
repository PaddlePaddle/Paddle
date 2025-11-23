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
#include "glog/logging.h"

#include "paddle/phi/core/distributed/store/tcp_store_libuv.h"

namespace phi::distributed::detail {

// SegmentedDataStream
void SegmentedDataStream::append(uv_buf_t buf) {
  if (buf.len == 0) {
    free(buf.base);
  } else {
    capacity += buf.len;
    _buffers.push_back(buf);
  }
}

bool SegmentedDataStream::readMany(char* dest, size_t size) {
  if (available() < size) {
    return false;
  }

  size_t remaining = size;
  char* write_base = dest;
  while (remaining > 0) {
    auto to_read = std::min(_buffers[_buff_idx].len - _buff_offset, remaining);
    ::memcpy(write_base, _buffers[_buff_idx].base + _buff_offset, to_read);
    _buff_offset += to_read;
    remaining -= to_read;
    write_base += to_read;
    if (_buff_offset >= _buffers[_buff_idx].len) {
      _buff_offset = 0;
      ++_buff_idx;
      if (_buff_idx >= _buffers.size() && remaining > 0) {
        PADDLE_THROW(common::errors::Fatal(paddle::string::Sprintf(
            "Read operation exceeds buffer boundary. ",
            "buffer index: %d, available: %d, remaining: %d",
            _buff_idx,
            _buffers.size(),
            remaining)));
      }
    }
  }
  _read_offset += size;
  return true;
}

template <typename T>
bool SegmentedDataStream::readValue(T& value) {
  return readMany(reinterpret_cast<char*>(&value), sizeof(T));
}

bool SegmentedDataStream::readKey(std::string& str) {
  uint64_t size = 0;
  if (!readValue(size)) return false;
  PADDLE_ENFORCE_LE(size,
                    phi::distributed::detail::MAX_KEY_LEN,
                    common::errors::InvalidArgument(paddle::string::Sprintf(
                        "Key size validation failed. size: %d, max: %d",
                        size,
                        phi::distributed::detail::MAX_KEY_LEN)));

  if (available() < size) return false;
  str.resize(size);
  return readMany(reinterpret_cast<char*>(str.data()), size);
}

bool SegmentedDataStream::readContent(std::vector<uint8_t>& data) {
  uint64_t size = 0;
  if (!readValue(size)) return false;
  auto size_in_bytes = size * sizeof(uint8_t);
  PADDLE_ENFORCE_LE(size_in_bytes,
                    MAX_CONTENT_LEN,
                    common::errors::InvalidArgument(paddle::string::Sprintf(
                        "Content size validation failed. size: %d, max: %d",
                        size_in_bytes,
                        MAX_CONTENT_LEN)));

  if (available() < size_in_bytes) return false;
  data.resize(size);
  return readMany(reinterpret_cast<char*>(data.data()), size_in_bytes);
}

size_t SegmentedDataStream::available() { return capacity - _read_offset; }

void SegmentedDataStream::commit() {
  if (_buff_idx >= _buffers.size() || _buff_offset >= _buffers[_buff_idx].len) {
    _buff_offset = 0;
    if (_buff_idx < _buffers.size()) ++_buff_idx;
  }

  for (size_t i = 0; i < _buff_idx; ++i) {
    free(_buffers[0].base);
    capacity -= _buffers[0].len;
    _buffers.pop_front();
  }
  _buff_idx = 0;
  _read_offset = _buff_offset_commit = _buff_offset;
}

void SegmentedDataStream::reset() {
  _buff_idx = 0;
  _read_offset = _buff_offset = _buff_offset_commit;
}

// LibUVHandle
std::shared_ptr<LibUVHandle> LibUVHandle::ptr() { return shared_from_this(); }

void LibUVHandle::close() {
  if (uv_is_closing(getRawHandle())) {
    return;
  }
  uv_close(getRawHandle(), handleClose);
}

void LibUVHandle::handleAvailable() {
  uv_handle_set_data(getRawHandle(), this);
}

void LibUVHandle::handleClose(uv_handle_t* uv_handle) {
  auto h = reinterpret_cast<LibUVHandle*>(uv_handle_get_data(uv_handle));
  h->onClose();
}

//   ==== LibUVTCPSocket ====
LibUVTCPSocket::LibUVTCPSocket(uv_loop_t* loop) {
  uv_tcp_init(loop, &client);
  if (int err = uv_tcp_nodelay(&client, 1)) {
    VLOG(2) << "The no-delay option is unavailable. err: " << err;
  }
}

uv_handle_t* LibUVTCPSocket::getRawHandle() {
  return reinterpret_cast<uv_handle_t*>(&client);
}

std::shared_ptr<LibUVTCPSocket> LibUVTCPSocket::ptr() {
  return std::static_pointer_cast<LibUVTCPSocket>(shared_from_this());
}

std::shared_ptr<LibUVTCPSocket> LibUVTCPSocket::getTCPSocket(
    uv_stream_t* handle) {
  auto h = reinterpret_cast<LibUVTCPSocket*>(
      uv_handle_get_data(reinterpret_cast<uv_handle_t*>(handle)));
  return h->ptr();
}

//   LibUVTCPServer
void LibUVTCPServer::setCallback(LibUVCallback&& callback) {
  _on_connect_callback = std::move(callback);
}

std::shared_ptr<LibUVTCPServer> LibUVTCPServer::createServer(uv_loop_t* loop,
                                                             std::uint16_t port,
                                                             bool useIpv6) {
  auto res = std::make_shared<LibUVTCPServer>(loop);
  res->handleAvailable();
  try {
    struct sockaddr_storage addr {};
    int uv_res = 0;
    if (useIpv6) {
      uv_res = uv_ip6_addr("::", port, (struct sockaddr_in6*)&addr);
    } else {
      uv_res = uv_ip4_addr("0.0.0.0", port, (struct sockaddr_in*)&addr);
    }
    PADDLE_ENFORCE_EQ(uv_res,
                      0,
                      common::errors::InvalidArgument(paddle::string::Sprintf(
                          "sockaddr parsing failure. port: %d, useIpv6:%d, "
                          "code: %d, name: %s, message: %s",
                          port,
                          useIpv6,
                          uv_res,
                          uv_err_name(uv_res),
                          uv_strerror(uv_res))));

    uv_res =
        uv_tcp_bind(res->getRawSocket(), (const struct ::sockaddr*)&addr, 0);
    PADDLE_ENFORCE_EQ(
        uv_res,
        0,
        common::errors::InvalidArgument(paddle::string::Sprintf(
            "Bind operation failed for the server socket. port: %d, "
            "useIpv6: %d, code: %d, name: %s, message: %s",
            port,
            useIpv6,
            uv_res,
            uv_err_name(uv_res),
            uv_strerror(uv_res))));

    uv_res = uv_listen(
        res->getRawStream(), FLAGS_tcp_max_syn_backlog, onNewConnection);
    PADDLE_ENFORCE_EQ(
        uv_res,
        0,
        common::errors::InvalidArgument(paddle::string::Sprintf(
            "Server socket unable to listen on local network interfaces. "
            "port: %d, useIpv6: %d, code: %d, name: %s, message: %s",
            port,
            useIpv6,
            uv_res,
            uv_err_name(uv_res),
            uv_strerror(uv_res))));
    res->setSocketPort();
  } catch (std::exception& ex) {
    res->close();
    throw;
  }
  return res;
}

void LibUVTCPServer::accept(const std::shared_ptr<LibUVTCPSocket>& socket) {
  int res = uv_accept(getRawStream(),
                      reinterpret_cast<uv_stream_t*>(socket->getRawHandle()));
  PADDLE_ENFORCE_EQ(
      res,
      0,
      common::errors::InvalidArgument(paddle::string::Sprintf(
          "Socket accept operation failed. code: %d, name: %s, message: %s",
          res,
          uv_err_name(res),
          uv_strerror(res))));
}

void LibUVTCPServer::setSocketPort() {
  sockaddr_storage addr_s{};
  int addr_len = sizeof(addr_s);
  if (uv_tcp_getsockname(reinterpret_cast<uv_tcp_t*>(getRawStream()),
                         reinterpret_cast<::sockaddr*>(&addr_s),
                         &addr_len) != 0) {
    throw std::runtime_error("the port number cannot be retrieved.");
  }
  if (addr_s.ss_family == AF_INET) {
    _port = ntohs(reinterpret_cast<sockaddr_in*>(&addr_s)->sin_port);
  } else {
    _port = ntohs(reinterpret_cast<sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

void LibUVTCPServer::onNewConnection(uv_stream_t* server, int status) {
  auto h = reinterpret_cast<LibUVTCPServer*>(
      uv_handle_get_data(reinterpret_cast<uv_handle_t*>(server)));
  h->_on_connect_callback(status);
}

// WriteUVContent
WriteUVContent::WriteUVContent(std::vector<uint8_t>&& in_data,
                               std::shared_ptr<LibUVHandle> handle)
    : data(std::move(in_data)), handle(std::move(handle)) {
  uv_req_set_data(reinterpret_cast<uv_req_t*>(&req), new RequestData());
}

void WriteUVContent::writeDone(uv_write_t* req, int status) {
  auto data_ptr = static_cast<RequestData*>(
      uv_req_get_data(reinterpret_cast<uv_req_t*>(req)));
  if (!data_ptr) return;

  auto self = std::move(data_ptr->strong_self);
  delete data_ptr;
  uv_req_set_data(reinterpret_cast<uv_req_t*>(req), nullptr);
  if (self && status) {
    VLOG(2) << "Write to client failed. code:" << status
            << " desc:" << uv_strerror(status)
            << " name:" << uv_err_name(status);
    self->handle->close();
  }
}

WriteUVContent::~WriteUVContent() {
  // safely clean up pending request data
  if (auto data = static_cast<RequestData*>(
          uv_req_get_data(reinterpret_cast<uv_req_t*>(&req)))) {
    delete data;
    uv_req_set_data(reinterpret_cast<uv_req_t*>(&req), nullptr);
  }
}

void WriteUVContent::send() {
  if (data.empty()) return;
  buf = uv_buf_init(reinterpret_cast<char*>(data.data()), data.size());
  int res = uv_write(&req,
                     reinterpret_cast<uv_stream_t*>(handle->getRawHandle()),
                     &buf,
                     1,
                     writeDone);
  if (res) {
    VLOG(2) << "Write failed. code:" << res << " desc:" << uv_strerror(res)
            << " name:" << uv_err_name(res);
    handle->close();
  } else {
    auto data_ptr = static_cast<RequestData*>(
        uv_req_get_data(reinterpret_cast<uv_req_t*>(&req)));
    if (data_ptr) {
      data_ptr->strong_self = shared_from_this();
    }
  }
}

// UVWriter
template <typename T>
void UVWriter::writeValue(T val) {
  uint8_t* val_ptr = reinterpret_cast<uint8_t*>(&val);
  data.insert(data.end(), val_ptr, val_ptr + sizeof(T));
}

void UVWriter::writeVector(const std::vector<uint8_t>& val) {
  writeValue<uint64_t>(val.size());
  data.insert(data.end(), val.begin(), val.end());
}

void UVWriter::writeString(const std::string& val) {
  writeValue<uint64_t>(val.size());
  data.insert(data.end(), val.data(), val.data() + val.size());
}

void UVWriter::send() {
  auto wd = std::make_shared<WriteUVContent>(std::move(data), handle);
  wd->send();
}

// LibUVClient
void LibUVClient::allocBuffer(uv_handle_t* handle,
                              size_t buf_size,
                              uv_buf_t* buf) {
  buf_size = std::min(buf_size, MAX_BUFFER_SIZE);
  buf->base = reinterpret_cast<char*>(malloc(buf_size));
  buf->len = buf_size;
}

void LibUVClient::readCallback(uv_stream_t* client,
                               ssize_t nread,
                               const uv_buf_t* buf) {
  auto uv_socket = LibUVTCPSocket::getTCPSocket(client);
  if (nread > 0) {
    try {
      uv_socket->doProcess(buf, nread);
      return;
    } catch (std::exception& ex) {
      VLOG(2) << "Failed to process incoming client message: " << ex.what();
      uv_socket->close();
    }
  } else if (nread == UV_EOF) {
    // EOF
    VLOG(5) << "Remote peer closed the connection.";
    uv_socket->close();
  } else if (nread < 0) {
    // error and EOF
    VLOG(5) << "Read callback handler exception. code:" << nread
            << " desc:" << uv_strerror(nread) << " name:" << uv_err_name(nread);
    uv_socket->close();
  }
  free(buf->base);
}

void LibUVClient::doProcess(const uv_buf_t* buf, size_t nread) {
  auto tmp = *buf;
  tmp.len = nread;
  stream.append(tmp);

  VLOG(5) << "process: " << std::string(buf->base, nread)
          << ", nread: " << nread;
  while (true) {
    stream.reset();
    uint32_t command = -1;
    if (!stream.readValue(command)) break;

    VLOG(5) << "Client parse command" << command;
    switch ((Command)command) {
      case Command::ADD:
        if (!doAddCommand()) return;
        break;
      case Command::GET:
        if (!doGetCommand()) return;
        break;
      case Command::CHECK:
        if (!doCheckCommand()) return;
        break;
      case Command::SET:
        if (!doSetCommand()) return;
        break;
      case Command::WAIT:
        if (!doWaitCommand()) return;
        break;
      default:
        VLOG(4) << "invalid command from Client, command: " << command;
        close();
        return;
    }
    stream.commit();
  }
}

bool LibUVClient::doSetCommand() {
  std::string key;
  if (!stream.readKey(key)) return false;

  std::vector<uint8_t> newData;
  if (!stream.readContent(newData)) return false;
  VLOG(7) << "set key:" << key << " address:" << this->address();
  store->set(key, newData);
  return true;
}

bool LibUVClient::doGetCommand() {
  std::string key;
  if (!stream.readKey(key)) return false;

  VLOG(7) << "get key: " << key << " address:" << this->address();
  const auto& data = store->get(key);
  UVWriter sw(ptr());
  sw.writeVector(data);
  sw.send();
  return true;
}

bool LibUVClient::doAddCommand() {
  std::string key;
  if (!stream.readKey(key)) return false;
  int64_t addVal = 0;
  if (!stream.readValue(addVal)) return false;

  addVal = store->add(key, addVal);
  VLOG(7) << "add key:" << key << " val: " << addVal
          << " address:" << this->address();
  UVWriter sw(ptr());
  sw.writeValue(addVal);
  sw.send();
  return true;
}

bool LibUVClient::doCheckCommand() {
  std::string key;
  if (!stream.readKey(key)) return false;

  VLOG(7) << "check key:" << key << " address:" << this->address();
  std::vector<std::string> keys = {key};
  UVWriter sw(ptr());
  if (store->checkKeys(keys)) {
    sw.writeValue(ReplyType::READY);
  } else {
    sw.writeValue(ReplyType::NOT_READY);
  }
  sw.send();
  return true;
}

bool LibUVClient::doWaitCommand() {
  std::string key;
  if (!stream.readKey(key)) return false;

  VLOG(7) << "wait key:  " << key << " address:" << this->address();
  if (store->waitKey(key, ptr())) {
    UVWriter sw(ptr());
    sw.writeValue(ReplyType::STOP_WAIT);
    sw.send();
    VLOG(7) << "wait send:  " << key;
  }
  return true;
}

void LibUVClient::onClose() { store->removeClient(ptr()); }

PADDLE_API std::string fmtSockAddr(const struct ::sockaddr* addr,
                                   socklen_t len) {
  char host[NI_MAXHOST], port[NI_MAXSERV];  // NOLINT
  int flags = NI_NUMERICSERV;
  int err =
      ::getnameinfo(addr, len, host, sizeof(host), port, sizeof(port), flags);
  if (err) {
    VLOG(1) << "Cannot resolve hostname, fallback to numeric. Error: " << err;
    // fallback to numeric
    flags |= NI_NUMERICHOST;
    err =
        ::getnameinfo(addr, len, host, sizeof(host), port, sizeof(port), flags);
    if (err) {
      VLOG(1) << "Numeric address resolution failed. Error: " << err;
      return "?UNKNOWN?";
    }
  }
  switch (addr->sa_family) {
    case AF_INET:
      return paddle::string::Sprintf("%s:%s", host, port);
    case AF_INET6:
      return paddle::string::Sprintf("[%s]:%s", host, port);
    default:
      return paddle::string::Sprintf("[%s]:%s", host, port);
  }
}

void LibUVClient::readStart() {
  struct ::sockaddr_storage addr {};
  int addrLen{sizeof(struct ::sockaddr_storage)};

  if (int err = uv_tcp_getpeername(
          &client, reinterpret_cast<struct ::sockaddr*>(&addr), &addrLen)) {
    VLOG(2) << "Remote endpoint resolution failed. err=" << uv_strerror(err);
  } else {
    _address =
        fmtSockAddr(reinterpret_cast<struct ::sockaddr*>(&addr), addrLen);
  }
  int res = uv_read_start(
      reinterpret_cast<uv_stream_t*>(&client), allocBuffer, readCallback);
  if (res) {
    VLOG(2) << "Read callback initialization failure. client:"
            << reinterpret_cast<void*>(this) << " code:" << res
            << " desc:" << uv_strerror(res) << " name:" << uv_err_name(res);
    close();
  }
}

std::shared_ptr<LibUVClient> LibUVClient::make(uv_loop_t* loop,
                                               LibUVMasterDaemon* store) {
  auto res = std::make_shared<LibUVClient>(loop, store);
  res->handleAvailable();
  return res;
}

std::shared_ptr<LibUVClient> LibUVClient::ptr() {
  return std::static_pointer_cast<LibUVClient>(shared_from_this());
}

//  LibUVMasterDaemon
void LibUVMasterDaemon::onConnect(int status) {
  auto client = LibUVClient::make(&loop_, this);
  addClient(client);
  try {
    _tcp_server->accept(client);
    client->readStart();
  } catch (std::exception& e) {
    VLOG(2) << "Accept client failed, err: " << e.what();
    client->close();
  }
}

void LibUVMasterDaemon::onExitRequest() {
  VLOG(4) << "begin to exit requested";
  uv_close(reinterpret_cast<uv_handle_t*>(&_exit_handle), nullptr);
  uv_stop(&loop_);
}

void LibUVMasterDaemon::init(const std::uint16_t& port) {
  try {
    _tcp_server = LibUVTCPServer::createServer(&loop_, port, /*useIpv6=*/false);
  } catch (std::exception& ex) {
    PADDLE_THROW(common::errors::Fatal(
        paddle::string::Sprintf("Bind to ipv4 address failed: %s", ex.what())));
  }
  _tcp_server->setCallback([this](auto status) { this->onConnect(status); });

  port_ = _tcp_server->port();
  PADDLE_ENFORCE_EQ(
      port_,
      port,
      common::errors::InvalidArgument(paddle::string::Sprintf(
          "listen fd is bound to port %d, but expected port %d", port_, port)));
}

LibUVMasterDaemon::LibUVMasterDaemon(int port) : port_(port) {
  // uv loop init
  PADDLE_ENFORCE_EQ(uv_loop_init(&loop_),
                    0,
                    common::errors::InvalidArgument("init libuv loop failed"));
  // uv async init
  PADDLE_ENFORCE_EQ(
      uv_async_init(&loop_, &_exit_handle, LibUVMasterDaemon::on_exit_request),
      0,
      common::errors::InvalidArgument("init libuv async event failed"));
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&_exit_handle), this);
}

LibUVMasterDaemon::~LibUVMasterDaemon() {
  if (!is_running()) {
    uv_close(reinterpret_cast<uv_handle_t*>(&_exit_handle), nullptr);
    uv_run(&loop_, UV_RUN_NOWAIT);
    if (uv_loop_close(&loop_) != 0) {
      VLOG(0) << "uv loop close failed";
    }
  } else {
    // the daemon thread cleanup libuv
    cleanup();
  }
}

void LibUVMasterDaemon::run() {
  VLOG(4) << "start LibUV master daemon loop";
  int res = uv_run(&loop_, UV_RUN_DEFAULT);
  if (res) {
    VLOG(4) << "LibUV master daemon loop done: " << res;
  }

  for (const auto& client : _clients) {
    client->close();
  }
  _tcp_server->close();

  while (true) {
    res = uv_loop_close(&loop_);
    if (res == 0) {
      break;
    }
    VLOG(3) << "uv_loop_close failed with:" << res
            << " err: " << uv_err_name(res)
            << " std error:" << uv_strerror(res);
    res = uv_run(&loop_, UV_RUN_NOWAIT);
    if (res != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
  }
  VLOG(3) << "LibUV master daemon loop cleanup finished.";
}

void LibUVMasterDaemon::stop() {
  int res = uv_async_send(&_exit_handle);
  if (res) {
    VLOG(2) << "stop with uv_async_send failed:" << res
            << " err:" << uv_err_name(res) << " std error:" << uv_strerror(res);
  }
}

void LibUVMasterDaemon::addClient(const std::shared_ptr<LibUVHandle>& client) {
  _clients.insert(client);
}

void LibUVMasterDaemon::removeClient(
    const std::shared_ptr<LibUVHandle>& client) {
  _clients.erase(client);
  clearWaitState(client);
}

void LibUVMasterDaemon::clearWaitState(
    const std::shared_ptr<LibUVHandle>& client) {
  if (_awaited_keys.find(client) == _awaited_keys.end()) {
    return;
  }
  _awaited_keys.erase(client);
  for (auto it = _waiting_sockets.begin(); it != _waiting_sockets.end();) {
    for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
      if (*vecIt == client) {
        vecIt = it->second.erase(vecIt);
      } else {
        ++vecIt;
      }
    }
    if (it->second.empty()) {
      it = _waiting_sockets.erase(it);
    } else {
      ++it;
    }
  }
}

void LibUVMasterDaemon::set(const std::string& key,
                            const std::vector<uint8_t>& value) {
  _tcp_store[key] = value;
  // notify all clients that have been waiting
  notifyWaitingClients(key);
}

const std::vector<uint8_t>& LibUVMasterDaemon::get(const std::string& key) {
  static std::vector<uint8_t> missing_key;
  return _tcp_store.count(key) ? _tcp_store.at(key) : missing_key;
}

int64_t LibUVMasterDaemon::add(const std::string& key, int64_t addVal) {
  std::vector<uint8_t> old_data;
  auto it = _tcp_store.find(key);
  if (it != _tcp_store.end()) {
    old_data = it->second;
    auto buf = reinterpret_cast<const char*>(it->second.data());
    auto len = it->second.size();
    addVal += std::stoll(std::string(buf, len));
  }
  auto addValStr = std::to_string(addVal);
  std::vector<uint8_t> newData =
      std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  _tcp_store[key] = newData;

  // notify all clients that have been waiting
  notifyWaitingClients(key);
  return addVal;
}

bool LibUVMasterDaemon::checkKeys(const std::vector<std::string>& keys) {
  return std::all_of(keys.begin(), keys.end(), [&](const std::string& s) {
    if (_tcp_store.count(s) > 0) {
      return true;
    }
    return false;
  });
}

bool LibUVMasterDaemon::waitKey(const std::string& key,
                                const std::shared_ptr<LibUVHandle>& client) {
  int num_to_await = 0;
  if (_tcp_store.find(key) == _tcp_store.end()) {
    _waiting_sockets[key].push_back(client);
    num_to_await++;
    VLOG(7) << "add to wait key:  " << key;
  } else {
    return true;
  }
  _awaited_keys[client] = num_to_await;
  return false;
}

void LibUVMasterDaemon::notifyWaitingClients(const std::string& key) {
  auto sockets_to_wait = _waiting_sockets.find(key);
  if (sockets_to_wait != _waiting_sockets.end()) {
    for (const auto& client : sockets_to_wait->second) {
      if (--_awaited_keys[client] == 0) {
        UVWriter sw(client->ptr());
        sw.writeValue(ReplyType::STOP_WAIT);
        sw.send();
      }
    }
    _waiting_sockets.erase(sockets_to_wait);
  }
}

std::unique_ptr<phi::distributed::detail::DaemonThread> create_libuv_tcpstore(
    const std::uint16_t& port) {
  auto res = std::make_unique<LibUVMasterDaemon>(port);
  res->init(port);
  return res;
}
}  // namespace phi::distributed::detail
