// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/service/heter_server.h"

#include "paddle/fluid/string/split.h"

namespace paddle {
namespace distributed {
// DEFINE_string(cert_path, "./cert.pem", "cert.pem path");
// DEFINE_string(key_path, "./key.pem", "key.pem path");

std::shared_ptr<HeterServer> HeterServer::s_instance_ = nullptr;

void HeterServer::RegisterServiceHandler(std::string message_name,
                                         HeterServiceHandler func) {
  service_.RegisterServiceHandler(message_name, func);
}

void HeterServer::StartHeterService(bool neeed_encrypt) {
  server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (neeed_encrypt) {
    options.ssl_options.default_cert.certificate = "/cert.pem";
    options.ssl_options.default_cert.private_key = "/key.pem";
  }
  if (server_.Start(endpoint_.c_str(), &options) != 0) {
    VLOG(0) << "HeterServer start fail. Try again.";
    auto ip_port = paddle::string::Split(endpoint_, ':');
    std::string ip = ip_port[0];
    int port = std::stoi(ip_port[1]);
    std::string int_ip_port = GetIntTypeEndpoint(ip, port);
    if (server_.Start(endpoint_.c_str(), &options) != 0) {
      LOG(ERROR) << "HeterServer start failed, ip_port= " << int_ip_port;
    }
  } else {
    VLOG(0) << "heter server start success! listen on " << endpoint_;
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    stoped_ = false;
    ready_ = 1;
  }
  condition_ready_.notify_all();
  VLOG(4) << "stopped: " << stoped_ << ", ready_: " << ready_;
  std::unique_lock<std::mutex> running_lock(mutex_);
  cv_.wait(running_lock, [&] {
    VLOG(4) << "Heter Server is Stop? " << stoped_;
    return stoped_;
  });
  VLOG(4) << "start service done";
}

void HeterServer::StartHeterInterService(bool neeed_encrypt) {
  server_inter_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (neeed_encrypt) {
    options.ssl_options.default_cert.certificate = "/cert.pem";
    options.ssl_options.default_cert.private_key = "/key.pem";
  }
  if (server_inter_.Start(endpoint_inter_.c_str(), &options) != 0) {
    VLOG(4) << "switch inter server start fail. Try again.";
    auto ip_port = paddle::string::Split(endpoint_inter_, ':');
    std::string ip = ip_port[0];
    int port = std::stoi(ip_port[1]);
    std::string int_ip_port = GetIntTypeEndpoint(ip, port);
    if (server_inter_.Start(endpoint_inter_.c_str(), &options) != 0) {
      LOG(ERROR) << "switch inter server start failed, ip_port= "
                 << int_ip_port;
    }
  } else {
    VLOG(4) << "switch inter server server start success! listen on "
            << endpoint_inter_;
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    stoped_ = false;
    ready_ = 1;
  }
  condition_ready_.notify_all();
  VLOG(4) << "stopped: " << stoped_ << ", ready_: " << ready_;
  std::unique_lock<std::mutex> running_lock(mutex_);
  cv_.wait(running_lock, [&] {
    VLOG(4) << "Heter Server is Stop? " << stoped_;
    return stoped_;
  });
  VLOG(4) << "start service done";
}

void HeterServer::SetFanin(const int& fan_in) { service_.SetFanin(fan_in); }

void HeterServer::WaitServerReady() {
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
  while (!this->ready_) {
    sleep(1);
  }
}

}  // end namespace distributed
}  // namespace paddle
