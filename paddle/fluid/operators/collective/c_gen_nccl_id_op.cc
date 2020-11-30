/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

class CGenNCCLIdOp : public framework::OperatorBase {
 public:
  CGenNCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put nccl id in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    int rank = Attr<int>("rank");

    std::string endpoint = Attr<std::string>("endpoint");
    std::vector<std::string> endpoint_list =
        Attr<std::vector<std::string>>("other_endpoints");

    std::string var_name = Output("Out");
    auto var = scope->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Output can not be Null"));
    auto nccl_id = var->GetMutable<ncclUniqueId>();
    if (rank == 0) {
      PADDLE_ENFORCE_EQ(platform::dynload::ncclGetUniqueId(nccl_id), 0,
                        platform::errors::InvalidArgument(
                            "ncclGetUniqueId failed with id %s", nccl_id));
      for (auto& ep : endpoint_list) {
        VLOG(3) << "sending nccl id to " << ep;
        SendNCCLID(ep, nccl_id);
      }
    } else {
      RecvNCCLID(endpoint, nccl_id);
    }
  }

 private:
  void RecvNCCLID(const std::string& ep, ncclUniqueId* nccl_id) {
    auto addr = paddle::string::Split(ep, ':');
    PADDLE_ENFORCE_EQ(
        addr.size(), 2UL,
        platform::errors::InvalidArgument(
            "The endpoint should contain host and port, but got %s.", ep));
    std::string host = addr[0];
    int port = std::stoi(addr[1]);

    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    int opt = 0;
    // creating socket fd
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Create server file descriptor failed."));
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
      PADDLE_THROW(platform::errors::Unavailable("Set socket options failed."));
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    int try_times = 0;
    int retry_time = 0;
    while (true) {
      if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        retry_time = 3 * (try_times + 1);
        LOG(WARNING)
            << "Socket bind worker " << ep
            << (try_times < 9
                    ? " failed, try again after " + std::to_string(retry_time) +
                          " seconds."
                    : " failed, try again after " + std::to_string(retry_time) +
                          " seconds. Bind on endpoint " + ep +
                          " failed. Please confirm whether the "
                          "communication port or GPU card is occupied.");
        std::this_thread::sleep_for(std::chrono::seconds(retry_time));
        ++try_times;
        continue;
      }
      break;
    }

    VLOG(3) << "listening on: " << ep;
    if (listen(server_fd, 3) < 0) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Listen on server file descriptor failed."));
    }

    if ((new_socket =
             accept(server_fd, reinterpret_cast<struct sockaddr*>(&address),
                    reinterpret_cast<socklen_t*>(&addrlen))) < 0) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Accept the new socket file descriptor failed."));
    }

    if (read(new_socket, buffer, 1024) < 0) {
      PADDLE_THROW(platform::errors::Unavailable("Read from socket failed."));
    }

    VLOG(3) << "recevived the ncclUniqueId";
    memcpy(nccl_id, buffer, NCCL_UNIQUE_ID_BYTES);

    VLOG(3) << "closing the socket server: " << ep;
    close(server_fd);
  }

  void SendNCCLID(const std::string& ep, ncclUniqueId* nccl_id) {
    auto addr = paddle::string::Split(ep, ':');
    PADDLE_ENFORCE_EQ(
        addr.size(), 2UL,
        platform::errors::InvalidArgument(
            "The endpoint should contain host and port, but got %s.", ep));
    std::string host = addr[0];
    int port = std::stoi(addr[1]);
    // struct sockaddr_in address;
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};

    memcpy(buffer, nccl_id, NCCL_UNIQUE_ID_BYTES);
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      PADDLE_THROW(platform::errors::Unavailable("Create socket failed."));
    }

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    char* ip = NULL;
    struct hostent* hp;
    if ((hp = gethostbyname(host.c_str())) == NULL) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Fail to get host by name %s.", host));
    }
    int i = 0;
    while (hp->h_addr_list[i] != NULL) {
      ip = inet_ntoa(*(struct in_addr*)hp->h_addr_list[i]);
      VLOG(3) << "gethostbyname  host:" << host << "  ->ip: " << ip;
      break;
    }
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
      PADDLE_THROW(
          platform::errors::Unavailable("Open address %s failed.", ep));
    }

    int try_times = 0;
    int retry_time = 0;
    while (true) {
      if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        retry_time = 3 * (try_times + 1);
        LOG(WARNING)
            << "Socket connect worker " << ep
            << (try_times < 9
                    ? " failed, try again after " + std::to_string(retry_time) +
                          " seconds."
                    : " failed, try again after " + std::to_string(retry_time) +
                          " seconds. Maybe that some process is occupied the "
                          "GPUs of this node now, and you should kill those "
                          "process manually.");
        std::this_thread::sleep_for(std::chrono::seconds(retry_time));
        ++try_times;
        continue;
      }
      VLOG(3) << "sending the ncclUniqueId to " << ep;
      send(sock, buffer, NCCL_UNIQUE_ID_BYTES, 0);
      break;
    }
    close(sock);
  }
};

class CGenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
CGenNCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "current listen endpoint");
    AddAttr<std::vector<std::string>>(
        "other_endpoints",
        "['trainer1_ip:port', 'trainer2_ip:port', ...] "
        "list of other trainer endpoints")
        .SetDefault({});
    AddAttr<int>("rank",
                 "(int default 0) "
                 "The rank of the trainer in distributed training.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_nccl_id, ops::CGenNCCLIdOp, ops::CGenNCCLIdOpMaker);
