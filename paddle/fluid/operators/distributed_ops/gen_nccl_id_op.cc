/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <ostream>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

class GenNCCLIdOp : public framework::OperatorBase {
 public:
  GenNCCLIdOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put nccl id in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    int trainer_id = Attr<int>("trainer_id");

    std::vector<std::string> trainers =
        Attr<std::vector<std::string>>("trainers");
    PADDLE_ENFORCE_GE(trainer_id, 0, platform::errors::InvalidArgument(
                                         "trainer_id %d is less than 0. Its "
                                         "valid range is [0, trainer_size)"));
    PADDLE_ENFORCE_LT(
        trainer_id, static_cast<int>(trainers.size()),
        platform::errors::OutOfRange("trainer_id %d is out of range. Its valid "
                                     "range is [0, trainer_size)",
                                     trainer_id));

    std::string endpoint = trainers[trainer_id];

    int nccl_comm_num = Attr<int>("nccl_comm_num");
    int use_hierarchical_allreduce = Attr<bool>("use_hierarchical_allreduce");
    int inter_nranks = Attr<int>("hierarchical_allreduce_inter_nranks");
    int inter_trainer_id = -1;
    int exter_trainer_id = -1;

    if (use_hierarchical_allreduce) {
      PADDLE_ENFORCE_GT(
          trainers.size(), 1,
          platform::errors::PreconditionNotMet(
              "The number of collective trainers %llu <= 1", trainers.size()));
      PADDLE_ENFORCE_GT(
          inter_nranks, 1,
          platform::errors::PreconditionNotMet(
              "inter_nranks %d <= 1 while in hierarchical allreduce mode",
              inter_nranks));
      PADDLE_ENFORCE_EQ(
          trainers.size() % inter_nranks, 0,
          platform::errors::PreconditionNotMet(
              "The number of trainers %llu mod inter_nranks %d is not equal 0",
              trainers.size(), inter_nranks));

      inter_trainer_id = trainer_id % inter_nranks;

      if (trainer_id % inter_nranks == 0) {
        exter_trainer_id = trainer_id / inter_nranks;
      }
    }

    std::ostringstream ss;
    for (size_t i = 0; i < trainers.size(); i++) {
      ss << trainers[i] << ",";
    }

    VLOG(1) << "trainer_id:" << trainer_id
            << ", use_hierarchical_allreduce:" << use_hierarchical_allreduce
            << ", inter_nranks:" << inter_nranks
            << ", inter_trainer_id:" << inter_trainer_id
            << ", exter_trainer_id:" << exter_trainer_id
            << ", trainers:" << ss.str();
  }

 private:
  void RecvNCCLID(const std::string& ep, ncclUniqueId* nccl_id,
                  int nccl_comm_num, bool use_hierarchical_allreduce,
                  int trainer_id, int inter_trainer_id, int exter_trainer_id) {
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

class GenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("NCCLID", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
GenNCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::vector<std::string>>(
        "trainers",
        "['trainer0_ip:port', 'trainer1_ip:port', ...] "
        "list of all trainer endpoints")
        .SetDefault({});
    AddAttr<int>("trainer_id",
                 "(int) "
                 "The index of the trainer in distributed training.");
    AddAttr<int>("nccl_comm_num",
                 "(int default 1) "
                 "The number of nccl communicator num.")
        .SetDefault(1);
    AddAttr<bool>("use_hierarchical_allreduce",
                  "(bool default false) "
                  "Wheter to use hierarchical allreduce.")
        .SetDefault(false);
    AddAttr<int>("hierarchical_allreduce_inter_nranks",
                 "(int default 1) "
                 "Wheter to use hierarchical allreduce.")
        .SetDefault(-1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_nccl_id, ops::GenNCCLIdOp, ops::GenNCCLIdOpMaker);
