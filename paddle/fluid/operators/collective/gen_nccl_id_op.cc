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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

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
    std::vector<std::string> trainers =
        Attr<std::vector<std::string>>("trainers");
    int trainer_id = Attr<int>("trainer_id");
    std::string endpoint = trainers[trainer_id];

    PADDLE_ENFORCE_GE(trainer_id, 0, platform::errors::InvalidArgument(
                                         "trainer_id %d is less than 0. Its "
                                         "valid range is [0, trainer_size)"));
    PADDLE_ENFORCE_LT(
        trainer_id, static_cast<int>(trainers.size()),
        platform::errors::OutOfRange("trainer_id %d is out of range. Its valid "
                                     "range is [0, trainer_size)",
                                     trainer_id));

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
            << ", nccl_comm_num:" << nccl_comm_num
            << ", inter_nranks:" << inter_nranks
            << ", inter_trainer_id:" << inter_trainer_id
            << ", exter_trainer_id:" << exter_trainer_id
            << ", trainers:" << ss.str();

    std::function<std::string(size_t)> func = platform::GetFlatNCCLVarName;
    // init flat
    if (trainer_id == 0) {
      // server endpoints
      std::vector<std::string> flat_endpoints;
      flat_endpoints.insert(flat_endpoints.begin(), trainers.begin() + 1,
                            trainers.end());
      SendBroadCastNCCLID(flat_endpoints, nccl_comm_num, func, scope);
    } else {
      RecvBroadCastNCCLID(endpoint, nccl_comm_num, func, scope);
    }

    if (!use_hierarchical_allreduce) {
      return;
    }

    PADDLE_ENFORCE_EQ(
        trainers.size() % inter_nranks, 0,
        platform::errors::PreconditionNotMet(
            "The number of trainers %llu mod inter_nranks %d is not equal 0",
            trainers.size(), inter_nranks));
    PADDLE_ENFORCE_GT(
        inter_nranks, 1,
        platform::errors::PreconditionNotMet(
            "inter_nranks %d <= 1 while in hierarchical allreduce mode",
            inter_nranks));

    func = platform::GetHierarchicalInterNCCLVarName;
    // hierarchical inter ncclid
    if (inter_trainer_id == 0) {
      std::ostringstream ss;
      ss << endpoint;
      std::vector<std::string> inter_endpoints;
      for (int i = trainer_id + 1; i < trainer_id + inter_nranks &&
                                   i < static_cast<int>(trainers.size());
           i++) {
        ss << ",";
        inter_endpoints.push_back(trainers[i]);
        ss << trainers[i];
      }
      VLOG(1) << "Hierarchical inter ring endpoints:" << ss.str();

      SendBroadCastNCCLID(inter_endpoints, nccl_comm_num, func, scope);
    } else {
      RecvBroadCastNCCLID(endpoint, nccl_comm_num, func, scope);
    }

    func = platform::GetHierarchicalExterNCCLVarName;
    // hierarchical exter ncclid
    if (exter_trainer_id == 0) {
      std::ostringstream ss;
      std::vector<std::string> exter_endpoints;
      ss << endpoint;
      for (size_t i = inter_nranks; i < trainers.size(); i += inter_nranks) {
        ss << ",";
        exter_endpoints.push_back(trainers[i]);
        ss << trainers[i];
      }
      VLOG(1) << "Hierarchical exter ring endpoints:" << ss.str();

      SendBroadCastNCCLID(exter_endpoints, nccl_comm_num, func, scope);
    } else if (exter_trainer_id > 0) {
      RecvBroadCastNCCLID(endpoint, nccl_comm_num, func, scope);
    }
  }

 private:
  static const char COMM_HEAD[];
  static const size_t HEAD_LEN;

  int StartServer(const std::string& ep) const {
    auto addr = paddle::string::Split(ep, ':');
    PADDLE_ENFORCE_EQ(
        addr.size(), 2UL,
        platform::errors::InvalidArgument(
            "The endpoint should contain host and port, but got %s.", ep));
    std::string host = addr[0];
    int port = std::stoi(addr[1]);

    // creating socket fd
    int server_fd;
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Create server file descriptor failed."));
    }

    int opt = 0;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
      PADDLE_THROW(platform::errors::Unavailable("Set socket options failed."));
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    int try_times = 0;
    while (true) {
      if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        LOG(WARNING)
            << "Socket bind worker " << ep << " failed with " << strerror(errno)
            << (try_times < 9
                    ? " failed, try again after 3 seconds."
                    : " failed, try again after 3 seconds. Bind on endpoint " +
                          ep +
                          " failed. Please confirm whether the "
                          "communication port or GPU card is occupied.");
        std::this_thread::sleep_for(std::chrono::seconds(3));
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

    return server_fd;
  }

  int AcceptClient(int server_fd) const {
    struct sockaddr_in client_addr;
    socklen_t addr_length = sizeof(client_addr);
    char buffer[1024] = {0};
    int conn = 0;

    while (true) {
      conn = accept(server_fd, reinterpret_cast<struct sockaddr*>(&client_addr),
                    &addr_length);
      if (read(conn, buffer, 1024) < 0) {
        close(conn);
      } else if (strncmp(buffer, COMM_HEAD, HEAD_LEN) == 0) {
        // accept client
        break;
      } else {
        close(conn);
      }
    }

    PADDLE_ENFORCE_GE(conn, 0,
                      platform::errors::Unavailable(
                          "Accept the new socket file descriptor failed."));
    return conn;
  }

  int ConnServer(const std::string& ep) const {
    auto addr = paddle::string::Split(ep, ':');
    PADDLE_ENFORCE_EQ(
        addr.size(), 2UL,
        platform::errors::InvalidArgument(
            "The endpoint should contain host and port, but got %s.", ep));
    std::string host = addr[0];
    int port = std::stoi(addr[1]);

    int sock = 0;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      PADDLE_THROW(platform::errors::Unavailable("Create socket failed."));
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, '0', sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

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
    if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
      PADDLE_THROW(
          platform::errors::Unavailable("Open address %s failed.", ep));
    }

    int try_times = 0;
    int retry_time = 0;
    while (true) {
      if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) <
          0) {
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
      if (send(sock, COMM_HEAD, HEAD_LEN, 0) < 0) {
        PADDLE_THROW(platform::errors::Unavailable("Send comm head failed."));
      }
      break;
    }
    return sock;
  }

  void RecvNCCLID(int conn, ncclUniqueId* nccl_id) const {
    char buffer[1024] = {0};
    int len = read(conn, buffer, 1024);
    PADDLE_ENFORCE_EQ(
        len, NCCL_UNIQUE_ID_BYTES,
        platform::errors::Unavailable("Received the ncclUniqueId failed"));
    VLOG(3) << "recevived the ncclUniqueId, len=" << len
            << " nccl_id_len=" << NCCL_UNIQUE_ID_BYTES;
    memcpy(nccl_id, buffer, NCCL_UNIQUE_ID_BYTES);
  }

  void SendNCCLID(int conn, ncclUniqueId* nccl_id) const {
    char buffer[1024] = {0};
    memcpy(buffer, nccl_id, NCCL_UNIQUE_ID_BYTES);
    if (send(conn, buffer, NCCL_UNIQUE_ID_BYTES, 0) < 0) {
      PADDLE_THROW(platform::errors::Unavailable("Send nccl id failed."));
    }
  }

  template <typename NCCLVarNameFunc>
  void SendBroadCastNCCLID(std::vector<std::string> servers, int nccl_comm_num,
                           NCCLVarNameFunc func,
                           const framework::Scope& scope) const {
    // connect with server
    std::vector<int> connects;
    for (auto server : servers) {
      int conn = ConnServer(server);
      connects.push_back(conn);
    }

    for (int i = 0; i < nccl_comm_num; ++i) {
      std::string var_name = func(i);
      auto var = scope.FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound("Variable with name %s is not found",
                                          var_name.c_str()));
      auto nccl_id = var->GetMutable<ncclUniqueId>();
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGetUniqueId(nccl_id));

      for (auto conn : connects) {
        SendNCCLID(conn, nccl_id);
      }
    }

    // close client
    for (auto conn : connects) {
      close(conn);
    }
  }

  template <typename NCCLVarNameFunc>
  void RecvBroadCastNCCLID(std::string endpoint, int nccl_comm_num,
                           NCCLVarNameFunc func,
                           const framework::Scope& scope) const {
    int server = StartServer(endpoint);
    int client = AcceptClient(server);

    for (int i = 0; i < nccl_comm_num; ++i) {
      std::string var_name = func(i);
      auto var = scope.FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound("Variable with name %s is not found",
                                          var_name.c_str()));
      auto nccl_id = var->GetMutable<ncclUniqueId>();

      RecvNCCLID(client, nccl_id);
    }
    close(client);
    close(server);
  }
};

const char GenNCCLIdOp::COMM_HEAD[] = "_pd_gen_comm_id_";
const size_t GenNCCLIdOp::HEAD_LEN = sizeof(GenNCCLIdOp::COMM_HEAD) - 1;

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
