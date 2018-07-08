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

#include <grpc++/grpc++.h>
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <thread>

#include "mpi.grpc.pb.h"
#include "mpi_client.h"
#include "safe_queue.h"
#include "var.h"

using grpc::Status;
using grpc::ServerContext;
using grpc::ClientContext;

using mpis::RequestContext;
using mpis::ReplyContext;
using mpis::MPIService;

/*************************************** CLIENT
 * *******************************************/

Var geneVar(std::string grpc, std::string name, std::string content, int tag) {
  content = content + std::to_string(tag);
  name = name + std::to_string(tag);

  Var var;
  var.grpc = grpc;
  var.value = content;
  var.length = content.length();
  var.tag = tag;
  var.name = name;

  return var;
}

void MPIAsyncSendVars(std::shared_ptr<grpc::Channel> channel,
                      int src,
                      const Var &var) {
  std::cout << "[MPI_CLIENT " << src << "]: async sending " << var.name
            << std::endl;
  MPIClient mpiClient(channel, src);
  mpiClient.SendRequest(var);
}

void RunClient(const int src, const std::string grpc) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(grpc, grpc::InsecureChannelCredentials());

  int tag_base = 100;
  std::string name = "batch_norm@GREND_tag_";
  std::string content = "1:2:3:4:5:5::_tag_";

  for (int i = 1; i < 100; ++i) {
    int tag = tag_base + i;
    Var var = geneVar(grpc, name, content, tag);
    std::thread mpi_send_thread(MPIAsyncSendVars, channel, src, var);
    mpi_send_thread.detach();
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  std::cout << "[MPIClient " << src << "]: send over" << std::endl;
}

/*************************************** SERVER
 * *******************************************/

struct mpi_receive_log {
  int src;
  int tag;
  int length;
  std::string var_name;
  std::string content;
};

void MPILogsProcess(SharedQueue<mpi_receive_log> &queue) {
  std::cout << "MPI SERVER "
            << " MPILogsProcess WAIT TO PRINT LOG " << std::endl;

  int consumedCounter = 0;
  for (;;) {
    if (!queue.empty()) {
      mpi_receive_log receive_log = queue.front();

      std::cout << "[MPI SERVER]: "
                << "src: " << receive_log.src << " tag: " << receive_log.tag
                << " length: " << receive_log.length
                << " var: " << receive_log.var_name
                << " content: " << receive_log.content << std::endl;

      queue.pop_front();
      consumedCounter++;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (consumedCounter >= 99) {
      std::cout << "[MPI SERVER]: "
                << " MPI SERVER RECEIVE " << consumedCounter
                << " MESSAGES, Shutdown" << std::endl;
      queue.shut_down();
      break;
    }
  }
}

void MPIServiceStop(SharedQueue<mpi_receive_log> &queue, grpc::Server *server) {
  while (true) {
    if (queue.is_shutdown()) {
      server->Shutdown();
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void MPIIrecvProcess(void *buf, int count, int source, int tag) {
  MPI_Request request;
  MPI_Status status;

  MPI_Irecv(buf, count, MPI_CHAR, source, tag, MPI_COMM_WORLD, &request);
  MPI_Wait(&request, &status);
}

void MPIAsyncRecvVars(const RequestContext &request,
                      const ReplyContext &response,
                      SharedQueue<mpi_receive_log> *queue) {
  std::stringstream id;
  id << std::this_thread::get_id();

  char bus[request.length()];
  MPIIrecvProcess(&bus, request.length(), request.src(), request.tag());

  mpi_receive_log receive_log;
  receive_log.src = request.src();
  receive_log.tag = request.tag();
  receive_log.var_name = request.var_name();
  receive_log.length = request.length();

  std::string content_str(bus);
  receive_log.content = content_str;

  queue->push_back(receive_log);
}

class MPIServiceImpl final : public mpis::MPIService::Service {
 public:
  MPIServiceImpl(int dst, SharedQueue<mpi_receive_log> *queue) {
    this->queue = queue;
    this->dst = dst;
  }

  Status ISendRequest(::grpc::ServerContext *context,
                      const ::mpis::RequestContext *request,
                      ::mpis::ReplyContext *response) override {
    std::thread mpi_receive_thread(
        MPIAsyncRecvVars, *request, *response, queue);
    mpi_receive_thread.detach();
    response->set_dst(this->dst);
    return Status::OK;
  };

 private:
  int dst;
  SharedQueue<mpi_receive_log> *queue;
};

void RunServer(int dst) {
  SharedQueue<mpi_receive_log> queue;
  std::thread mpi_receive_log_thread(MPILogsProcess, std::ref(queue));
  mpi_receive_log_thread.detach();

  grpc::ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  std::string server_address("0.0.0.0:50051");
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  MPIServiceImpl service(dst, &queue);
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  std::cout << "[MPIServer " << dst << "]: "
            << "Server listening on " << server_address << std::endl;

  std::thread mpi_server_exit(MPIServiceStop, std::ref(queue), server.get());

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

/*************************************** MAIN
 * *******************************************/
int main(int argc, char **argv) {
  int flag = 0;
  int rank = -1;
  int size = 1;
  int provided;

  MPI_Initialized(&flag);

  if (!flag) {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  std::cout << "MPI: "
            << "RANK: " << rank << " SIZE: " << size << std::endl;

  if (rank == 0) {
    std::cout << "rank 0: RunServer" << rank << std::endl;
    RunServer(rank);
  } else if (rank == 1) {
    std::cout << "rank 1: RunClient" << rank << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    RunClient(rank, "0.0.0.0:50051");
  } else {
    std::cout << "rank error: " << rank << std::endl;
  }

  MPI_Finalize();

  return 0;
}