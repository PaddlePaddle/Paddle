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

#include <gtest/gtest.h>
#include <memory>
#include "ParameterService.pb.h"
#include "paddle/math/Vector.h"
#include "paddle/pserver/ProtoServer.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

DEFINE_string(server_addr, "127.0.0.1", "Server address");
DEFINE_int64(dim, 50000000, "Data size");
DEFINE_bool(test_proto_server, true, "whether to test ProtoServer");
DEFINE_bool(benchmark, false, "Do benchmark. Skip some tests");

using namespace paddle;  // NOLINT

class MyServer : public ProtoServer {
public:
  explicit MyServer(int port, int rdmaCpu = -1)
      : ProtoServer(FLAGS_server_addr, port, rdmaCpu),
        status_(PSERVER_STATUS_NOT_SET) {
    REGISTER_SERVICE_FUNCTION(MyServer, getStatus);
    REGISTER_SERVICE_FUNCTION(MyServer, setStatus);
    REGISTER_SERVICE_FUNCTION_EX(MyServer, getStatusEx);
  }
  void getStatus(const GetStatusRequest& request,
                 ProtoResponseCallback callback) {
    (void)request;
    GetStatusResponse response;
    response.set_status(status_);
    callback(response);
  }

  void getStatusEx(const GetStatusRequest& request,
                   std::unique_ptr<MsgReader> msgReader,
                   ProtoResponseCallbackEx callback) {
    (void)request;
    GetStatusResponse response;
    response.set_status(status_);
    buffer_.resize(msgReader->getNextBlockLength());
    msgReader->readNextBlock(&buffer_[0]);
    callback(response, {{&buffer_[0], buffer_.size()}});
  }

  void setStatus(const SetStatusRequest& request,
                 ProtoResponseCallback callback) {
    SetStatusResponse response;
    status_ = request.status();
    callback(response);
  }

protected:
  PServerStatus status_;
  std::string buffer_;
};

TEST(ProtoServer, regular) {
  ProtoClient* client;
  if (FLAGS_rdma_tcp == "rdma")
    client = new ProtoClient(FLAGS_server_addr, FLAGS_port, F_RDMA);
  else
    client = new ProtoClient(FLAGS_server_addr, FLAGS_port, F_TCP);
  {
    GetStatusRequest request;
    GetStatusResponse response;
    auto msgReader = client->sendAndRecv("getStatus", request, &response);
    EXPECT_EQ(response.status(), PSERVER_STATUS_NOT_SET);
    EXPECT_EQ(msgReader->getNumBlocks(), (size_t)0);
  }

  {
    SetStatusRequest request;
    SetStatusResponse response;
    request.set_status(PSERVER_STATUS_PARAMETER_READY);
    client->sendAndRecv("setStatus", request, &response);
  }

  {
    GetStatusRequest request;
    GetStatusResponse response;
    client->sendAndRecv("getStatus", request, &response);
    EXPECT_EQ(response.status(), PSERVER_STATUS_PARAMETER_READY);
  }

  delete client;
}

TEST(ProtoServer, extended) {
#ifndef PADDLE_ONLY_CPU
  ProtoClient* client;
  if (FLAGS_rdma_tcp == "rdma")
    client = new ProtoClient(FLAGS_server_addr, FLAGS_port, F_RDMA);
  else
    client = new ProtoClient(FLAGS_server_addr, FLAGS_port, F_TCP);
  int64_t dataSize = FLAGS_dim * sizeof(real);

  GpuVector gpuParam(FLAGS_dim);
  GpuVector gpuGrad(FLAGS_dim);
  CpuVector cpuParam(FLAGS_dim);
  CpuVector cpuGrad(FLAGS_dim);

  gpuParam.rand();
  gpuGrad.rand();
  cpuParam.rand();
  cpuGrad.rand();

  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 10; ++i) {
      cpuGrad.copyFrom(gpuGrad);
      if (FLAGS_test_proto_server) {
        GetStatusRequest request;
        GetStatusResponse response;
        {
          REGISTER_TIMER("sendAndRecv");
          auto msgReader =
              client->sendAndRecv("getStatusEx",
                                  request,
                                  {{cpuGrad.getData(), (size_t)dataSize}},
                                  &response);

          EXPECT_EQ(msgReader->getNumBlocks(), (size_t)1);
          EXPECT_EQ(msgReader->getNextBlockLength(), (size_t)dataSize);
          msgReader->readNextBlock(cpuParam.getData());
        }
        if (!FLAGS_benchmark) {
          real* v1 = cpuGrad.getData();
          real* v2 = cpuParam.getData();
          real sum1 = 0, sum2 = 0;
          for (int j = 0; j < FLAGS_dim; ++j) {
            sum1 += v1[j];
            sum2 += v2[j];
          }
          EXPECT_EQ(sum1, sum2);
        }
      }
      gpuParam.copyFrom(cpuParam);

      LOG_EVERY_N(INFO, 10) << "i=" << i;
    }
    globalStat.printAllStatus();
    globalStat.reset();
  }

  delete client;
#endif
}

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  MyServer server(FLAGS_port, FLAGS_rdma_tcp == "rdma" ? 0 : -1);
  server.start();
  usleep(10000);

  return RUN_ALL_TESTS();
}
