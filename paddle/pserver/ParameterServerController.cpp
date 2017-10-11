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

#include "ParameterServerController.h"

namespace paddle {

ParameterServerController::ParameterServerController(
    const ParameterServerConfig& config) {
  // round robin to load balance RDMA server ENGINE
  std::vector<std::string> devices;
  int rdmaCpu = 0;
  int onlineCpus = rdma::numCpus();
  int numPorts = config.ports_num() + config.ports_num_for_sparse();

  if (config.nics().empty()) {
    parameterServers_.resize(numPorts);
    for (int i = 0; i < numPorts; ++i) {
      if (config.rdma_tcp() == "rdma") {
        parameterServers_[i].reset(
            new ParameterServer2(std::string(), config.port() + i, rdmaCpu++));
        rdmaCpu = rdmaCpu % onlineCpus;
      } else {
        parameterServers_[i].reset(
            new ParameterServer2(std::string(), config.port() + i));
      }
      CHECK(parameterServers_[i]->init()) << "Fail to initialize parameter "
                                             "server on port "
                                          << config.port() + i;
    }
  } else {
    str::split(config.nics(), ',', &devices);
    parameterServers_.resize(devices.size() * numPorts);
    for (int i = 0; i < numPorts; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (config.rdma_tcp() == "rdma") {
          parameterServers_[i * devices.size() + j].reset(new ParameterServer2(
              getIpAddr(devices[j]), config.port() + i, rdmaCpu++));
          rdmaCpu = rdmaCpu % onlineCpus;
        } else {
          parameterServers_[i * devices.size() + j].reset(
              new ParameterServer2(getIpAddr(devices[j]), config.port() + i));
        }
        CHECK(parameterServers_[i * devices.size() + j]->init())
            << "Fail to initialize parameter server with device " << devices[j]
            << config.port() + i;
      }
    }
  }
}

ParameterServerController::~ParameterServerController() { this->wait(); }

ParameterServerController* ParameterServerController::createFromGflags() {
  ParameterServerConfig config;

  config.set_nics(FLAGS_nics);
  config.set_rdma_tcp(FLAGS_rdma_tcp);
  config.set_port(FLAGS_port);
  config.set_ports_num(FLAGS_ports_num);
  config.set_ports_num_for_sparse(FLAGS_ports_num_for_sparse);

  return create(config);
}

ParameterServerController* ParameterServerController::create(
    const ParameterServerConfig& config) {
  return new ParameterServerController(config);
}

void ParameterServerController::start() {
  LOG(INFO) << "number of parameterServer instances: "
            << parameterServers_.size();
  int i = 0;
  for (const auto& parameterServer : parameterServers_) {
    LOG(INFO) << "Starting parameterServer[" << i << "]";
    parameterServer->start();
    i++;
  }
}

void ParameterServerController::wait() {
  int i = 0;
  for (const auto& parameterServer : parameterServers_) {
    LOG(INFO) << "Waiting parameterServer[" << i << "]";
    parameterServer->join();
    i++;
  }
}

}  // namespace paddle
