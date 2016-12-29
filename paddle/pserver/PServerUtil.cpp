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

#include "PServerUtil.h"

namespace paddle {

void PServerUtil::init() {
  // round robin to load balance RDMA server ENGINE
  std::vector<std::string> devices;
  int rdmaCpu = 0;
  int onlineCpus = rdma::numCpus();
  int numPorts = FLAGS_ports_num + FLAGS_ports_num_for_sparse;

  if (FLAGS_nics.empty()) {
    pservers_.resize(numPorts);
    for (int i = 0; i < numPorts; ++i) {
      if (FLAGS_rdma_tcp == "rdma") {
        pservers_[i].reset(
            new ParameterServer2(std::string(), FLAGS_port + i, rdmaCpu++));
        rdmaCpu = rdmaCpu % onlineCpus;
      } else {
        pservers_[i].reset(new ParameterServer2(std::string(), FLAGS_port + i));
      }
      CHECK(pservers_[i]->init()) << "Fail to initialize parameter server"
                                  << FLAGS_port + i;
    }
  } else {
    str::split(FLAGS_nics, ',', &devices);
    pservers_.resize(devices.size() * numPorts);
    for (int i = 0; i < numPorts; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (FLAGS_rdma_tcp == "rdma") {
          pservers_[i * devices.size() + j].reset(new ParameterServer2(
              getIpAddr(devices[j]), FLAGS_port + i, rdmaCpu++));
          rdmaCpu = rdmaCpu % onlineCpus;
        } else {
          pservers_[i * devices.size() + j].reset(
              new ParameterServer2(getIpAddr(devices[j]), FLAGS_port + i));
        }
        CHECK(pservers_[i * devices.size() + j]->init())
            << "Fail to initialize parameter server" << devices[j]
            << FLAGS_port + i;
      }
    }
  }
}

void PServerUtil::start() {
  LOG(INFO) << "pserver sizes : " << pservers_.size();
  int i = 0;
  for (const auto &pserver : pservers_) {
    LOG(INFO) << "pserver started : " << i;
    pserver->start();
    i++;
  }
}

void PServerUtil::join() {
  LOG(INFO) << "pserver sizes : " << pservers_.size();
  for (const auto &pserver : pservers_) {
    pserver->join();
  }
}

}  // namespace paddle
