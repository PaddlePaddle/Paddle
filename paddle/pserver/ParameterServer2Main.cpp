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

#include <fstream>
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

#include "ParameterServer2.h"
#include "RDMANetwork.h"
#include "paddle/utils/Flags.h"

using namespace paddle;  // NOLINT

int main(int argc, char** argv) {
  initMain(argc, argv);

  std::vector<std::string> devices;
  std::vector<std::shared_ptr<ParameterServer2>> pservers;

  // round robin to loadbalance RDMA server ENGINE
  int rdmaCpu = 0;
  int onlineCpus = rdma::numCpus();
  int numPorts = FLAGS_ports_num + FLAGS_ports_num_for_sparse;
  if (FLAGS_nics.empty()) {
    pservers.resize(numPorts);
    for (int i = 0; i < numPorts; ++i) {
      if (FLAGS_rdma_tcp == "rdma") {
        pservers[i].reset(
            new ParameterServer2(std::string(), FLAGS_port + i, rdmaCpu++));
        rdmaCpu = rdmaCpu % onlineCpus;
      } else {
        pservers[i].reset(new ParameterServer2(std::string(), FLAGS_port + i));
      }
      CHECK(pservers[i]->init()) << "Fail to initialize parameter server"
                                 << FLAGS_port + i;
      LOG(INFO) << "pserver started : " << FLAGS_port + i;
      pservers[i]->start();
    }
  } else {
    str::split(FLAGS_nics, ',', &devices);
    pservers.resize(devices.size() * numPorts);
    for (int i = 0; i < numPorts; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (FLAGS_rdma_tcp == "rdma") {
          pservers[i * devices.size() + j].reset(new ParameterServer2(
              getIpAddr(devices[j]), FLAGS_port + i, rdmaCpu++));
          rdmaCpu = rdmaCpu % onlineCpus;
        } else {
          pservers[i * devices.size() + j].reset(
              new ParameterServer2(getIpAddr(devices[j]), FLAGS_port + i));
        }
        CHECK(pservers[i * devices.size() + j]->init())
            << "Fail to initialize parameter server" << devices[j]
            << FLAGS_port + i;
        LOG(INFO) << "pserver started : " << devices[j] << ":"
                  << FLAGS_port + i;
        pservers[i * devices.size() + j]->start();
      }
    }
  }

  for (auto& pserver : pservers) {
    pserver->join();
  }

  return 0;
}
