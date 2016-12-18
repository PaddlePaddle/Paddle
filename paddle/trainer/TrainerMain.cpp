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

#include <fenv.h>
#include "paddle/pserver/ParameterServer2.h"
#include "paddle/utils/Excepts.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/StringUtil.h"

#include "ParamUtil.h"
#include "Trainer.h"
#include "paddle/pserver/RDMANetwork.h"

DEFINE_bool(start_pserver, false, "Whether to start pserver");
DECLARE_int32(gpu_id);
DEFINE_string(job, "train", "one of (train, test, checkgrad)");
DECLARE_int32(start_pass);
DECLARE_string(config);
DECLARE_string(init_model_path);
DECLARE_string(rdma_tcp);

using namespace paddle;  // NOLINT

int main(int argc, char** argv) {
  // write logs instantly (never buffer log messages)
  FLAGS_logbuflevel = -1;

  initMain(argc, argv);
  initPython(argc, argv);

  std::vector<std::unique_ptr<ParameterServer2>> pservers;
  std::vector<std::string> devices;

  if (FLAGS_start_pserver) {
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
          pservers[i].reset(
              new ParameterServer2(std::string(), FLAGS_port + i));
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
  }
  Trainer trainer;
  auto config = TrainerConfigHelper::createFromFlags();
  CHECK(config != nullptr) << "no valid config";

  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  trainer.init(config, FLAGS_job == "test");

  if (FLAGS_job == "train") {
    trainer.train();
  } else if (FLAGS_job == "checkgrad") {
    trainer.checkGradient();
  } else if (FLAGS_job == "test") {
    trainer.test();
  } else if (FLAGS_job == "time") {
    trainer.time();
  } else {
    LOG(FATAL) << "Unknown job type: " << FLAGS_job;
  }

  return 0;
}
