/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

namespace paddle {
namespace ps {

class Communicator {
 public:
  Communicator();

  explicit Communicator(const std::map<std::string, std::string> &envs_) {
    VLOG(3) << "Communicator Init Envs";
    for (auto &iter : envs_) {
      envs[iter.first] = iter.second;
      VLOG(3) << iter.first << ": " << iter.second;
    }
    barrier_table_id_ = std::stoi(envs.at("barrier_table_id"));
    trainer_id_ = std::stoi(envs.at("trainer_id"));
    trainers_ = std::stoi(envs.at("trainers"));
  }
  virtual ~Communicator() {}

  virtual void Init() = 0;

  virtual void Start() = 0;

  virtual void Stop() = 0;
 
};

}
}