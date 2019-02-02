/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/trainer_factory.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {

trainerMap g_trainer_map;

std::string TrainerFactory::TrainerTypeList() {
  std::string trainer_types;
  for (auto iter = g_trainer_map.begin(); iter != g_trainer_map.end(); ++iter) {
    if (iter != g_trainer_map.begin()) {
      trainer_types += ", ";
    }
    trainer_types += iter->first;
  }
  return trainer_types;
}

std::shared_ptr<TrainerBase> TrainerFactory::CreateTrainer(
    std::string trainer_class) {
  if (g_trainer_map.count(trainer_class) < 1) {
    exit(-1);
  }
  return g_trainer_map[trainer_class]();
}

}  // namespace framework
}  // namespace paddle
