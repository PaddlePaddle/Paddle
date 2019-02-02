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

#pragma once

#include <memory>
#include <string>
#include "paddle/fluid/framework/trainer.h"

namespace paddle {
namespace framework {
typedef std::shared_ptr<TrainerBase> (*CreatetrainerFunction)();
typedef std::unordered_map<std::string, CreatetrainerFunction> trainerMap;
extern trainerMap g_trainer_map;

#define REGISTER_TRAINER_CLASS(trainer_class)                   \
  namespace {                                                   \
  std::shared_ptr<TrainerBase> Creator_##trainer_class() {      \
    return std::shared_ptr<TrainerBase>(new trainer_class);     \
  }                                                             \
  class __Registerer_##trainer_class {                          \
   public:                                                      \
    __Registerer_##trainer_class() {                            \
      g_trainer_map[#trainer_class] = &Creator_##trainer_class; \
    }                                                           \
  };                                                            \
  __Registerer_##trainer_class g_registerer_##trainer_class;    \
  }  // namespace

class TrainerFactory {
 public:
  static std::string TrainerTypeList();
  static std::shared_ptr<TrainerBase> CreateTrainer(std::string trainer_class);
};
}  // namespace framework
}  // namespace paddle
