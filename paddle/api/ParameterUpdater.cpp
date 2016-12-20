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

#include "PaddleAPI.h"

#include "PaddleAPIPrivate.h"
#include "paddle/trainer/ThreadParameterUpdater.h"

ParameterUpdater::ParameterUpdater() : m(new ParameterUpdaterPrivate()) {}

ParameterUpdater *ParameterUpdater::createLocalUpdater(
    OptimizationConfig *config) {
  auto param = new ParameterUpdater();
  param->m->updater.reset(new paddle::SgdThreadUpdater(config->m->getConfig()));
  return param;
}

ParameterUpdater::~ParameterUpdater() { delete m; }

void ParameterUpdater::init(const GradientMachine &gm) {
  m->updater->init(gm.m->machine->getParameters());
}

void ParameterUpdater::startPass() { m->updater->startPass(); }

void ParameterUpdater::finishPass() {}
