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

#include "PaddleAPI.h"

#include "PaddleAPIPrivate.h"
#ifndef PADDLE_WITHOUT_GOLANG
#include "paddle/trainer/NewRemoteParameterUpdater.h"
#endif
#include "paddle/trainer/RemoteParameterUpdater.h"
#include "paddle/trainer/ThreadParameterUpdater.h"

ParameterUpdater::ParameterUpdater() : m(new ParameterUpdaterPrivate()) {}

ParameterUpdater *ParameterUpdater::createLocalUpdater(
    OptimizationConfig *config) {
  auto updater = new ParameterUpdater();
  updater->m->updater.reset(
      new paddle::SgdThreadUpdater(config->m->getConfig()));
  return updater;
}

ParameterUpdater *ParameterUpdater::createNewRemoteUpdater(
    OptimizationConfig *config,
    const std::string pserverSpec,
    const bool useEtcd) throw(UnsupportError) {
#ifndef PADDLE_WITHOUT_GOLANG
  auto updater = new ParameterUpdater();
  updater->m->updater.reset(new paddle::NewRemoteParameterUpdater(
      config->m->getConfig(), pserverSpec, useEtcd));
  return updater;
#else
  throw UnsupportError("not compiled with WITH_GOLANG");
#endif
}

ParameterUpdater *ParameterUpdater::createRemoteUpdater(
    OptimizationConfig *config, int passCount, bool useSparseUpdater) {
  auto updater = new ParameterUpdater();
  auto remoteUpdater = new paddle::RemoteParameterUpdater(
      config->m->getConfig(), passCount, nullptr);
  if (useSparseUpdater) {
    std::unique_ptr<paddle::ParameterUpdater> remoteUpdaterPtr(remoteUpdater);
    auto sparseRemoteUpdater =
        new paddle::SparseRemoteParameterUpdaterComposite(
            config->m->getConfig(),
            passCount,
            false,
            std::move(remoteUpdaterPtr));
    updater->m->updater.reset(sparseRemoteUpdater);
  } else {
    updater->m->updater.reset(remoteUpdater);
  }
  return updater;
}

ParameterUpdater::~ParameterUpdater() { delete m; }

void ParameterUpdater::init(const GradientMachine &gm) {
  m->updater->init(gm.m->machine->getNonStaticParameters());
}

void ParameterUpdater::startPass() { m->updater->startPass(); }

void ParameterUpdater::finishPass() { m->updater->finishPass(); }

PassType ParameterUpdater::startBatch(size_t batchSize) {
  return m->updater->startBatch((int64_t)batchSize);
}

void ParameterUpdater::finishBatch(float cost) {
  m->updater->finishBatch(cost);
}

void ParameterUpdater::update(Parameter *param) {
  auto paddleParam = param->m->getPtr();
  m->updater->update(paddleParam);
}

void ParameterUpdater::getParametersRemote(bool fullSize, bool apply) {
  m->updater->getParametersRemote(fullSize, apply);
}

void ParameterUpdater::restore() { m->updater->restore(); }

void ParameterUpdater::apply() { m->updater->apply(); }

void ParameterUpdater::catchUpWith() { m->updater->catchUpWith(); }
