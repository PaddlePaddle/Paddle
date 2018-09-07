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

#include "paddle/parameter/ParameterOptimizer.h"
#include <algorithm>
#include "Internal.h"
#include "PaddleAPI.h"
#include "PaddleAPIPrivate.h"

struct ParameterOptimizerPrivate {
  std::unique_ptr<paddle::ParameterOptimizer> optimizer;
};

struct ParameterTraverseCallbackPrivate {
  paddle::ParameterOptimizer::TraverseCallback callback;

  ParameterTraverseCallbackPrivate() {}

  ParameterTraverseCallbackPrivate(
      const paddle::ParameterOptimizer::TraverseCallback& callback)
      : callback(callback) {}

  void apply(const std::vector<Vector*>& vecs,
             const ParameterConfig& conf,
             size_t sparseId) {
    std::vector<paddle::VectorPtr> real_vecs;
    real_vecs.resize(vecs.size());
    std::transform(vecs.begin(), vecs.end(), real_vecs.begin(), [](Vector* v) {
      if (v) {
        return *(paddle::VectorPtr*)(v->getSharedPtr());
      } else {
        return paddle::VectorPtr();
      }
    });

    paddle::ParameterConfig& real_conf =
        *(paddle::ParameterConfig*)(const_cast<ParameterConfig&>(conf)
                                        .getRawPtr());
    callback(real_vecs.data(), real_conf, sparseId);
  }
};

ParameterOptimizer::ParameterOptimizer() : m(new ParameterOptimizerPrivate()) {}

ParameterOptimizer::~ParameterOptimizer() { delete m; }

ParameterOptimizer* ParameterOptimizer::create(OptimizationConfig* config) {
  CHECK(config != nullptr);
  auto retOptimizer = new ParameterOptimizer();
  retOptimizer->m->optimizer.reset(
      paddle::ParameterOptimizer::create(config->m->getConfig(), false));
  return retOptimizer;
}

void ParameterOptimizer::init(size_t numRows, const ParameterConfig* config) {
  auto& conf = *(paddle::ParameterConfig*)(const_cast<ParameterConfig*>(config)
                                               ->getRawPtr());
  m->optimizer->init(numRows, &conf);
}

void ParameterOptimizer::startPass() { m->optimizer->startPass(); }

void ParameterOptimizer::finishPass() { m->optimizer->finishPass(); }

void ParameterOptimizer::startBatch(size_t numSamplesProcessed) {
  constexpr size_t high_1 = 1UL << (sizeof(size_t) * 8 - 1);
  CHECK_EQ(numSamplesProcessed & high_1, 0UL);  // Safely cast.
  m->optimizer->startBatch((int64_t)numSamplesProcessed);
}

void ParameterOptimizer::finishBatch() { m->optimizer->finishBatch(); }

void ParameterOptimizer::update(const std::vector<Vector*>& vecs,
                                const ParameterConfig& conf,
                                size_t sparseId) {
  ParameterTraverseCallbackPrivate invoker(
      [&](const paddle::VectorPtr _vecs[],
          const paddle::ParameterConfig& config,
          size_t sid = -1UL) { m->optimizer->update(_vecs, config, sid); });
  invoker.apply(vecs, conf, sparseId);
}

std::vector<int> ParameterOptimizer::getParameterTypes() const {
  std::vector<int> returnValue;
  staticCastVector(&returnValue, m->optimizer->getParameterTypes());
  return returnValue;
}

ParameterTraverseCallback::ParameterTraverseCallback()
    : m(new ParameterTraverseCallbackPrivate()) {}

ParameterTraverseCallback::~ParameterTraverseCallback() { delete m; }

void ParameterTraverseCallback::apply(const std::vector<Vector*>& vecs,
                                      const ParameterConfig& conf,
                                      size_t sparseId) {
  m->apply(vecs, conf, sparseId);
}

ParameterTraverseCallback* ParameterOptimizer::needSpecialTraversal(
    const ParameterConfig& config) const {
  auto& param_config =
      *(paddle::ParameterConfig*)const_cast<ParameterConfig&>(config)
           .getRawPtr();
  auto callback = m->optimizer->needSpecialTraversal(param_config);
  if (callback) {
    auto retCallback = new ParameterTraverseCallback();
    retCallback->m->callback = callback;
    return retCallback;
  } else {
    return nullptr;
  }
}
