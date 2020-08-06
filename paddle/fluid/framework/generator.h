/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include <ThreadPool.h>
#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

class Generator {
 public:
  Generator() {}
  virtual ~Generator() {}
  virtual void SetCurrentSeed(uint64_t seed) = 0;
  virtual uint64_t CurrentSeed() const = 0;
  virtual uint64_t Seed() = 0;
  virtual uint64_t SetState() = 0;
  virtual Tensor GetState() = 0;
};

// DatasetImpl is the implementation of Dataset,
// it holds memory data if user calls load_into_memory
template <typename T>
class CPUGeneratorImpl : public Generator {
 public:
  CPUGeneratorImpl();
  virtual ~CPUGeneratorImpl() {}

  virtual void SetCurrentSeed(uint64_t seed);
  virtual uint64_t CurrentSeed();
  virtual uint64_t Seed();
  virtual uint64_t SetState();
  virtual Tensor GetState();

 private:
};

}  // end namespace framework
}  // end namespace paddle
