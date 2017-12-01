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

#include "ThreadLocal.h"
#include "Common.h"

#include <gflags/gflags.h>

DEFINE_bool(thread_local_rand_use_global_seed,
            false,
            "Whether to use global seed in thread local rand.");

namespace paddle {
// namespace framework {

unsigned int ThreadLocalRand::defaultSeed_ = 1;
ThreadLocal<unsigned int> ThreadLocalRand::seed_;

unsigned int* ThreadLocalRand::getSeed() {
  unsigned int* p = seed_.get(false /*createLocal*/);
  if (!p) {  // init seed
    if (FLAGS_thread_local_rand_use_global_seed) {
      p = new unsigned int(defaultSeed_);
    } else if (getpid() == getTID()) {  // main thread
      // deterministic, but differs from global srand()
      p = new unsigned int(defaultSeed_ - 1);
    } else {
      p = new unsigned int(defaultSeed_ + getTID());
      VLOG(3) << "thread use undeterministic rand seed:" << *p;
    }
    seed_.set(p);
  }
  return p;
}

ThreadLocal<std::default_random_engine> ThreadLocalRandomEngine::engine_;
std::default_random_engine& ThreadLocalRandomEngine::get() {
  auto engine = engine_.get(false);
  if (!engine) {
    engine = new std::default_random_engine;
    int defaultSeed = ThreadLocalRand::getDefaultSeed();
    engine->seed(FLAGS_thread_local_rand_use_global_seed
                     ? defaultSeed
                     : defaultSeed + getTID());
    engine_.set(engine);
  }
  return *engine;
}

pid_t getTID() {
#if defined(__APPLE__) || defined(__OSX__)
  // syscall is deprecated: first deprecated in macOS 10.12.
  // syscall is unsupported;
  // syscall pid_t tid = syscall(SYS_thread_selfid);
  uint64_t tid;
  pthread_threadid_np(NULL, &tid);
#else
#ifndef __NR_gettid
#define __NR_gettid 224
#endif
  pid_t tid = syscall(__NR_gettid);
#endif
  CHECK_NE((int)tid, -1);
  return tid;
}

// }  // namespace framework
}  // namespace paddle
