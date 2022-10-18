/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <pthread.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

class ThreadBarrier {
public:
    ThreadBarrier(int count = 1) {
        PADDLE_ENFORCE_EQ(pthread_barrier_init(&_barrier, NULL, count), 0);
    }
    ~ThreadBarrier() {
        PADDLE_ENFORCE_EQ(pthread_barrier_destroy(&_barrier), 0);
    }
    void reset(int count) {
        PADDLE_ENFORCE_EQ(pthread_barrier_destroy(&_barrier), 0);
        PADDLE_ENFORCE_EQ(pthread_barrier_init(&_barrier, NULL, count), 0);
    }
    void wait() {
        int err = pthread_barrier_wait(&_barrier);
        PADDLE_ENFORCE((err = pthread_barrier_wait(&_barrier), err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD), 
            platform::errors::External("err:%d", err));
    }
private:
    pthread_barrier_t _barrier;
};

}  // end namespace framework
}  // end namespace paddle
