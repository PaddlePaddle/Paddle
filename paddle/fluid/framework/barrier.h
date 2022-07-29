// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace paddle {
namespace framework {

class Barrier {
public:
    explicit Barrier(int count = 1) {
        CHECK(count >= 1);
        PCHECK(0 == pthread_barrier_init(&_barrier, NULL, count));
    }
    ~Barrier() {
        PCHECK(0 == pthread_barrier_destroy(&_barrier));
    }
    void reset(int count) {
        CHECK(count >= 1);
        PCHECK(0 == pthread_barrier_destroy(&_barrier));
        PCHECK(0 == pthread_barrier_init(&_barrier, NULL, count));
    }
    void wait() {
        int err = pthread_barrier_wait(&_barrier);
        PCHECK((err = pthread_barrier_wait(&_barrier), err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD));
    }
private:
    pthread_barrier_t _barrier;
    DISABLE_COPY_AND_ASSIGN(Barrier);
};

}
}
