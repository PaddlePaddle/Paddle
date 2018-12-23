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
#include <sys/time.h>
#include <pmmintrin.h>
#include <string>
#include <vector>

namespace paddle {
namespace platform {

// A Standard Timer implementation for debugging

class Timer {
 public:
    Timer() {
        reset();
    }

    inline void reset() {
        _start.tv_sec = 0;
        _start.tv_usec = 0;

        _count = 0;
        _elapsed = 0;
        _paused = true;
    }

    inline void start() {
        reset();
        resume();
    }

    inline void pause() {
        if (_paused) {
            return;
        }
        _elapsed += tickus();
        ++_count;
        _paused = true;
    }

    inline void resume() {
        gettimeofday(&_start, NULL);
        _paused = false;
    }

    inline int count() const {
        return _count;
    }

    inline double elapsed_us() const {
        return static_cast<double>(_elapsed);
    }
    inline double elapsed_ms() const {
        return _elapsed / 1000.0;
    }
    inline double elapsed_sec() const {
        return _elapsed / 1000000.0;
    }

 private:
    struct timeval _start;
    struct timeval _now;

    int32_t _count;
    int64_t _elapsed;
    bool _paused;

    inline int64_t tickus() {
        gettimeofday(&_now, NULL);
        return (_now.tv_sec - _start.tv_sec) * 1000 * 1000L +
            (_now.tv_usec - _start.tv_usec);
    }
};
