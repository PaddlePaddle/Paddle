// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <mutex>

template <typename T>
class SingletonBase {
public:
    static T &getInstance() {
        // Use double-checked locking to ensure thread safety.
        if (instance_ == nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance_ == nullptr) {
                instance_ = new T;
                atexit(cleanup);
            }
        }
        return *instance_;
    }

    // Disable copy constructor and assignment operator.
    SingletonBase(const SingletonBase &) = delete;
    SingletonBase &operator=(const SingletonBase &) = delete;

protected:
    // Constructors and destructors are protected to ensure access only through derived classes.
    SingletonBase() {}

    virtual ~SingletonBase() {}

private:
    static void cleanup() {
        if (instance_ != nullptr) {
            delete instance_;
            instance_ = nullptr;
        }
    }

    static T *instance_;
    static std::mutex mutex_;
};

template <typename T>
T *SingletonBase<T>::instance_ = nullptr;

template <typename T>
std::mutex SingletonBase<T>::mutex_;
