/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

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
#include <vector>
#include <unordered_map>

class MHASeqData {
public:
    std::vector<int> qkvo_seq_len;
    std::vector<int> lo_hi_windows;
};

class MHASeqDataSingleton {
public:
    static MHASeqDataSingleton& Instance() {
        static MHASeqDataSingleton instance;
        return instance;
    }

    MHASeqDataSingleton(MHASeqDataSingleton const&) = delete;
    void operator=(MHASeqDataSingleton const&) = delete;

    MHASeqData& Data(std::string key) { return map_[key]; }

private:
    MHASeqDataSingleton() {}
    std::unordered_map<std::string, MHASeqData> map_;
};