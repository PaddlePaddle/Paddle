// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <paddle/fluid/framework/op_registry.h>

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "glog/logging.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#include <chrono>

using namespace paddle;
using namespace imperative;

inline void benchmark_fluid_accuracy_check(std::shared_ptr<imperative::VarBase>& X, std::shared_ptr<imperative::VarBase>& Out) {
    imperative::Tracer tracer;

    framework::AttributeMap attrs;
    
    attrs["use_mkldnn"] = false;
    attrs["scale"] = 2;
    attrs["bias"] = 3;
    attrs["bias_after_scale"] = true;

    // NameVarBaseMap = std::map<std::string, std::vector<std::shared_ptr<VarBase>>>
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};

    platform::CPUPlace place;
    
    size_t max_num_runs = 10;
    for(size_t i = 0; i < max_num_runs; i++) {
        tracer.TraceOp("scale", ins, outs, attrs, place, true);
        if(i != max_num_runs - 1) {
            ins = {{ "X", outs["Out"] }};
            outs = {{"Out", { std::shared_ptr<imperative::VarBase>(new imperative::VarBase(true, "Out")) }}};
        }
    }
    
    auto *engine = tracer.GetEngine();
    std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
    engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
    engine->Execute();
    
    // Fwd Check: Expects 8189 with max_num_runs = 10
    auto* tensor = outs["Out"][0]->MutableVar()->GetMutable<framework::LoDTensor>(); 
    float* t_ptr = tensor->mutable_data<float>(place);
    PADDLE_ENFORCE(t_ptr[0] == 8189.0, 
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 8189.0));

    // Grad Check: Expects 1024.0 with max_num_runs = 10
    auto* grad_tensor = X->MutableGradVar()->GetMutable<framework::LoDTensor>(); 
    float* g_ptr = grad_tensor->mutable_data<float>(place);
    PADDLE_ENFORCE(g_ptr[0] == 1024.0, 
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1024.0));
    
}


inline void benchmark_fluid(std::shared_ptr<imperative::VarBase>& X, std::shared_ptr<imperative::VarBase>& Out) {
    imperative::Tracer tracer;

    framework::AttributeMap attrs;
    
    attrs["use_mkldnn"] = false;
    attrs["scale"] = 2;
    attrs["bias"] = 3;
    attrs["bias_after_scale"] = true;

    // NameVarBaseMap = std::map<std::string, std::vector<std::shared_ptr<VarBase>>>
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};

    platform::CPUPlace place;
    
    size_t max_num_runs = 5000;
    for(size_t i = 0; i < max_num_runs; i++) {
        tracer.TraceOp("scale", ins, outs, attrs, place, true);
        if(i != max_num_runs - 1) {
            ins = {{ "X", outs["Out"] }};
            outs = {{"Out", { std::shared_ptr<imperative::VarBase>(new imperative::VarBase(true, "Out")) }}};
        }
    }
    
    auto *engine = tracer.GetEngine();
    std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
    engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
    engine->Execute();
}

/*
TEST(Benchmark, FluidAccuracy) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(new imperative::VarBase(true, "Out"));
    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2,4,4,4};
    platform::CPUPlace place;
  
    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    benchmark_fluid_accuracy_check(X, Out);
}
*/

TEST(Benchmark, FluidPerformance) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::shared_ptr<imperative::VarBase> Out(new imperative::VarBase(true, "Out"));
    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2,4,4,4};
    platform::CPUPlace place;
  
    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    auto t_start = std::chrono::high_resolution_clock::now();
    
    benchmark_fluid(X, Out);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;
}

USE_OP(scale);

