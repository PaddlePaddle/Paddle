// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <future>  // NOLINT
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ThreadPool.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/jit/program_desc_tracer.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/core/compat/arg_map_context.h"

namespace paddle {
namespace imperative {

enum class AmpLevel;

enum class AmpDtype;

using GarbageCollectorMap =
    std::map<platform::Place,
             std::unique_ptr<paddle::framework::GarbageCollector>>;

class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "dygraph_tmp") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

class Tracer {
  DISABLE_COPY_AND_ASSIGN(Tracer);

 public:
  Tracer()
      : basic_engine_(new BasicEngine()),
        program_desc_tracer_(new jit::ProgramDescTracer()),
        generator_(new UniqueNameGenerator()) {
    expected_place_ = platform::CPUPlace();
  }

  ~Tracer() = default;

  template <typename VarType>
  void TraceOp(const std::string& type,
               const NameVarMap<VarType>& ins,
               const NameVarMap<VarType>& outs,
               framework::AttributeMap attrs,
               const platform::Place& place,
               bool trace_backward,
               const std::map<std::string, std::string>& inplace_map = {},
               paddle::framework::AttributeMap* passed_default_attrs_ = nullptr,
               bool use_default_attr_map = true);

  template <typename VarType>
  void TraceOpImpl(
      const std::string& type,
      const NameVarMap<VarType>& ins,
      const NameVarMap<VarType>& outs,
      framework::AttributeMap& attrs,  // NOLINT
      const platform::Place& place,
      bool trace_backward,
      const std::map<std::string, std::string>& inplace_map = {},
      paddle::framework::AttributeMap* passed_default_attrs_ = nullptr,
      bool use_default_attr_map = true);

  void TraceOp(const std::string& type,
               const NameVarBaseMap& ins,
               const NameVarBaseMap& outs,
               framework::AttributeMap attrs,
               const std::map<std::string, std::string>& inplace_map = {});

  void TraceOp(const std::string& type,
               const NameTensorMap& ins,
               const NameTensorMap& outs,
               paddle::framework::AttributeMap& attrs,  // NOLINT
               const std::map<std::string, std::string>& inplace_map = {});

  void TraceOp(const std::string& type,
               const NameTensorMap& ins,
               const NameTensorMap& outs,
               paddle::framework::AttributeMap attrs);

  void TraceOp(const std::string& type,
               const NameTensorMap& ins,
               const NameTensorMap& outs,
               paddle::framework::AttributeMap& attrs,  // NOLINT
               const paddle::platform::Place& place,
               paddle::framework::AttributeMap* default_attrs,
               bool use_default_attr_map,
               const std::map<std::string, std::string>& inplace_map = {});

  bool ComputeRequiredGrad(const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs,
                           bool trace_backward);
  bool ComputeRequiredGrad(const NameTensorMap& ins,
                           const NameTensorMap& outs,
                           bool trace_backward);

  void SetEnableProgramDescTracing(bool enabled) {
    enable_program_desc_tracing_ = enabled;
  }

  bool IsProgramDescTracingEnabled() const {
    return enable_program_desc_tracing_;
  }

  jit::ProgramDescTracer* GetProgramDescTracer() {
    return program_desc_tracer_.get();
  }

  // Note(Aurelius84): The `tmp` is used as prefix key while naming a temporary
  // intermediate var both in imperative and static mode. But the
  // `UniqueNameGenerator` in C++ and `unique_name.py` in Python doesn't share
  // the same auto-increment id. It will create a variable repeatedly with same
  // name like `tmp_0` in some cases when transform dygraph into static layers.
  // So we modify the default prefix key into `eager_tmp` to distinguish with
  // static graph.
  std::string GenerateUniqueName(std::string key = "dygraph_tmp") {
    return generator_->Generate(key);
  }

  BasicEngine* GetEngine() const { return basic_engine_.get(); }

  platform::Place ExpectedPlace() const { return expected_place_; }

  void SetExpectedPlace(platform::Place place);

  bool HasGrad() const { return has_grad_; }

  void SetHasGrad(bool has_grad) { has_grad_ = has_grad; }

  void SetAmpLevel(AmpLevel level) {
    VLOG(4) << "set amp_level to " << static_cast<unsigned int>(level);
    amp_level_ = level;
  }

  AmpLevel GetAmpLevel() const { return amp_level_; }

  void SetAmpDtype(std::string amp_dtype) {
    VLOG(4) << "set amp_dtype to " << amp_dtype;
    if (amp_dtype == "float16") {
      amp_dtype_ = phi::DataType::FLOAT16;
    } else if (amp_dtype == "bfloat16") {
      amp_dtype_ = phi::DataType::BFLOAT16;
    } else {
      amp_dtype_ = phi::DataType::FLOAT32;
    }
  }

  std::string GetAmpDtype() const {
    if (amp_dtype_ == phi::DataType::FLOAT16) {
      return std::string("float16");
    } else if (amp_dtype_ == phi::DataType::BFLOAT16) {
      return std::string("bfloat16");
    } else {
      return std::string("float32");
    }
  }

  phi::KernelSignature GetExpectedKernelSignature(
      const std::string& type,
      const NameTensorMap& ins,
      const NameTensorMap& outs,
      framework::AttributeMap attrs) const;

  paddle::framework::GarbageCollector* MutableGarbageCollectorIfNotExists(
      const platform::Place& place);

 private:
  std::unique_ptr<BasicEngine> basic_engine_;
  std::unique_ptr<jit::ProgramDescTracer> program_desc_tracer_;
  std::unique_ptr<UniqueNameGenerator> generator_;
  platform::Place expected_place_;
  GarbageCollectorMap gcs_;

  static thread_local bool enable_program_desc_tracing_;
  static thread_local bool has_grad_;
  static thread_local AmpLevel amp_level_;
  static thread_local phi::DataType amp_dtype_;
};

// To access static variable current_tracer
const std::shared_ptr<Tracer>& GetCurrentTracer();
void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer_);
void IncreaseVarbaseReferenceCountUntilCopyComplete(
    const std::shared_ptr<imperative::VarBase>& var,
    const platform::Place& place);

void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad);

}  // namespace imperative
}  // namespace paddle
