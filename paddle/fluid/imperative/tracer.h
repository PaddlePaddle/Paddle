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
#include "paddle/common/macros.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/utils/test_macros.h"

COMMON_DECLARE_bool(use_stride_kernel);
namespace paddle {
namespace imperative {

enum class AmpLevel;

enum class AmpDtype;

using GarbageCollectorMap =
    std::map<phi::Place, std::unique_ptr<paddle::framework::GarbageCollector>>;

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
        generator_(new UniqueNameGenerator()) {
    expected_place_ = phi::CPUPlace();
  }

  ~Tracer() = default;

  template <typename VarType>
  void TraceOp(const std::string& type,
               const NameVarMap<VarType>& ins,
               const NameVarMap<VarType>& outs,
               framework::AttributeMap attrs,
               const phi::Place& place,
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
      const phi::Place& place,
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
               const phi::Place& place,
               paddle::framework::AttributeMap* default_attrs,
               bool use_default_attr_map,
               const std::map<std::string, std::string>& inplace_map = {});

  bool ComputeRequiredGrad(const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs,
                           bool trace_backward);
  bool ComputeRequiredGrad(const NameTensorMap& ins,
                           const NameTensorMap& outs,
                           bool trace_backward);

  // Note(Aurelius84): The `tmp` is used as prefix key while naming a temporary
  // intermediate var both in imperative and static graph mode. But the
  // `UniqueNameGenerator` in C++ and `unique_name.py` in Python doesn't share
  // the same auto-increment id. It will create a variable repeatedly with same
  // name like `tmp_0` in some cases when transform dygraph into static layers.
  // So we modify the default prefix key into `eager_tmp` to distinguish with
  // static graph.
  std::string GenerateUniqueName(std::string key = "dygraph_tmp") {
    return generator_->Generate(key);
  }

  BasicEngine* GetEngine() const { return basic_engine_.get(); }

  phi::Place ExpectedPlace() const { return expected_place_; }

  TEST_API void SetExpectedPlace(phi::Place place);

  TEST_API bool HasGrad() const;

  TEST_API void SetHasGrad(bool has_grad);

  TEST_API void SetUsePromote(bool use_promote);

  TEST_API bool GetUsePromote() const;

  TEST_API void SetAmpLevel(AmpLevel level);

  TEST_API AmpLevel GetAmpLevel() const;

  void SetAmpDtype(std::string amp_dtype);

  std::string GetAmpDtype() const;

  phi::DataType GetAmpPhiDtype() const;

  TEST_API void DisableLayoutAutoTune();

  TEST_API void EnableLayoutAutoTune();

  TEST_API bool UseLayoutAutoTune();
  TEST_API void SetPythonStack(std::string stack_str);
  TEST_API std::string GetPythonStack();
  phi::KernelSignature GetExpectedKernelSignature(
      const std::string& type,
      const NameTensorMap& ins,
      const NameTensorMap& outs,
      framework::AttributeMap attrs) const;

  paddle::framework::GarbageCollector* MutableGarbageCollectorIfNotExists(
      const phi::Place& place);

 private:
  std::unique_ptr<BasicEngine> basic_engine_;
  std::unique_ptr<UniqueNameGenerator> generator_;
  phi::Place expected_place_;
  GarbageCollectorMap gcs_;
  static thread_local std::string python_stack_;
  static thread_local bool enable_program_desc_tracing_;
  static thread_local bool use_layout_autotune_;
};

// To access static variable current_tracer
const std::shared_ptr<Tracer>& GetCurrentTracer();
TEST_API void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer_);
const std::shared_ptr<AmpAttrs>& GetCurrentAmpAttrs();
void IncreaseVarbaseReferenceCountUntilCopyComplete(
    const std::shared_ptr<imperative::VarBase>& var, const phi::Place& place);

void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad);

}  // namespace imperative
}  // namespace paddle
