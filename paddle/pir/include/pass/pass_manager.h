// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <vector>

#include "paddle/pir/include/pass/pass.h"

namespace pir {

class IrContext;
class Operation;
class Program;
class PassInstrumentation;
class PassInstrumentor;

namespace detail {
class PassAdaptor;
}

class IR_API PassManager {
 public:
  explicit PassManager(IrContext *context, uint8_t opt_level = 2);

  ~PassManager() = default;

  const std::vector<std::unique_ptr<Pass>> &passes() const { return passes_; }

  bool empty() const { return passes_.empty(); }

  void clear() { passes_.clear(); }

  IrContext *context() const { return context_; }

  bool Run(Program *program);

  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

  class IRPrinterOption {
   public:
    using PrintCallBack = std::function<void()>;

    explicit IRPrinterOption(
        const std::function<bool(Pass *, Operation *)> &enable_print_before =
            [](Pass *, Operation *) { return true; },
        const std::function<bool(Pass *, Operation *)> &enable_print_after =
            [](Pass *, Operation *) { return true; },
        bool print_module = true,
        bool print_on_change = true)
        : enable_print_before_(enable_print_before),
          enable_print_after_(enable_print_after),
          print_module_(print_module),
          print_on_change_(print_on_change) {
      assert((enable_print_before_ || enable_print_after_) &&
             "expected at least one valid filter function");
    }

    ~IRPrinterOption() = default;

    void PrintBeforeIfEnabled(Pass *pass,
                              Operation *op,
                              const PrintCallBack &print_callback) {
      if (enable_print_before_ && enable_print_before_(pass, op)) {
        print_callback();
      }
    }

    void PrintAfterIfEnabled(Pass *pass,
                             Operation *op,
                             const PrintCallBack &print_callback) {
      if (enable_print_after_ && enable_print_after_(pass, op)) {
        print_callback();
      }
    }

    bool print_module() const { return print_module_; }

    bool print_on_change() const { return print_on_change_; }

   private:
    // The enable_print_before_ and enable_print_after_ can be used to specify
    // the pass to be printed. The default is to print all passes.
    std::function<bool(Pass *, Operation *)> enable_print_before_;
    std::function<bool(Pass *, Operation *)> enable_print_after_;

    bool print_module_;

    bool print_on_change_;

    // TODO(liuyuanle): Add flags to control printing behavior.
  };

  void EnableIRPrinting(std::unique_ptr<IRPrinterOption> option =
                            std::make_unique<IRPrinterOption>());

  void EnablePassTiming(bool print_module = true);

  void EnablePrintStatistics();

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

  void SetValueReplacedHook(const VALUE_REPLACED_HOOK_FUNC &hook) {
    value_replaced_hook_ = hook;
  }

 private:
  bool Initialize(IrContext *context);

  bool Run(Operation *op);

 private:
  IrContext *context_;

  uint8_t opt_level_;

  bool verify_{true};

  bool disable_log_{false};

  std::vector<std::unique_ptr<Pass>> passes_;

  std::unique_ptr<Pass> pass_adaptor_;

  std::unique_ptr<PassInstrumentor> instrumentor_;

  VALUE_REPLACED_HOOK_FUNC value_replaced_hook_ = nullptr;

  // For access member of pass_adaptor_.
  friend class detail::PassAdaptor;
};

}  // namespace pir
