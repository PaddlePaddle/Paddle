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
#include <iostream>
#include <memory>
#include <vector>

#include "paddle/ir/core/program.h"

namespace ir {

class IrContext;
class Operation;
class Program;
class Pass;
class PassInstrumentation;
class PassInstrumentor;

namespace detail {
class PassAdaptor;
}

class PassManager {
 public:
  explicit PassManager(IrContext *context, uint8_t opt_level = 2);

  ~PassManager() = default;

  const std::vector<std::unique_ptr<Pass>> &passes() const { return passes_; }

  bool Empty() const { return passes_.empty(); }

  IrContext *context() const { return context_; }

  bool Run(Program *program);

  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

  class IRPrinterOption {
   public:
    using PrintCallBack = std::function<void(std::ostream &)>;

    explicit IRPrinterOption(
        const std::function<bool(Pass *, Operation *)> &enable_print_before =
            [](Pass *, Operation *) { return true; },
        const std::function<bool(Pass *, Operation *)> &enable_print_after =
            [](Pass *, Operation *) { return true; },
        bool print_module = true,
        bool print_on_change = true,
        std::ostream &os = std::cout)
        : enable_print_before_(enable_print_before),
          enable_print_after_(enable_print_after),
          print_module_(print_module),
          print_on_change_(print_on_change),
          os(os) {
      assert((enable_print_before_ || enable_print_after_) &&
             "expected at least one valid filter function");
    }

    ~IRPrinterOption() = default;

    void PrintBeforeIfEnabled(Pass *pass,
                              Operation *op,
                              const PrintCallBack &print_callback) {
      if (enable_print_before_ && enable_print_before_(pass, op)) {
        print_callback(os);
      }
    }

    void PrintAfterIfEnabled(Pass *pass,
                             Operation *op,
                             const PrintCallBack &print_callback) {
      if (enable_print_after_ && enable_print_after_(pass, op)) {
        print_callback(os);
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

    std::ostream &os;

    // TODO(liuyuanle): Add flags to control printing behavior.
  };

  void EnableIRPrinting(std::unique_ptr<IRPrinterOption> config);

  void EnablePassTiming(bool print_module = true);

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

 private:
  bool Initialize(IrContext *context);

  bool Run(Operation *op);

 private:
  IrContext *context_;

  uint8_t opt_level_;

  bool verify_{true};

  std::vector<std::unique_ptr<Pass>> passes_;

  std::unique_ptr<Pass> pass_adaptor_;

  std::unique_ptr<PassInstrumentor> instrumentor_;

  // For access member of pass_adaptor_.
  friend class detail::PassAdaptor;
};

}  // namespace ir
