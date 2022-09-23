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

#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"

#include <gtest/gtest.h>
#include <llvm/Support/FormatVariadic.h>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/host_context/mlir_program_executor.h"
#include "paddle/infrt/kernel/basic_kernels.h"
#include "paddle/infrt/kernel/control_flow_kernels.h"
#include "paddle/infrt/kernel/tensor_kernels.h"
#include "paddle/infrt/kernel/tensor_shape_kernels.h"
#include "paddle/infrt/kernel/test_kernels.h"

namespace infrt {
namespace host_context {

TEST(MlirToRuntimeTranslate, basic) {
  mlir::MLIRContext context;

  auto source = R"ROC(
func @main() -> () {
  %v0 = infrt.constant.f32 1.0
  %v1 = infrt.constant.f32 2.0
  %v2 = "infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32
  %v3 = "infrt.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  "infrt.print.f32"(%v1) : (f32) -> ()

  infrt.return
}
)ROC";

  auto module = dialect::LoadMlirSource(&context, source);
  EXPECT_TRUE(mlir::succeeded(module->verify()));

  KernelRegistry registry;
  kernel::RegisterFloatBasicKernels(&registry);
  kernel::RegisterIntBasicKernels(&registry);

  TestMlir(module.get(), &registry);
}

TEST(TestMlir, basic) {
  mlir::MLIRContext context;

  auto source = R"ROC(
func @main() -> () {
  %v0 = infrt.constant.f32 1.0
  %v1 = infrt.constant.f32 2.0
  %v2 = "infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32
  %v3 = "infrt.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  "infrt.print.f32"(%v1) : (f32) -> ()

  infrt.return
}
)ROC";

  auto module = dialect::LoadMlirSource(&context, source);
  EXPECT_TRUE(mlir::succeeded(module->verify()));

  KernelRegistry registry;
  kernel::RegisterFloatBasicKernels(&registry);
  kernel::RegisterIntBasicKernels(&registry);

  TestMlir(module.get(), &registry);
}

TEST(TestMlir, shadow_copy_tensor_profile) {
  mlir::MLIRContext* context = infrt::Global::getMLIRContext();

  auto head = R"ROC(
func @predict(%a: !infrt.dense_tensor<CPU, FP32, NCHW>, %b: !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) {
)ROC";

  auto tpl0 =
      "%a{0} = dt.shallow_copy_tensor %a : !infrt.dense_tensor<CPU, FP32, "
      "NCHW> -> "
      "!infrt.dense_tensor<CPU, FP32, NCHW>";
  auto tpl1 =
      "%b{0} = dt.shallow_copy_tensor %b : !infrt.dense_tensor<CPU, FP32, "
      "NCHW> -> "
      "!infrt.dense_tensor<CPU, FP32, NCHW>";

  auto end = R"ROC(
infrt.return %a0, %b0: !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>
}
  )ROC";

  std::stringstream ss;
  ss << head;
  for (int i = 0; i < 2000; i++) {
    ss << llvm::formatv(tpl0, i).str() << "\n";
    ss << llvm::formatv(tpl1, i).str() << "\n";
  }
  ss << end;

  auto content = ss.str();

  // LOG(INFO) << "content: " << content << std::endl;

  auto module = dialect::LoadMlirSource(context, content);
  EXPECT_TRUE(mlir::succeeded(module->verify()));

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTestKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);
  kernel::RegisterControlFlowKernels(&registry);

  MlirProgramExecutor executor(*module, &registry);
  executor.BuildFunctions();

  auto* func = executor.LookupFunc("predict");
  ASSERT_TRUE(func);

  std::vector<Value*> in_args;
  std::vector<ValueRef> out_args(
      {ValueRef(new Value(tensor::DenseHostTensor())),
       ValueRef(new Value(tensor::DenseHostTensor()))});

  auto create_tensor = [] {
    tensor::DenseHostTensor a(tensor::TensorShape{{200, 3000}},
                              DType(DType::Kind::F32));
    auto* data = reinterpret_cast<float*>(a.raw_data());
    for (int i = 0; i < a.shape().GetNumElements(); i++) {
      data[i] = i;
    }
    return a;
  };

  std::vector<ValueRef> inputs({ValueRef(new Value(create_tensor())),
                                ValueRef(new Value(create_tensor()))});
  in_args.assign({inputs[0].get(), inputs[1].get()});

  for (int i = 0; i < 500; i++) {
    func->Execute(
        llvm::ArrayRef<Value*>(in_args.data(), in_args.size()),
        llvm::MutableArrayRef<ValueRef>(out_args.data(), out_args.size()));
  }
}

}  // namespace host_context
}  // namespace infrt
