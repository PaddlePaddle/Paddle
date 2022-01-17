/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/custom_kernel.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/extension.h"
#include "paddle/fluid/framework/op_kernel_info_helper.h"
#include "paddle/pten/core/kernel_factory.h"

TEST(OpKernelInfoHelper, GetOpName) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::INT8;

  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);

  EXPECT_TRUE(op_name_1 == paddle::framework::OpKernelInfoHelper::GetOpName(
                               op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetKernelKey) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);

  EXPECT_TRUE(
      pten::KernelKey(backend, layout, dtype) ==
      paddle::framework::OpKernelInfoHelper::GetKernelKey(op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetKernelFn) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  paddle::CustomKernelFunc kernel_fn{nullptr};
  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);
  op_kernel_info_1.SetKernelFn(std::move(kernel_fn));

  EXPECT_TRUE(kernel_fn == paddle::framework::OpKernelInfoHelper::GetKernelFn(
                               op_kernel_info_1));
}

TEST(OpKernelInfoHelper, GetVariadicKernelFn) {
  std::string op_name_1 = "dot";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::FLOAT32;

  void* variadic_func{nullptr};
  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);
  op_kernel_info_1.SetVariadicKernelFn(std::move(variadic_func));

  EXPECT_TRUE(variadic_func ==
              paddle::framework::OpKernelInfoHelper::GetVariadicKernelFn(
                  op_kernel_info_1));
}

TEST(CustomKernel, RegisterKernelWithMetaInfoMap) {
  std::string op_name_1 = "dot";
  std::string op_name_2 = "add";
  pten::Backend backend = pten::Backend::CPU;
  pten::DataLayout layout = pten::DataLayout::ANY;
  pten::DataType dtype = pten::DataType::INT8;

  auto op_kernel_info_1 =
      paddle::OpKernelInfo(op_name_1, backend, layout, dtype);
  paddle::OpKernelInfoMap::Instance()[op_name_1].emplace_back(
      std::move(op_kernel_info_1));
  auto op_kernel_info_2 =
      paddle::OpKernelInfo(op_name_2, backend, layout, dtype);
  paddle::OpKernelInfoMap::Instance()[op_name_2].emplace_back(
      std::move(op_kernel_info_2));

  // has CompatiblePtenKernel
  EXPECT_TRUE(pten::KernelFactory::Instance().kernels().end() !=
              pten::KernelFactory::Instance().kernels().find("dot"));
  EXPECT_TRUE(pten::KernelFactory::Instance().kernels().end() !=
              pten::KernelFactory::Instance().kernels().find("add"));

  // key not exist
  pten::KernelKey kernel_key(backend, layout, dtype);
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key) ==
      pten::KernelFactory::Instance().kernels()["dot"].end());
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["add"].find(kernel_key) ==
      pten::KernelFactory::Instance().kernels()["add"].end());

  paddle::framework::RegisterKernelWithMetaInfoMap(
      paddle::OpKernelInfoMap::Instance());

  // registed
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["dot"].find(kernel_key) !=
      pten::KernelFactory::Instance().kernels()["dot"].end());
  EXPECT_TRUE(
      pten::KernelFactory::Instance().kernels()["add"].find(kernel_key) !=
      pten::KernelFactory::Instance().kernels()["add"].end());
}
