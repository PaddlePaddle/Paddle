/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

namespace phi {
namespace tests {

// TODO(chenweihang): add more unittests later

TEST(KernelKey, ConstructAndOStream) {
  phi::KernelKey key(
      phi::Backend::CPU, phi::DataLayout::NCHW, phi::DataType::FLOAT32);
  EXPECT_EQ(key.backend(), phi::Backend::CPU);
  EXPECT_EQ(key.layout(), phi::DataLayout::NCHW);
  EXPECT_EQ(key.dtype(), phi::DataType::FLOAT32);
  std::ostringstream oss;
  oss << key;
  std::cout << oss.str();
  oss.flush();
}

TEST(KernelFactory, SelectedKernelMap) {
  auto kernel_map = phi::KernelFactory::Instance().SelectKernelMap("scale");
  EXPECT_GT(kernel_map.size(), 1UL);
  for (auto& iter : kernel_map) {
    std::cout << iter.first << ": " << iter.second;
  }
}

template <typename T, typename Context>
void TestKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& param,
                DenseTensor* out) {}

TEST(KernelRegistry, SetFP32Input) {
  phi::KernelKey kernel_key(
      phi::Backend::CPU, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT16);
  auto test_kernel =
      phi::KernelFactory::Instance().SelectKernel("test", kernel_key);
  EXPECT_TRUE(test_kernel.IsValid());
  auto& arg_defs = test_kernel.args_def();
  auto& input_defs = arg_defs.input_defs();
  auto& attr_defs = arg_defs.attribute_defs();
  auto& output_defs = arg_defs.output_defs();
  EXPECT_EQ(input_defs.size(), 2UL);
  EXPECT_EQ(attr_defs.size(), 0UL);
  EXPECT_EQ(output_defs.size(), 1UL);
  EXPECT_EQ(input_defs.at(0).dtype, phi::DataType::FLOAT16);
  EXPECT_EQ(input_defs.at(1).dtype, phi::DataType::FLOAT32);
  EXPECT_EQ(output_defs.at(0).dtype, phi::DataType::FLOAT16);
}

TEST(AttributeType, OStream) {
  std::ostringstream oss;
  oss << phi::AttributeType::UNDEFINED;
  EXPECT_EQ(oss.str(), "Undefined");
  oss.str("");
  oss << phi::AttributeType::BOOL;
  EXPECT_EQ(oss.str(), "bool");
  oss.str("");
  oss << phi::AttributeType::INT32;
  EXPECT_EQ(oss.str(), "int");
  oss.str("");
  oss << phi::AttributeType::INT64;
  EXPECT_EQ(oss.str(), "int64_t");
  oss.str("");
  oss << phi::AttributeType::FLOAT32;
  EXPECT_EQ(oss.str(), "float");
  oss.str("");
  oss << phi::AttributeType::FLOAT64;
  EXPECT_EQ(oss.str(), "double");
  oss.str("");
  oss << phi::AttributeType::STRING;
  EXPECT_EQ(oss.str(), "string");
  oss.str("");
  oss << phi::AttributeType::BOOLS;
  EXPECT_EQ(oss.str(), "vector<bool>");
  oss.str("");
  oss << phi::AttributeType::INT32S;
  EXPECT_EQ(oss.str(), "vector<int>");
  oss.str("");
  oss << phi::AttributeType::INT64S;
  EXPECT_EQ(oss.str(), "vector<int64_t>");
  oss.str("");
  oss << phi::AttributeType::FLOAT32S;
  EXPECT_EQ(oss.str(), "vector<float>");
  oss.str("");
  oss << phi::AttributeType::FLOAT64S;
  EXPECT_EQ(oss.str(), "vector<double>");
  oss.str("");
  oss << phi::AttributeType::STRINGS;
  EXPECT_EQ(oss.str(), "vector<string>");
  oss.str("");
  oss << phi::AttributeType::SCALAR;
  EXPECT_EQ(oss.str(), "Scalar");
  oss.str("");
  oss << phi::AttributeType::SCALARS;
  EXPECT_EQ(oss.str(), "vector<Scalar>");
  oss.str("");
  oss << phi::AttributeType::INT_ARRAY;
  EXPECT_EQ(oss.str(), "IntArray");
  oss.str("");
  oss << phi::AttributeType::DATA_TYPE;
  EXPECT_EQ(oss.str(), "DataType");
  oss.str("");
  oss << phi::AttributeType::DATA_LAYOUT;
  EXPECT_EQ(oss.str(), "DataLayout");
  oss.str("");
  oss << phi::AttributeType::PLACE;
  EXPECT_EQ(oss.str(), "Place");
  oss.str("");
}

}  // namespace tests
}  // namespace phi

PD_REGISTER_KERNEL(test,
                   CPU,
                   ALL_LAYOUT,
                   phi::tests::TestKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
}
