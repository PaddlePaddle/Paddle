// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <torch/all.h>
#include <torch/library.h>

#include "gtest/gtest.h"

at::Tensor mymuladd_cpu(at::Tensor a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

template <typename T>
T generic_add(T a, T b) {
  return a + b;
}

class TestClass : public torch::CustomClassHolder {
 public:
  int value;
  std::string name;

  TestClass() : value(0), name("default") {
    std::cout << "TestClass::TestClass() - Default constructor" << std::endl;
  }

  TestClass(int v) : value(v), name("single_param") {  // NOLINT
    std::cout << "TestClass::TestClass(int) - Single parameter constructor"
              << std::endl;
  }

  TestClass(int v, const std::string& n) : value(v), name(n) {
    std::cout
        << "TestClass::TestClass(int, string) - Double parameters constructor"
        << std::endl;
  }

  int getValue() const {
    std::cout << "TestClass::getValue() - getter" << std::endl;
    return value;
  }

  const std::string& getName() const {
    std::cout << "TestClass::getName() - getter" << std::endl;
    return name;
  }

  void setValue(int v) {
    std::cout << "TestClass::setValue(int) - setter (int)" << std::endl;
    value = v;
  }

  void setName(const std::string& n) {
    std::cout << "TestClass::setName(string) - setter (string)" << std::endl;
    name = n;
  }

  static int getDefaultValue() {
    std::cout << "TestClass::getDefaultValue() - static method" << std::endl;
    return 42;
  }

  static int addValues(int a, int b) {
    std::cout << "TestClass::addValues(int, int) - static method" << std::endl;
    return a + b;
  }
};

TORCH_LIBRARY(example_library, m) {
  // Note that "float" in the schema corresponds to the C++ double type
  // and the Python float type.
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.class_<TestClass>("TestClass")
      .def(torch::init<>())
      .def(torch::init<int>())
      .def(torch::init<int, std::string>())
      .def("getValue", &TestClass::getValue)
      .def("getName", &TestClass::getName)
      .def("setValue", &TestClass::setValue)
      .def("setName", &TestClass::setName)
      .def_static("getDefaultValue", &TestClass::getDefaultValue)
      .def_static("addValues", &TestClass::addValues);
}

TEST(test_torch_library, TestLibraryOperators) {
  auto qualified_name = "example_library::mymuladd";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  function_args.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  function_args.add_arg(torch::IValue(2.0));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_tensor());
  auto result_tensor = result.get_value().to_tensor();
}

TEST(test_torch_library, TestLibraryClasses) {
  auto qualified_name = "example_library::TestClass";
  const auto& class_registry = torch::ClassRegistry::instance();
  bool has_class = class_registry.has_class(qualified_name);
  ASSERT_TRUE(has_class);
  torch::FunctionArgs constructor_args;
  constructor_args.add_arg(torch::IValue(10));
  constructor_args.add_arg(torch::IValue("example"));

  // Call constructor
  auto instance = class_registry.call_constructor_with_args(qualified_name,
                                                            constructor_args);
  ASSERT_TRUE(instance.get_value().is_custom_class());

  // Call getValue
  auto get_value_result = class_registry.call_method_with_args(
      qualified_name, "getValue", instance.get_value(), torch::FunctionArgs());
  ASSERT_TRUE(get_value_result.get_value().is_int());
  int value = get_value_result.get_value().to_int();
  ASSERT_EQ(value, 10);

  // Call setValue
  torch::FunctionArgs set_value_args;
  set_value_args.add_arg(torch::IValue(20));
  class_registry.call_method_with_args(
      qualified_name, "setValue", instance.get_value(), set_value_args);
  ASSERT_EQ(instance.get_value().to_custom_class<TestClass>()->value, 20);
  auto get_value_after_set = class_registry.call_method_with_args(
      qualified_name, "getValue", instance.get_value(), torch::FunctionArgs());
  ASSERT_EQ(get_value_after_set.get_value().to_int(), 20);

  // Call getName
  auto get_name_result = class_registry.call_method_with_args(
      qualified_name, "getName", instance.get_value(), torch::FunctionArgs());
  ASSERT_TRUE(get_name_result.get_value().is_string());
  std::string name = get_name_result.get_value().to_string();
  ASSERT_EQ(name, "example");

  // Call setName
  torch::FunctionArgs set_name_args;
  set_name_args.add_arg(torch::IValue("new_example"));
  class_registry.call_method_with_args(
      qualified_name, "setName", instance.get_value(), set_name_args);
  ASSERT_EQ(instance.get_value().to_custom_class<TestClass>()->name,
            "new_example");
  auto get_name_after_set = class_registry.call_method_with_args(
      qualified_name, "getName", instance.get_value(), torch::FunctionArgs());
  ASSERT_EQ(get_name_after_set.get_value().to_string(), "new_example");

  // Call static method getDefaultValue
  auto get_default_value_result = class_registry.call_static_method_with_args(
      qualified_name, "getDefaultValue", torch::FunctionArgs());
  ASSERT_TRUE(get_default_value_result.get_value().is_int());
  int default_value = get_default_value_result.get_value().to_int();
  ASSERT_EQ(default_value, 42);

  // Call static method addValues
  torch::FunctionArgs add_values_args;
  add_values_args.add_arg(torch::IValue(5));
  add_values_args.add_arg(torch::IValue(7));
  auto add_values_result = class_registry.call_static_method_with_args(
      qualified_name, "addValues", add_values_args);
  ASSERT_TRUE(add_values_result.get_value().is_int());
  int sum = add_values_result.get_value().to_int();
  ASSERT_EQ(sum, 12);
}

TORCH_LIBRARY_IMPL(example_library, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
}

TORCH_LIBRARY_FRAGMENT(example_library_fragment, m) {
  m.def("int_add", &generic_add<int>);
}

TORCH_LIBRARY_FRAGMENT(example_library_fragment, m) {
  m.def("string_concat", &generic_add<std::string>);
}

TEST(test_torch_library, TestFragmentOperators) {
  auto qualified_name_int_add = "example_library_fragment::int_add";
  auto* op_int_add =
      torch::OperatorRegistry::instance().find_operator(qualified_name_int_add);
  ASSERT_NE(op_int_add, nullptr);
  auto impl_it_int_add =
      op_int_add->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it_int_add, op_int_add->implementations.end());
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(3));
  function_args.add_arg(torch::IValue(4));
  auto result = impl_it_int_add->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_int());
  int sum = result.get_value().to_int();
  ASSERT_EQ(sum, 7);

  auto qualified_name_string_concat = "example_library_fragment::string_concat";
  auto* op_string_concat = torch::OperatorRegistry::instance().find_operator(
      qualified_name_string_concat);
  ASSERT_NE(op_string_concat, nullptr);
  auto impl_it_string_concat =
      op_string_concat->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it_string_concat, op_string_concat->implementations.end());
  torch::FunctionArgs string_args;
  string_args.add_arg(torch::IValue(std::string("Hello, ")));
  string_args.add_arg(torch::IValue(std::string("World!")));
  auto string_result =
      impl_it_string_concat->second.call_with_args(string_args);
  ASSERT_TRUE(string_result.get_value().is_string());
  std::string concatenated_string = string_result.get_value().to_string();
  ASSERT_EQ(concatenated_string, "Hello, World!");
}

at::Tensor cast_with_scalar_type(at::Tensor input, c10::ScalarType dtype) {
  return input.toType(dtype);
}

TORCH_LIBRARY(example_library_with_scalar_type_input, m) {
  m.def("cast_with_scalar_type", &cast_with_scalar_type);
}

TEST(test_torch_library, TestScalarTypeInput) {
  auto qualified_name =
      "example_library_with_scalar_type_input::cast_with_scalar_type";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  function_args.add_arg(torch::IValue(at::kDouble));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_tensor());
  auto result_tensor = result.get_value().to_tensor();
  ASSERT_EQ(result_tensor.dtype(), at::kDouble);
}

int fn_with_int_const(int const x) { return x + 1; }

TORCH_LIBRARY(example_library_with_int_const, m) {
  m.def("fn_with_int_const", &fn_with_int_const);
}

TEST(test_torch_library, TestIntConst) {
  auto qualified_name = "example_library_with_int_const::fn_with_int_const";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(3));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_int());
  int value = result.get_value().to_int();
  ASSERT_EQ(value, 4);
}

int fn_with_optional_input(torch::optional<int64_t> x) {
  if (x.has_value()) {
    return x.value() + 1;
  } else {
    return -1;
  }
}

TORCH_LIBRARY(example_library_with_optional_input, m) {
  m.def("fn_with_optional_input", &fn_with_optional_input);
}

TEST(test_torch_library, TestOptionalInput) {
  auto qualified_name =
      "example_library_with_optional_input::fn_with_optional_input";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  // Test with value
  torch::FunctionArgs function_args_with_value;
  function_args_with_value.add_arg(torch::IValue(int64_t(5)));
  auto result_with_value =
      impl_it->second.call_with_args(function_args_with_value);
  ASSERT_TRUE(result_with_value.get_value().is_int());
  int value_with_value = result_with_value.get_value().to_int();
  ASSERT_EQ(value_with_value, 6);

  // Test without value (nullopt)
  torch::FunctionArgs function_args_without_value;
  function_args_without_value.add_arg(torch::IValue());
  auto result_without_value =
      impl_it->second.call_with_args(function_args_without_value);
  ASSERT_TRUE(result_without_value.get_value().is_int());
  int value_without_value = result_without_value.get_value().to_int();
  ASSERT_EQ(value_without_value, -1);
}

int fn_with_arrayref_input(c10::ArrayRef<int64_t> x) {
  int sum = 0;
  for (const auto& val : x) {
    sum += val;
  }
  return sum;
}

TORCH_LIBRARY(example_library_with_arrayref_input, m) {
  m.def("fn_with_arrayref_input", &fn_with_arrayref_input);
}

TEST(test_torch_library, TestArrayRefInput) {
  auto qualified_name =
      "example_library_with_arrayref_input::fn_with_arrayref_input";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(std::vector<int64_t>({1, 2, 3, 4})));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_int());
  int value = result.get_value().to_int();
  ASSERT_EQ(value, 10);
}

int fn_with_mix_optional_arrayref_input(
    c10::optional<c10::ArrayRef<int64_t>> x) {
  if (x.has_value()) {
    int sum = 0;
    for (const auto& val : x.value()) {
      sum += val;
    }
    return sum;
  } else {
    return -1;
  }
}

TORCH_LIBRARY(example_library_with_mix_optional_arrayref_input, m) {
  m.def("fn_with_mix_optional_arrayref_input",
        &fn_with_mix_optional_arrayref_input);
}

TEST(test_torch_library, TestMixOptionalArrayRefInput) {
  auto qualified_name =
      "example_library_with_mix_optional_arrayref_input::"
      "fn_with_mix_optional_arrayref_input";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  // Test with value
  torch::FunctionArgs function_args_with_value;
  function_args_with_value.add_arg(
      torch::IValue(std::vector<int64_t>({1, 2, 3, 4})));
  auto result_with_value =
      impl_it->second.call_with_args(function_args_with_value);
  ASSERT_TRUE(result_with_value.get_value().is_int());
  int value_with_value = result_with_value.get_value().to_int();
  ASSERT_EQ(value_with_value, 10);

  // Test without value (nullopt)
  torch::FunctionArgs function_args_without_value;
  function_args_without_value.add_arg(torch::IValue());
  auto result_without_value =
      impl_it->second.call_with_args(function_args_without_value);
  ASSERT_TRUE(result_without_value.get_value().is_int());
  int value_without_value = result_without_value.get_value().to_int();
  ASSERT_EQ(value_without_value, -1);
}

void fn_with_optional_tensor_const_ref_input(
    torch::optional<at::Tensor> const& x) {}

TORCH_LIBRARY(example_library_with_optional_tensor_const_ref_input, m) {
  m.def("fn_with_optional_tensor_const_ref_input",
        &fn_with_optional_tensor_const_ref_input);
}

TEST(test_torch_library, TestOptionalTensorConstRefInput) {
  auto qualified_name =
      "example_library_with_optional_tensor_const_ref_input::"
      "fn_with_optional_tensor_const_ref_input";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  // Test with value
  torch::FunctionArgs function_args_with_value;
  function_args_with_value.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  impl_it->second.call_with_args(function_args_with_value);

  // Test without value (nullopt)
  torch::FunctionArgs function_args_without_value;
  function_args_without_value.add_arg(torch::IValue());
  impl_it->second.call_with_args(function_args_without_value);
}

// Function that returns a list of two tensors (instead of tuple)
std::vector<at::Tensor> return_tensor_list(const at::Tensor& input, int dim) {
  // Simply create two tensors of different sizes as demonstration
  auto first_part = at::ones({2}, input.options());
  auto second_part = at::ones({2}, input.options());

  return {first_part, second_part};
}

// Function that actually returns std::tuple<Tensor, Tensor>
std::tuple<at::Tensor, at::Tensor> return_tensor_tuple(const at::Tensor& input,
                                                       int dim) {
  // Create two tensors and return as tuple
  auto first_part = at::ones({2}, input.options());
  auto second_part =
      at::ones({3}, input.options());  // Different size to verify

  return std::make_tuple(first_part, second_part);
}

// Function that actually returns std::tuple<Tensor, Tensor>
std::tuple<at::Tensor, at::Tensor, at::Tensor> return_tensor_tuple_3(
    const at::Tensor& input, int dim) {
  // Create two tensors and return as tuple
  auto first_part = at::ones({2}, input.options());
  auto second_part =
      at::ones({3}, input.options());  // Different size to verify
  auto third_part = at::ones({4}, input.options());

  return std::make_tuple(first_part, second_part, third_part);
}

TORCH_LIBRARY(example_library_with_tuple_return, m) {
  m.def("split_tensor_list", &return_tensor_list);
  m.def("split_tensor_tuple", &return_tensor_tuple);
  m.def("split_tensor_tuple_3", &return_tensor_tuple_3);
}

TEST(test_torch_library, TestTupleReturn) {
  // Test vector<Tensor> return (list)
  auto qualified_name_list =
      "example_library_with_tuple_return::split_tensor_list";
  auto* op_list =
      torch::OperatorRegistry::instance().find_operator(qualified_name_list);
  ASSERT_NE(op_list, nullptr);
  auto impl_it_list = op_list->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it_list, op_list->implementations.end());

  // Create a test tensor [0, 1, 2, 3] with shape [4]
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};
  auto input_tensor = at::from_blob(data.data(), {4}, at::kFloat).clone();

  torch::FunctionArgs function_args_list;
  function_args_list.add_arg(torch::IValue(input_tensor));
  function_args_list.add_arg(torch::IValue(0));  // split along dimension 0

  auto result_list = impl_it_list->second.call_with_args(function_args_list);

  // Verify the result is a GenericList (vector of tensors)
  ASSERT_TRUE(result_list.get_value().is_list());

  auto list_val = result_list.get_value().to_list();
  ASSERT_EQ(list_val.size(), 2);

  // Check first tensor should have size [2]
  auto first_tensor_list = list_val[0].to_tensor();
  ASSERT_EQ(first_tensor_list.size(0), 2);

  // Check second tensor should have size [2]
  auto second_tensor_list = list_val[1].to_tensor();
  ASSERT_EQ(second_tensor_list.size(0), 2);

  // Test std::tuple<Tensor, Tensor> return (tuple)
  auto qualified_name_tuple =
      "example_library_with_tuple_return::split_tensor_tuple";
  auto* op_tuple =
      torch::OperatorRegistry::instance().find_operator(qualified_name_tuple);
  ASSERT_NE(op_tuple, nullptr);
  auto impl_it_tuple = op_tuple->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it_tuple, op_tuple->implementations.end());

  torch::FunctionArgs function_args_tuple;
  function_args_tuple.add_arg(torch::IValue(input_tensor));
  function_args_tuple.add_arg(torch::IValue(0));  // split along dimension 0

  auto result_tuple = impl_it_tuple->second.call_with_args(function_args_tuple);

  // Verify the result is a tuple
  ASSERT_TRUE(result_tuple.get_value().is_tuple());

  auto tuple_val = result_tuple.get_value().to_tuple();
  ASSERT_EQ(tuple_val.size(), 2);

  // Check first tensor should have size [2]
  auto first_tensor_tuple = tuple_val[0].to_tensor();
  ASSERT_EQ(first_tensor_tuple.size(0), 2);

  // Check second tensor should have size [3] (different from first)
  auto second_tensor_tuple = tuple_val[1].to_tensor();
  ASSERT_EQ(second_tensor_tuple.size(0), 3);

  // Test std::tuple<Tensor, Tensor, Tensor> return (tuple)
  auto qualified_name_tuple_3 =
      "example_library_with_tuple_return::split_tensor_tuple_3";
  auto* op_tuple_3 =
      torch::OperatorRegistry::instance().find_operator(qualified_name_tuple_3);
  ASSERT_NE(op_tuple_3, nullptr);
  auto impl_it_tuple_3 =
      op_tuple_3->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it_tuple_3, op_tuple_3->implementations.end());

  torch::FunctionArgs function_args_tuple_3;
  function_args_tuple_3.add_arg(torch::IValue(input_tensor));
  function_args_tuple_3.add_arg(torch::IValue(0));  // split along dimension 0

  auto result_tuple_3 =
      impl_it_tuple_3->second.call_with_args(function_args_tuple_3);

  // Verify the result is a tuple
  ASSERT_TRUE(result_tuple_3.get_value().is_tuple());

  auto tuple_val_3 = result_tuple_3.get_value().to_tuple();
  ASSERT_EQ(tuple_val_3.size(), 3);

  // Check first tensor should have size [2]
  auto first_tensor_tuple_3 = tuple_val_3[0].to_tensor();
  ASSERT_EQ(first_tensor_tuple_3.size(0), 2);

  // Check second tensor should have size [3] (different from first)
  auto second_tensor_tuple_3 = tuple_val_3[1].to_tensor();
  ASSERT_EQ(second_tensor_tuple_3.size(0), 3);

  // Check third tensor should have size [4] (different from first and second)
  auto third_tensor_tuple_3 = tuple_val_3[2].to_tensor();
  ASSERT_EQ(third_tensor_tuple_3.size(0), 4);
}

// Test for const reference parameters fix
void fn_with_const_ref_param(const int& x, const std::string& str) {
  // Simple function to test const reference parameter handling
}

TORCH_LIBRARY(example_library_const_ref_fix, m) {
  m.def("fn_with_const_ref_param", &fn_with_const_ref_param);
}

TEST(test_torch_library, TestConstRefParameterFix) {
  auto qualified_name =
      "example_library_const_ref_fix::fn_with_const_ref_param";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  // Test with const reference parameters
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(42));
  function_args.add_arg(torch::IValue(std::string("test")));

  // This should not throw compilation errors
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_none());  // void function returns None
}

TEST(test_torch_library, TestClassRegistryHasClass) {
  auto qualified_name = "example_library::TestClass";
  const auto& class_registry = torch::ClassRegistry::instance();
  bool has_class = class_registry.has_class(qualified_name);
  ASSERT_TRUE(has_class);
}

TEST(test_torch_library, TestClassRegistryHasNonExistentClass) {
  auto qualified_name = "example_library::NonExistentClass";
  const auto& class_registry = torch::ClassRegistry::instance();
  bool has_class = class_registry.has_class(qualified_name);
  ASSERT_FALSE(has_class);
}

TEST(test_torch_library, TestClassRegistryPrintAllClasses) {
  const auto& class_registry = torch::ClassRegistry::instance();
  class_registry.print_all_classes();
}

TEST(test_torch_library, TestOperatorRegistryHasOperator) {
  auto qualified_name = "example_library::mymuladd";
  const auto& operator_registry = torch::OperatorRegistry::instance();
  bool has_operator = operator_registry.has_operator(qualified_name);
  ASSERT_TRUE(has_operator);
}

TEST(test_torch_library, TestOperatorRegistryHasNonExistentOperator) {
  auto qualified_name = "example_library::non_existent_op";
  const auto& operator_registry = torch::OperatorRegistry::instance();
  bool has_operator = operator_registry.has_operator(qualified_name);
  ASSERT_FALSE(has_operator);
}

TEST(test_torch_library, TestOperatorRegistryPrintAllOperators) {
  const auto& operator_registry = torch::OperatorRegistry::instance();
  operator_registry.print_all_operators();
}

TEST(test_torch_library, TestLibraryPrintInfo) {
  torch::Library lib("example_library_test_print_info");
  lib.print_info();
}

TEST(test_torch_library, TestIValueNone) {
  torch::IValue ival = torch::IValue();
  ASSERT_TRUE(ival.is_none());
  ASSERT_EQ(ival.to_repr(), "None");
  ASSERT_EQ(ival.type_string(), "None");
}

TEST(test_torch_library, TestIValueBool) {
  torch::IValue ival = true;
  ASSERT_TRUE(ival.is_bool());
  ASSERT_EQ(ival.to_repr(), "true");
  ASSERT_EQ(ival.type_string(), "Bool");
}

TEST(test_torch_library, TestIValueInt) {
  torch::IValue ival = 42;
  ASSERT_TRUE(ival.is_int());
  ASSERT_EQ(ival.to_repr(), "42");
  ASSERT_EQ(ival.type_string(), "Int");
}

TEST(test_torch_library, TestIValueDouble) {
  torch::IValue ival = 3.14;
  ASSERT_TRUE(ival.is_double());
  ASSERT_TRUE(ival.to_repr().find("3.14") != std::string::npos);
  ASSERT_EQ(ival.type_string(), "Double");
}

TEST(test_torch_library, TestIValueString) {
  torch::IValue ival = std::string("hello");
  ASSERT_TRUE(ival.is_string());
  ASSERT_EQ(ival.to_repr(), "\"hello\"");
  ASSERT_EQ(ival.type_string(), "String");
}

TEST(test_torch_library, TestIValueTensor) {
  at::Tensor tensor = at::ones({2, 2}, at::kFloat);
  torch::IValue ival = tensor;
  ASSERT_TRUE(ival.is_tensor());
  ASSERT_EQ(ival.type_string(), "Tensor");
}

TEST(test_torch_library, TestIValueList) {
  std::vector<torch::IValue> vec = {1, 2, 3};
  torch::IValue ival = torch::IValue(vec);
  ASSERT_TRUE(ival.is_list());
  ASSERT_EQ(ival.to_repr(), "[1, 2, 3]");
  ASSERT_EQ(ival.type_string(), "List");
}

TEST(test_torch_library, TestIValueTuple) {
  torch::IValue ival = torch::IValue(std::make_tuple(1, true, "three"));
  ASSERT_TRUE(ival.is_tuple());
  ASSERT_EQ(ival.to_repr(), "(1, true, \"three\")");
  ASSERT_EQ(ival.type_string(), "Tuple");
}
