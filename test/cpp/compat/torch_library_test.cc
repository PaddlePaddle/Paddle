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

#include <sstream>

#include "gtest/gtest.h"
#include "test/cpp/utils/exception_test_utils.h"

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

torch::CppFunction MakeKwonlySchemaMethodForTestClass() {
  torch::CppFunction method(
      [](const torch::FunctionArgs& args) -> torch::IValue {
        if (args.has_named_args()) {
          throw std::runtime_error(
              "Schema-normalized class method should not receive named args");
        }
        if (args.size() != 3) {
          throw std::runtime_error("Expected 3 normalized arguments");
        }

        auto instance = args.get<torch::intrusive_ptr<TestClass>>(0);
        const auto idx_repr = args.get_value(1).is_none()
                                  ? std::string("none")
                                  : std::to_string(args.get<int64_t>(1));
        return torch::IValue(instance->name + "|" + idx_repr + "|" +
                             args.get<std::string>(2));
      });
  // The self type is irrelevant here; this test only exercises kwarg
  // forwarding and schema normalization on the instance-method overload.
  method.bind_schema(torch::jit::parseSchema(
      "kwonly_forwarding(Tensor self, *, int? idx=None, str mode=\"nearest\") "
      "-> str"));
  return method;
}

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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
      op_int_add->implementations.find(c10::DispatchKey::CPU);
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
      op_string_concat->implementations.find(c10::DispatchKey::CPU);
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

int schema_only_add(int a, int b) { return a + b; }

int schema_and_impl_add(int a, int b) { return a + b; }

int name_only_add(int a, int b) { return a + b; }

int overload_name_add(int a, int b) { return a + b; }

int dispatch_probe_cpu(int x) { return x + 1; }

int dispatch_probe_cuda(int x) { return x + 2; }

int impl_block_schema_and_fn(int x) { return x * 2; }

TORCH_LIBRARY(example_library_with_mdef_cases, m) {
  m.def("schema_only_add(int a, int b) -> int");
  m.def("schema_and_impl_add(int a, int b) -> int", &schema_and_impl_add);
  m.def("name_only_add", &name_only_add);
  m.def("schema_only_no_impl(int x) -> int");
  m.def("overload.name(int a, int b) -> int", &overload_name_add);
  m.def("dispatch_probe(int x) -> int");
}

TORCH_LIBRARY_IMPL(example_library_with_mdef_cases, CPU, m) {
  m.impl("schema_only_add", &schema_only_add);
  m.impl("dispatch_probe", &dispatch_probe_cpu);
}

TORCH_LIBRARY_IMPL(example_library_with_mdef_cases, CUDA, m) {
  m.impl("dispatch_probe", &dispatch_probe_cuda);
}

TORCH_LIBRARY_IMPL(example_library_mdef_impl_block, CPU, m) {
  // def() in IMPL block is explicitly ignored.
  m.def("impl_block_schema_only(int x) -> int");
  m.def("impl_block_schema_and_fn(int x) -> int", &impl_block_schema_and_fn);
}

at::Tensor add_scalar_to_float_tensor(const at::Tensor& input, double value) {
  at::Tensor in_contig = input.contiguous();
  at::Tensor output = at::empty(in_contig.sizes(), in_contig.options());
  const float* in_ptr = in_contig.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();
  for (int64_t idx = 0; idx < output.numel(); ++idx) {
    out_ptr[idx] = in_ptr[idx] + static_cast<float>(value);
  }
  return output;
}

at::Tensor mdef_schema_matrix_basic_types(const at::Tensor& x,
                                          int i,
                                          double f,
                                          bool b,
                                          const std::string& s,
                                          const std::string& d,
                                          double n,
                                          std::optional<int64_t> z) {
  const double bias = static_cast<double>(i) + f + (b ? 1.0 : 0.0) +
                      static_cast<double>(s.size()) +
                      static_cast<double>(d.size()) + n +
                      static_cast<double>(z.value_or(0));
  return add_scalar_to_float_tensor(x, bias);
}

double mdef_schema_matrix_number_aliases(double a, double b) { return a + b; }

std::tuple<at::Tensor, int64_t> mdef_schema_matrix_optional_types(
    std::optional<int64_t> i,
    std::optional<double> f,
    std::optional<bool> b,
    std::optional<std::string> s,
    std::optional<at::Tensor> t) {
  const int64_t score = i.value_or(0) + static_cast<int64_t>(f.value_or(0.0)) +
                        (b.value_or(false) ? 1 : 0) +
                        static_cast<int64_t>(s ? s->size() : 0);
  at::Tensor base = t.has_value() ? *t : at::zeros({1}, at::kFloat);
  return {add_scalar_to_float_tensor(base, static_cast<double>(score)), score};
}

std::tuple<at::Tensor, at::Tensor> mdef_schema_matrix_tuple_optional(
    std::optional<std::tuple<at::Tensor, int64_t, double, bool, std::string>>
        payload) {
  if (!payload.has_value()) {
    return {at::zeros({1}, at::kFloat), at::ones({1}, at::kFloat)};
  }
  const auto& [x, i, f, b, s] = *payload;
  const double rhs = static_cast<double>(i) + f + (b ? 1.0 : 0.0) + s.size();
  return {x, add_scalar_to_float_tensor(x, rhs)};
}

std::string mdef_schema_matrix_defaults_mix(int i,
                                            double f,
                                            bool b,
                                            const std::string& quoted,
                                            const std::string& ident) {
  return std::to_string(i) + "|" +
         std::to_string(static_cast<int64_t>(f * 10.0)) + "|" +
         (b ? "1" : "0") + "|" + quoted + "|" + ident;
}

void mdef_schema_matrix_alias_and_kwonly(const at::Tensor& x,
                                         std::optional<int64_t> idx,
                                         const std::string& mode) {
  if (idx.has_value()) {
    (void)x[idx.value()];
  }
  (void)mode;
}

TORCH_LIBRARY(example_library_mdef_schema_matrix, m) {
  m.def(
      "basic_types(Tensor x, int i, float f, bool b, str s, Device d, Scalar "
      "n, NoneType z=None) -> Tensor",
      &mdef_schema_matrix_basic_types);
  m.def("number_aliases(Scalar a, number b) -> Scalar",
        &mdef_schema_matrix_number_aliases);
  m.def(
      "optional_types(int? i=None, float? f=None, bool? b=None, str? s=None, "
      "Tensor? t=None) -> (Tensor, int)",
      &mdef_schema_matrix_optional_types);
  m.def(
      "tuple_optional((Tensor, int, float, bool, str)? payload=None) -> "
      "(Tensor, Tensor)",
      &mdef_schema_matrix_tuple_optional);
  m.def(
      "defaults_mix(int i=3, float f=-2.5, bool b=true, str quoted=\"abc\", "
      "str ident=cpu) -> str",
      &mdef_schema_matrix_defaults_mix);
  m.def(
      "alias_and_kwonly(Tensor(a!) x, *, int? idx=None, str mode=\"nearest\") "
      "-> ()",
      &mdef_schema_matrix_alias_and_kwonly);
  m.def("variadic_signature(Tensor x, ...) -> ...",
        [](const torch::FunctionArgs& args) -> torch::IValue {
          int64_t sum = 0;
          for (size_t i = 1; i < args.size(); ++i) {
            sum += args.get<int64_t>(i);
          }
          return torch::IValue(sum);
        });
}

TEST(test_torch_library, TestMDefRegistrationPathsCallResult) {
  struct CallCase {
    const char* qualified_name;
    std::vector<torch::IValue> args;
    int64_t expected;
  };

  const std::vector<CallCase> cases = {
      {"example_library_with_mdef_cases::schema_only_add",
       {torch::IValue(11), torch::IValue(31)},
       42},
      {"example_library_with_mdef_cases::schema_and_impl_add",
       {torch::IValue(19), torch::IValue(23)},
       42},
      {"example_library_with_mdef_cases::name_only_add",
       {torch::IValue(20), torch::IValue(22)},
       42},
      // Dotted overload-style names should preserve suffix before '('.
      {"example_library_with_mdef_cases::overload.name",
       {torch::IValue(40), torch::IValue(2)},
       42},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.qualified_name);

    auto* op = torch::OperatorRegistry::instance().find_operator(
        test_case.qualified_name);
    ASSERT_NE(op, nullptr);

    auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
    ASSERT_NE(impl_it, op->implementations.end());

    torch::FunctionArgs function_args;
    for (const auto& arg : test_case.args) {
      function_args.add_arg(arg);
    }

    auto result = impl_it->second.call_with_args(function_args);
    ASSERT_TRUE(result.get_value().is_int());
    EXPECT_EQ(result.get_value().to_int(), test_case.expected);
  }
}

TEST(test_torch_library, TestMDefSchemaOnlyWithoutImplHasNoImplementation) {
  auto qualified_name = "example_library_with_mdef_cases::schema_only_no_impl";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(op->implementations.empty());
}

TEST(test_torch_library, TestMDefRegistersMultipleDispatchImplementations) {
  auto qualified_name = "example_library_with_mdef_cases::dispatch_probe";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);

  auto cpu_it = op->implementations.find(torch::DispatchKey::CPU);
  auto cuda_it = op->implementations.find(torch::DispatchKey::CUDA);
  ASSERT_NE(cpu_it, op->implementations.end());
  ASSERT_NE(cuda_it, op->implementations.end());

  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(41));
  auto cpu_result = cpu_it->second.call_with_args(function_args);
  auto cuda_result = cuda_it->second.call_with_args(function_args);
  ASSERT_TRUE(cpu_result.get_value().is_int());
  ASSERT_TRUE(cuda_result.get_value().is_int());
  EXPECT_EQ(cpu_result.get_value().to_int(), 42);
  EXPECT_EQ(cuda_result.get_value().to_int(), 43);
}

TEST(test_torch_library, TestMDefInImplBlockIsNoop) {
  {
    auto qualified_name =
        "example_library_mdef_impl_block::impl_block_schema_only";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    EXPECT_EQ(op, nullptr);
  }

  {
    auto qualified_name =
        "example_library_mdef_impl_block::impl_block_schema_and_fn";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    EXPECT_EQ(op, nullptr);
  }
}

TEST(test_torch_library, TestMDefSchemaMatrixBasicTypesCallResult) {
  auto qualified_name = "example_library_mdef_schema_matrix::basic_types";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  function_args.add_arg(torch::IValue(1));
  function_args.add_arg(torch::IValue(2.5));
  function_args.add_arg(torch::IValue(true));
  function_args.add_arg(torch::IValue(std::string("ab")));
  function_args.add_arg(torch::IValue(std::string("cpu")));
  function_args.add_arg(torch::IValue(3.5));
  function_args.add_arg(torch::IValue(int64_t(4)));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_tensor());
  auto out = result.get_value().to_tensor();
  EXPECT_EQ(out.sizes(), at::IntArrayRef({2, 2}));
  EXPECT_FLOAT_EQ(out[0][0].item<float>(), 18.0f);
}

TEST(test_torch_library, TestMDefSchemaMatrixNumberAliasesCallResult) {
  auto qualified_name = "example_library_mdef_schema_matrix::number_aliases";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(19.5));
  function_args.add_arg(torch::IValue(22.5));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_double());
  EXPECT_DOUBLE_EQ(result.get_value().to_double(), 42.0);
}

TEST(test_torch_library, TestMDefSchemaMatrixOptionalAndTupleCallResult) {
  {
    auto qualified_name = "example_library_mdef_schema_matrix::optional_types";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    ASSERT_NE(op, nullptr);
    auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
    ASSERT_NE(impl_it, op->implementations.end());

    torch::FunctionArgs args_with_values;
    args_with_values.add_arg(torch::IValue(int64_t(5)));
    args_with_values.add_arg(torch::IValue(2.0));
    args_with_values.add_arg(torch::IValue(true));
    args_with_values.add_arg(torch::IValue(std::string("abc")));
    args_with_values.add_arg(torch::IValue(at::ones({1}, at::kFloat)));
    auto result = impl_it->second.call_with_args(args_with_values);
    ASSERT_TRUE(result.get_value().is_tuple());
    const auto tuple_val = result.get_value().to_tuple();
    ASSERT_EQ(tuple_val.size(), 2UL);
    EXPECT_FLOAT_EQ(tuple_val[0].to_tensor()[0].item<float>(), 12.0f);
    EXPECT_EQ(tuple_val[1].to_int(), 11);
  }

  {
    auto qualified_name = "example_library_mdef_schema_matrix::tuple_optional";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    ASSERT_NE(op, nullptr);
    auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
    ASSERT_NE(impl_it, op->implementations.end());

    torch::FunctionArgs args_with_payload;
    args_with_payload.add_arg(torch::IValue(std::make_tuple(
        at::ones({1}, at::kFloat), int64_t(2), 3.0, true, std::string("ab"))));
    auto result = impl_it->second.call_with_args(args_with_payload);
    ASSERT_TRUE(result.get_value().is_tuple());
    const auto tuple_val = result.get_value().to_tuple();
    ASSERT_EQ(tuple_val.size(), 2UL);
    EXPECT_FLOAT_EQ(tuple_val[0].to_tensor()[0].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(tuple_val[1].to_tensor()[0].item<float>(), 9.0f);
  }
}

TEST(test_torch_library,
     TestMDefSchemaMatrixDefaultsAliasAndVariadicCallResult) {
  {
    auto qualified_name = "example_library_mdef_schema_matrix::defaults_mix";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    ASSERT_NE(op, nullptr);
    auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
    ASSERT_NE(impl_it, op->implementations.end());

    torch::FunctionArgs function_args;
    function_args.add_arg(torch::IValue(3));
    function_args.add_arg(torch::IValue(-2.5));
    function_args.add_arg(torch::IValue(true));
    function_args.add_arg(torch::IValue(std::string("abc")));
    function_args.add_arg(torch::IValue(std::string("cpu")));
    auto result = impl_it->second.call_with_args(function_args);
    ASSERT_TRUE(result.get_value().is_string());
    EXPECT_EQ(result.get_value().to_string(), "3|-25|1|abc|cpu");
  }

  {
    auto qualified_name =
        "example_library_mdef_schema_matrix::variadic_signature";
    auto* op =
        torch::OperatorRegistry::instance().find_operator(qualified_name);
    ASSERT_NE(op, nullptr);
    auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
    ASSERT_NE(impl_it, op->implementations.end());

    torch::FunctionArgs function_args;
    function_args.add_arg(torch::IValue(at::zeros({1}, at::kFloat)));
    function_args.add_arg(torch::IValue(10));
    function_args.add_arg(torch::IValue(20));
    function_args.add_arg(torch::IValue(12));
    auto result = impl_it->second.call_with_args(function_args);
    ASSERT_TRUE(result.get_value().is_int());
    EXPECT_EQ(result.get_value().to_int(), 42);
  }
}

TEST(test_torch_library, TestMDefKeywordOnlyCallBehavior) {
  auto qualified_name = "example_library_mdef_schema_matrix::alias_and_kwonly";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  {
    torch::FunctionArgs args_with_optional_none;
    args_with_optional_none.add_arg(torch::IValue(at::ones({4}, at::kFloat)));
    args_with_optional_none.add_arg(torch::arg("idx") = torch::IValue());
    args_with_optional_none.add_arg(torch::arg("mode") = "nearest");
    auto result = impl_it->second.call_with_args(args_with_optional_none);
    EXPECT_TRUE(result.get_value().is_none());
  }

  {
    torch::FunctionArgs args_with_defaults;
    args_with_defaults.add_arg(torch::IValue(at::ones({4}, at::kFloat)));
    auto result = impl_it->second.call_with_args(args_with_defaults);
    EXPECT_TRUE(result.get_value().is_none());
  }

  {
    torch::FunctionArgs positional_kwonly_args;
    positional_kwonly_args.add_arg(torch::IValue(at::ones({4}, at::kFloat)));
    positional_kwonly_args.add_arg(torch::IValue(int64_t(2)));
    EXPECT_ANY_THROW(
        (void)impl_it->second.call_with_args(positional_kwonly_args));
  }
}

TEST(test_torch_library, TestFunctionArgsRejectsDuplicateKeywordArgument) {
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::arg("idx") = int64_t(1));
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { function_args.add_arg(torch::arg("idx") = int64_t(2)); },
      "Duplicate keyword argument `idx`");
}

TEST(test_torch_library, TestFunctionArgsAdditionalBranches) {
  torch::FunctionArgs args;
  EXPECT_THROW(args.add_arg(torch::arg("missing")), std::runtime_error);

  args.add_arg("cpu");
  args.add_arg(torch::IValue(int64_t(7)));
  args.add_arg(int64_t(3));
  args.add_arg(torch::arg("mode") = "nearest");
  args.add_arg(torch::arg("idx") = int64_t(2));

  ASSERT_EQ(args.size(), 3UL);
  ASSERT_EQ(args.named_size(), 2UL);
  EXPECT_TRUE(args.has_named_args());
  EXPECT_FALSE(args.empty());
  EXPECT_EQ(args.get<std::string>(0), "cpu");

  const int64_t& ref_value = args.get<const int64_t&>(1);
  const int64_t const_value = args.get<const int64_t>(2);
  EXPECT_EQ(ref_value, 7);
  EXPECT_EQ(const_value, 3);

  const auto args_text = args.to_string();
  EXPECT_NE(args_text.find("kwargs={"), std::string::npos);
  EXPECT_NE(args_text.find("mode"), std::string::npos);
  EXPECT_NE(args_text.find("idx"), std::string::npos);

  auto from_vector = torch::FunctionArgs::from_vector(
      std::vector<torch::IValue>{torch::IValue(int64_t(11))});
  EXPECT_EQ(from_vector.get<int64_t>(0), 11);
}

TEST(test_torch_library, TestFunctionArgsErrorBranches) {
  torch::FunctionArgs args;
  args.add_arg(torch::IValue(int64_t(1)));

  EXPECT_THROW((void)args.get<std::string>(0), std::runtime_error);
  EXPECT_THROW((void)args.get_value(1), std::out_of_range);
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { (void)args.to_tuple<int64_t, int64_t>(); },
      "Argument count mismatch");
}

TEST(test_torch_library, TestFunctionResultErrorBranches) {
  torch::FunctionResult empty_result;
  EXPECT_FALSE(empty_result.has_value());
  EXPECT_THROW((void)empty_result.get<int64_t>(), std::runtime_error);

  torch::FunctionResult string_result(torch::IValue(std::string("abc")));
  EXPECT_THROW((void)string_result.get<int64_t>(), std::runtime_error);
}

TEST(test_torch_library, TestCppFunctionWrapperAndUninitializedErrors) {
  torch::CppFunction uninitialized;
  EXPECT_FALSE(uninitialized.valid());
  EXPECT_THROW((void)uninitialized.call(), std::runtime_error);
  EXPECT_THROW((void)uninitialized.call(1), std::runtime_error);
  EXPECT_THROW((void)uninitialized.call_with_args(torch::FunctionArgs()),
               std::runtime_error);

  std::function<torch::IValue(const torch::FunctionArgs&)> ctor_thrower =
      [](const torch::FunctionArgs& args) -> torch::IValue {
    (void)args;
    throw std::runtime_error("boom_ctor");
  };
  torch::CppFunction ctor_wrapped(ctor_thrower);
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { (void)ctor_wrapped.call_with_args(torch::FunctionArgs()); },
      "Constructor failed: boom_ctor");

  auto throw_in_free_function = +[](int x) -> int {
    (void)x;
    throw std::runtime_error("boom_fn");
  };
  torch::CppFunction free_fn_wrapped(throw_in_free_function);
  torch::FunctionArgs single_arg;
  single_arg.add_arg(torch::IValue(int64_t(1)));
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { (void)free_fn_wrapped.call_with_args(single_arg); },
      "Function call failed: boom_fn");

  auto throwing_callable =
      [](const torch::FunctionArgs& args) -> torch::IValue {
    (void)args;
    throw std::runtime_error("boom_lambda");
  };
  torch::CppFunction lambda_wrapped(throwing_callable);
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { (void)lambda_wrapped.call_with_args(torch::FunctionArgs()); },
      "Lambda execution failed: boom_lambda");
}

TEST(test_torch_library, TestCppFunctionSchemaNormalizationErrorBranches) {
  {
    torch::CppFunction fn([](const torch::FunctionArgs& args) -> torch::IValue {
      return torch::IValue(args.get<int64_t>(0) + args.get<int64_t>(1));
    });
    fn.bind_schema(torch::jit::parseSchema("normalize(int a, int b=1) -> int"));

    torch::FunctionArgs too_many_positional;
    too_many_positional.add_arg(torch::IValue(int64_t(1)));
    too_many_positional.add_arg(torch::IValue(int64_t(2)));
    too_many_positional.add_arg(torch::IValue(int64_t(3)));
    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() { (void)fn.call_with_args(too_many_positional); },
        "Too many positional arguments");
  }

  {
    torch::CppFunction fn([](const torch::FunctionArgs& args) -> torch::IValue {
      return torch::IValue(args.get<int64_t>(0) + args.get<int64_t>(1));
    });
    fn.bind_schema(
        torch::jit::parseSchema("normalize_kw(int a, *, int b=1) -> int"));

    torch::FunctionArgs positional_kwonly;
    positional_kwonly.add_arg(torch::IValue(int64_t(1)));
    positional_kwonly.add_arg(torch::IValue(int64_t(2)));
    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() { (void)fn.call_with_args(positional_kwonly); }, "keyword-only");

    torch::FunctionArgs unknown_kw;
    unknown_kw.add_arg(torch::IValue(int64_t(1)));
    unknown_kw.add_arg(torch::arg("unknown") = int64_t(2));
    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() { (void)fn.call_with_args(unknown_kw); },
        "Unknown keyword argument `unknown`");
  }

  {
    torch::CppFunction fn([](const torch::FunctionArgs& args) -> torch::IValue {
      return torch::IValue(args.get<int64_t>(0) + args.get<int64_t>(1));
    });
    fn.bind_schema(
        torch::jit::parseSchema("normalize_dup(int a, int b) -> int"));

    torch::FunctionArgs duplicated;
    duplicated.add_arg(torch::IValue(int64_t(1)));
    duplicated.add_arg(torch::IValue(int64_t(2)));
    duplicated.add_arg(torch::arg("b") = int64_t(3));
    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() { (void)fn.call_with_args(duplicated); }, "already provided");
  }

  {
    torch::CppFunction fn([](const torch::FunctionArgs& args) -> torch::IValue {
      return torch::IValue(args.get<int64_t>(0) + args.get<int64_t>(1));
    });
    fn.bind_schema(
        torch::jit::parseSchema("normalize_missing(int a, int b) -> int"));

    torch::FunctionArgs missing_required;
    missing_required.add_arg(torch::IValue(int64_t(1)));
    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() { (void)fn.call_with_args(missing_required); },
        "Missing required argument `b`");
  }
}

TEST(test_torch_library, TestCppFunctionSchemaNormalizationVarargPassthrough) {
  torch::CppFunction fn([](const torch::FunctionArgs& args) -> torch::IValue {
    int64_t sum = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      sum += args.get<int64_t>(i);
    }
    return torch::IValue(sum);
  });
  fn.bind_schema(
      torch::jit::parseSchema("normalize_vararg(int a, ...) -> int"));

  torch::FunctionArgs inputs;
  inputs.add_arg(torch::IValue(int64_t(1)));
  inputs.add_arg(torch::IValue(int64_t(2)));
  inputs.add_arg(torch::IValue(int64_t(3)));
  auto result = fn.call_with_args(inputs);
  ASSERT_TRUE(result.get_value().is_int());
  EXPECT_EQ(result.get_value().to_int(), 6);
}

TEST(test_torch_library, TestCppFunctionArityMismatchFromFunctionTraits) {
  torch::CppFunction add_two_ints(&schema_only_add);
  torch::FunctionArgs missing_one;
  missing_one.add_arg(torch::IValue(int64_t(1)));
  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() { (void)add_two_ints.call_with_args(missing_one); },
      "Function expects 2 arguments, got 1");
}

TEST(test_torch_library, TestClassMethodArityMismatchFromFunctionTraits) {
  auto qualified_name = "example_library::TestClass";
  const auto& class_registry = torch::ClassRegistry::instance();

  torch::FunctionArgs constructor_args;
  constructor_args.add_arg(torch::IValue(10));
  constructor_args.add_arg(torch::IValue("example"));
  auto instance = class_registry.call_constructor_with_args(qualified_name,
                                                            constructor_args);

  test::utils::ExpectThrowContains<std::runtime_error>(
      [&]() {
        (void)class_registry.call_method_with_args(qualified_name,
                                                   "setValue",
                                                   instance.get_value(),
                                                   torch::FunctionArgs());
      },
      "Method expects 1 arguments");
}

TEST(test_torch_library,
     TestClassMethodKwonlyArgsForwardedThroughInstanceOverload) {
  auto qualified_name = "example_library::TestClass";
  auto method_name = "kwonlyForwarding";
  auto& class_registry = torch::ClassRegistry::instance();

  class_registry.register_method(
      qualified_name, method_name, MakeKwonlySchemaMethodForTestClass());

  torch::FunctionArgs constructor_args;
  constructor_args.add_arg(torch::IValue(10));
  constructor_args.add_arg(torch::IValue("example"));
  auto instance = class_registry.call_constructor_with_args(qualified_name,
                                                            constructor_args);

  {
    torch::FunctionArgs kwonly_args;
    kwonly_args.add_arg(torch::arg("idx") = int64_t(7));
    kwonly_args.add_arg(torch::arg("mode") = "linear");

    auto result = class_registry.call_method_with_args(
        qualified_name, method_name, instance.get_value(), kwonly_args);
    ASSERT_TRUE(result.get_value().is_string());
    EXPECT_EQ(result.get_value().to_string(), "example|7|linear");
  }

  {
    torch::FunctionArgs positional_kwonly_args;
    positional_kwonly_args.add_arg(torch::IValue(int64_t(7)));

    test::utils::ExpectThrowContains<std::runtime_error>(
        [&]() {
          (void)class_registry.call_method_with_args(qualified_name,
                                                     method_name,
                                                     instance.get_value(),
                                                     positional_kwonly_args);
        },
        "keyword-only");
  }
}

TEST(test_torch_library, TestMDefSchemaDefaultsAppliedByCallWithArgs) {
  auto qualified_name = "example_library_mdef_schema_matrix::defaults_mix";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs args_without_values;
  auto result = impl_it->second.call_with_args(args_without_values);
  ASSERT_TRUE(result.get_value().is_string());
  EXPECT_EQ(result.get_value().to_string(), "3|-25|1|abc|cpu");
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(at::ones({2, 2}, at::kFloat)));
  function_args.add_arg(torch::IValue(at::kDouble));
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_tensor());
  auto result_tensor = result.get_value().to_tensor();
  ASSERT_EQ(result_tensor.dtype(), at::kDouble);
}

TEST(test_torch_library, TestRegisterImplementationAtRuntime) {
  auto qualified_name = "runtime_example::runtime_add";
  auto& registry = torch::OperatorRegistry::instance();

  registry.register_implementation(qualified_name,
                                   c10::DispatchKey::CPU,
                                   torch::CppFunction(&generic_add<int>));

  auto* op = registry.find_operator(qualified_name);
  ASSERT_NE(op, nullptr);

  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(11));
  function_args.add_arg(torch::IValue(31));
  auto result = impl_it->second.call_with_args(function_args);

  ASSERT_TRUE(result.get_value().is_int());
  ASSERT_EQ(result.get_value().to_int(), 42);
}

TEST(test_torch_library, TestLibraryPrintInfoWithDispatchKey) {
  torch::Library library(torch::Library::IMPL,
                         "runtime_library_info",
                         std::make_optional(c10::DispatchKey::CPU),
                         __FILE__,
                         __LINE__);

  std::ostringstream captured_output;
  auto* original_buffer = std::cout.rdbuf(captured_output.rdbuf());
  library.print_info();
  std::cout.rdbuf(original_buffer);

  auto output = captured_output.str();
  ASSERT_NE(output.find("Library Info: IMPL"), std::string::npos);
  ASSERT_NE(output.find("namespace=runtime_library_info"), std::string::npos);
  ASSERT_NE(output.find("dispatch_key="), std::string::npos);
}

int fn_with_int_const(int const x) { return x + 1; }

TORCH_LIBRARY(example_library_with_int_const, m) {
  m.def("fn_with_int_const", &fn_with_int_const);
}

TEST(test_torch_library, TestIntConst) {
  auto qualified_name = "example_library_with_int_const::fn_with_int_const";
  auto* op = torch::OperatorRegistry::instance().find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it_list = op_list->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it_tuple = op_tuple->implementations.find(c10::DispatchKey::CPU);
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
      op_tuple_3->implementations.find(c10::DispatchKey::CPU);
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
  auto impl_it = op->implementations.find(c10::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  // Test with const reference parameters
  torch::FunctionArgs function_args;
  function_args.add_arg(torch::IValue(42));
  function_args.add_arg(torch::IValue(std::string("test")));

  // This should not throw compilation errors
  auto result = impl_it->second.call_with_args(function_args);
  ASSERT_TRUE(result.get_value().is_none());  // void function returns None
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

TEST(test_torch_library, TestOperatorRegistryLateSchemaBindsExistingImpl) {
  auto& operator_registry = torch::OperatorRegistry::instance();
  const std::string qualified_name =
      "example_library_registry_branch::late_schema_bind";

  operator_registry.register_implementation(
      qualified_name,
      torch::DispatchKey::CPU,
      torch::CppFunction([](const torch::FunctionArgs& args) -> torch::IValue {
        return torch::IValue(args.get<int64_t>(0) + args.get<int64_t>(1));
      }));

  auto* op = operator_registry.find_operator(qualified_name);
  ASSERT_NE(op, nullptr);
  auto impl_it = op->implementations.find(torch::DispatchKey::CPU);
  ASSERT_NE(impl_it, op->implementations.end());

  torch::FunctionArgs one_arg;
  one_arg.add_arg(torch::IValue(int64_t(5)));
  EXPECT_ANY_THROW((void)impl_it->second.call_with_args(one_arg));

  operator_registry.register_schema(qualified_name,
                                    "late_schema_bind(int x, int y=3) -> int");

  auto bound_result = impl_it->second.call_with_args(one_arg);
  ASSERT_TRUE(bound_result.get_value().is_int());
  EXPECT_EQ(bound_result.get_value().to_int(), 8);
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
