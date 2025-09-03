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

#include <torch/library.h>
#include "glog/logging.h"
#include "paddle/common/exception.h"

namespace torch {

// ClassRegistry
void ClassRegistry::register_class(const std::string& namespace_name,
                                   const std::string& class_name) {
  std::string qualified_name = namespace_name + "::" + class_name;
  classes_[qualified_name] =
      std::make_unique<ClassRegistration>(namespace_name, class_name);
  VLOG(3) << "Registered class: " << qualified_name;
}

void ClassRegistry::register_constructor(const std::string& qualified_name,
                                         CppFunction&& func) {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }
  it->second->constructors.push_back(
      std::make_shared<CppFunction>(std::move(func)));
  VLOG(3) << "Registered constructor for: " << qualified_name
          << " (total: " << it->second->constructors.size() << ")";
}

void ClassRegistry::register_method(const std::string& qualified_name,
                                    const std::string& method_name,
                                    CppFunction&& func) {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }
  it->second->methods[method_name] =
      std::make_shared<CppFunction>(std::move(func));
  VLOG(3) << "Registered method: " << qualified_name << "::" << method_name;
}

void ClassRegistry::register_static_method(const std::string& qualified_name,
                                           const std::string& method_name,
                                           CppFunction&& func) {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }
  it->second->static_methods[method_name] =
      std::make_shared<CppFunction>(std::move(func));
  VLOG(3) << "Registered static method: " << qualified_name
          << "::" << method_name;
}

FunctionResult ClassRegistry::call_method_with_args(
    const std::string& qualified_name,
    const std::string& method_name,
    const FunctionArgs& args) const {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }

  auto& class_reg = it->second;
  auto method_it = class_reg->methods.find(method_name);
  if (method_it == class_reg->methods.end()) {
    PADDLE_THROW(common::errors::NotFound("Method %s not found in class %s!",
                                          method_name.c_str(),
                                          qualified_name.c_str()));
  }

  try {
    VLOG(3) << "Executing " << qualified_name << "::" << method_name
            << " (instance) with " << args.size() << " args";
    auto result = method_it->second->call_with_args(args);

    if (result.has_value()) {
      VLOG(3) << "Instance method executed successfully with return value";
    } else {
      VLOG(3) << "Instance method executed successfully (void)";
    }
    return result;
  } catch (const std::exception& e) {
    VLOG(3) << "Instance method execution failed: " << e.what();
    throw;
  }
}

FunctionResult ClassRegistry::call_method_with_args(
    const std::string& qualified_name,
    const std::string& method_name,
    const IValue& instance,
    const FunctionArgs& args) const {
  FunctionArgs full_args;
  full_args.add_arg(instance);
  for (size_t i = 0; i < args.size(); ++i) {
    full_args.add_arg(args.get_value(i));
  }
  return call_method_with_args(qualified_name, method_name, full_args);
}

FunctionResult ClassRegistry::call_constructor_with_args(
    const std::string& qualified_name, const FunctionArgs& args) const {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }

  auto& class_reg = it->second;
  if (class_reg->constructors.empty()) {
    PADDLE_THROW(common::errors::NotFound(
        "No constructor registered for class %s!", qualified_name.c_str()));
  }

  VLOG(3) << "Creating instance of " << qualified_name << " with "
          << args.size() << " args";
  VLOG(3) << "Available constructors: " << class_reg->constructors.size();

  for (size_t i = 0; i < class_reg->constructors.size(); ++i) {
    try {
      VLOG(3) << "Trying constructor " << (i + 1) << "...";
      auto result = class_reg->constructors[i]->call_with_args(args);
      VLOG(3) << "Constructor " << (i + 1) << " executed successfully";
      return result;
    } catch (const std::exception& e) {
      VLOG(3) << "Constructor " << (i + 1) << " failed: " << e.what();
    }
  }

  PADDLE_THROW(common::errors::InvalidArgument(
      "No suitable constructor found for class %s!", qualified_name.c_str()));
}

FunctionResult ClassRegistry::call_static_method_with_args(
    const std::string& qualified_name,
    const std::string& method_name,
    const FunctionArgs& args) const {
  auto it = classes_.find(qualified_name);
  if (it == classes_.end()) {
    PADDLE_THROW(common::errors::NotFound("Class %s not found in registry!",
                                          qualified_name.c_str()));
  }

  auto& class_reg = it->second;
  auto method_it = class_reg->static_methods.find(method_name);
  if (method_it == class_reg->static_methods.end()) {
    PADDLE_THROW(
        common::errors::NotFound("Static method %s not found in class %s!",
                                 method_name.c_str(),
                                 qualified_name.c_str()));
  }

  try {
    VLOG(3) << "Executing " << qualified_name << "::" << method_name
            << " (static) with " << args.size() << " args";
    auto result = method_it->second->call_with_args(args);

    if (result.has_value()) {
      VLOG(3) << "Static method executed successfully with return value";
    } else {
      VLOG(3) << "Static method executed successfully (void return)";
    }
    return result;
  } catch (const std::exception& e) {
    VLOG(3) << "Error executing static method: " << e.what();
    throw;
  }
}

void ClassRegistry::print_all_classes() const {
  std::ostringstream oss;
  oss << "\n=== Registered Classes ===" << std::endl;
  for (const auto& [qualified_name, registration] : classes_) {
    oss << "Class: " << qualified_name << std::endl;

    if (!registration->constructors.empty()) {
      oss << "  Constructors: " << registration->constructors.size()
          << " available" << std::endl;
    }

    if (!registration->methods.empty()) {
      oss << "  Methods: ";
      for (const auto& [method_name, _] : registration->methods) {
        oss << method_name << " ";
      }
      oss << std::endl;
    }

    if (!registration->static_methods.empty()) {
      oss << "  Static Methods: ";
      for (const auto& [method_name, _] : registration->static_methods) {
        oss << method_name << " ";
      }
      oss << std::endl;
    }
  }
  oss << "==========================" << std::endl;
  std::cout << oss.str();
}

// OperatorRegistry
void OperatorRegistry::register_schema(const std::string& qualified_name,
                                       const std::string& schema) {
  auto& op = get_or_create_operator(qualified_name);
  op.schema = schema;
  VLOG(3) << "Registered schema: " << qualified_name << " -> " << schema;
}

void OperatorRegistry::register_implementation(
    const std::string& qualified_name, DispatchKey key, CppFunction&& func) {
  auto& op = get_or_create_operator(qualified_name);
  op.implementations[key] = std::move(func);
  VLOG(3) << "Registered implementation: " << qualified_name << " for "
          << dispatch_key_to_string(key);
}

OperatorRegistration* OperatorRegistry::find_operator(
    const std::string& qualified_name) {
  auto it = operators_.find(qualified_name);
  return (it != operators_.end()) ? &it->second : nullptr;
}

void OperatorRegistry::print_all_operators() const {
  std::stringstream oss;
  oss << "\n=== Registered Operators ===" << std::endl;
  for (const auto& [name, op] : operators_) {
    oss << "Operator: " << name << std::endl;
    if (!op.schema.empty()) {
      oss << "  Schema: " << op.schema << std::endl;
    }
    oss << "  Implementations: ";
    for (const auto& [key, impl] : op.implementations) {
      oss << dispatch_key_to_string(key) << " ";
    }
    oss << std::endl;
  }
  oss << "=========================" << std::endl;
  std::cout << oss.str();
}

// Library
Library::Library(Kind kind,
                 const std::string& ns,
                 std::optional<DispatchKey> dispatch_key,
                 const char* file,
                 uint32_t line)
    : kind_(kind),
      ns_(ns),
      dispatch_key_(dispatch_key),
      file_(file),
      line_(line) {
  std::stringstream oss;
  oss << "Created Library: kind=" << kind_to_string(kind)
      << ", namespace=" << ns;
  if (dispatch_key) {
    oss << ", dispatch_key=" << dispatch_key_to_string(*dispatch_key);
  }
  VLOG(3) << oss.str() << std::endl;
}

Library::Library(const std::string& ns)  // NOLINT
    : kind_(DEF), ns_(ns), file_(nullptr), line_(0) {
  VLOG(3) << "Created Library: namespace=" << ns << std::endl;
}

Library& Library::def(const std::string& schema) & {
  if (kind_ == IMPL) {
    VLOG(3)
        << "Warning: def() should not be called in TORCH_LIBRARY_IMPL block";
    return *this;
  }

  // Simple schema extraction: if it contains '(', extract the part before '('
  auto op_name = extract_op_name(schema);
  auto qualified_name = ns_ + "::" + op_name;

  OperatorRegistry::instance().register_schema(qualified_name, schema);
  return *this;
}

void Library::print_info() const {
  std::ostringstream oss;
  oss << "Library Info: " << kind_to_string(kind_) << ", namespace=" << ns_;
  if (dispatch_key_) {
    oss << ", dispatch_key=" << dispatch_key_to_string(*dispatch_key_);
  }
  std::cout << oss.str() << std::endl;
}

}  // namespace torch
