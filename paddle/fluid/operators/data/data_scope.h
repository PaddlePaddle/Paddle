/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

extern "C" {
#include <xxhash.h>
}

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"

// When in inference scenario, the scopes will not be written by two threads in
// a mean time, but a scope may be read by multiple threads concurrently, and
// the mutex will cause serious performance issue.
// So the mutex is disabled when `ON_INFER`.
#ifdef PADDLE_ON_INFERENCE
#define SCOPE_KIDS_READER_LOCK
#define SCOPE_KIDS_WRITER_LOCK
#define SCOPE_VARS_READER_LOCK
#define SCOPE_VARS_WRITER_LOCK
#else
#define SCOPE_KIDS_READER_LOCK AutoRDLock auto_lock(&kids_lock_);
#define SCOPE_KIDS_WRITER_LOCK AutoWRLock auto_lock(&kids_lock_);
#define SCOPE_VARS_READER_LOCK AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK AutoWRLock auto_lock(&vars_lock_);
#endif

namespace paddle {
namespace framework {

class Scope;
class Variable;

template <typename T>
using BlockingQueue = operators::reader::BlockingQueue<T>;

/**
 * @brief DataScope that manage all variables in data pipeline.
 *
 * In data pipeline, we need a queue between each OPs to buffer data
 * to support data prefetch and OP running asynchronously, DataScope
 * contains name -> Variable map as {name: BlockingQueue<Variable>}
 */
class DataScope : public Scope {
 public:

   DataScope() {}

  /// Create a sub-scope. Returns a reference other than a pointer so
  /// to prevent from manual deletion.
  /// Mark it to const because that new kid scope cannot change parent scope.
   DataScope& NewScope() const {
     DataScope* child = new DataScope(this);
     {
       SCOPE_KIDS_WRITER_LOCK
       kids_.push_back(child);
     }
     return *child;
   }

  /// Create a sub-scope for current scope but do not record it in the kids to
  /// avoid performance problems.
   std::unique_ptr<DataScope> NewTmpScope() const {
     return std::unique_ptr<DataScope>(new DataScope(this));
   }

  // void EraseVars(const std::vector<std::string>& var_names) {
  //   std::set<std::string> var_set(var_names.begin(), var_names.end());
  //   SCOPE_VARS_WRITER_LOCK
  //   for (auto it = var_queues_.begin(); it != var_queues_.end();) {
  //     if (var_set.find(it->first) != var_set.end()) {
  //       it = var_queues_.erase(it);
  //     } else {
  //       ++it;
  //     }
  //   }
  // }
  //
  // void EraseVarsExcept(const std::unordered_set<Variable*>& vars) {
  //   SCOPE_VARS_WRITER_LOCK
  //   for (auto iter = var_queues_.begin(); iter != var_queues_.end();) {
  //     if (vars.count(iter->second.get()) != 0) {
  //       ++iter;
  //     } else {
  //       var_queues_.erase(iter++);
  //     }
  //   }
  // }

  // /// Find a variable in the scope or any of its ancestors.  Returns
  // /// nullptr if cannot find.
  // /// Caller doesn't own the returned Variable.
  // Variable* FindVar(const std::string& name) const {
  //   SCOPE_VARS_READER_LOCK
  //   return FindVarInternal(name);
  // }

  // // Get a variable in the scope or any of its ancestors. Enforce
  // /// the returned Variable is not nullptr
  // Variable* GetVar(const std::string& name) const {
  //   auto* var = FindVar(name);
  //   PADDLE_ENFORCE_NOT_NULL(
  //       var, platform::errors::NotFound("Cannot find %s in scope.", name));
  //   return var;
  // }

  /// Find a variable in the current scope.
  /// Return nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindLocalVar(const std::string& name) const {
    SCOPE_VARS_READER_LOCK
    return FindVarLocally(name);
  }

  const Scope* parent() const { return parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  // const Scope* FindScope(const Variable* var) const;

  // /// Find the scope or an ancestor scope that contains the given variable name.
  // const Scope* FindScope(const std::string& name) const;

  // void DeleteScope(Scope* scope) const;

  // /// Drop all kids scopes belonged to this scope.
  // void DropKids();

  // /// Find if a scope exists in the kid scopes
  // bool HasKid(const Scope* scope) const;

  // const std::list<Scope*>& kids() const { return kids_; }

  // enumerate all the variables current contains.
  std::vector<std::string> LocalVarNames() const {
    std::vector<std::string> known_vars;
    {
      SCOPE_VARS_READER_LOCK
      known_vars.reserve(this->var_queues_.size());
      for (auto& p : var_queues_) {
        known_vars.emplace_back(p.first);
      }
    }
    return known_vars;
  }

  // // Rename variable to a new name
  // void Rename(const std::string& origin_name,
  //             const std::string& new_name) const;
  //
  // // Rename variable to a new name and return the new name
  // std::string Rename(const std::string& origin_name) const;

 protected:
  // struct KeyHasher {
  //   std::size_t operator()(const std::string& key) const {
  //     return XXH32(key.c_str(), key.size(), 1);
  //   }
  // };

  mutable std::unordered_map<std::string, std::unique_ptr<BlockingQueue<Variable>>, KeyHasher> var_queues_;

 private:
  // Call NewScope for a sub-scope.
  explicit DataScope(Scope const* parent) : parent_(parent) {}

  // Called by Var.
  Variable* VarInternal(const std::string& name) {
    auto* v = FindVarLocally(name);
    if (v != nullptr) return v;

    auto q = GetBlockingQueue(name);
    v = new Variable();
    q->Send(*v);
    VLOG(3) << "Create Variable BlockingQueue and Create a Variable in it" << name;
    return v;
  }

  Variable* FindVarInternal(const std::string& name) const {
    auto var = FindVarLocally(name);
    if (var != nullptr) {
      return var;
    }
    return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
  }

  // // Called by FindScope.
  // const Scope* FindScopeInternal(const Variable* var) const {
  //   for (auto& kv : var_queues_) {
  //     if (kv.second.get() == var) {
  //       return this;
  //     }
  //   }
  //   return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
  // }

  // // Called by FindScope.
  const Scope* FindScopeInternal(const std::string& name) const {
    if (var_queues_.find(name) != var_queues_.end()) {
      return this;
    }
    return (parent_ == nullptr) ? nullptr : parent_->FindScope(name);
  }

  // // Called by Rename.
  void RenameInternal(const std::string& origin_name,
                             const std::string& new_name) const {
    auto origin_it = var_queues_.find(origin_name);
    PADDLE_ENFORCE_NE(
        origin_it, var_queues_.end(),
        platform::errors::NotFound(
            "Original variable with name %s is not found in the scope.",
            origin_name));
    auto new_it = var_queues_.find(new_name);
    PADDLE_ENFORCE_EQ(
        new_it, var_queues_.end(),
        platform::errors::AlreadyExists(
            "The variable with name %s already exists in the scope.", new_name));
    var_queues_[new_name].reset(origin_it->second.release());
    var_queues_.erase(origin_it);
  }

  // Called by FindVarInternal and Var.
  Variable* FindVarLocally(const std::string& name) const {
    auto it = var_queues_.find(name);
    if (it != var_queues_.end()) {
      auto q =  it->second.get();
      Variable* v = nullptr;
      if (q->Size() <= 0 || !q->Receive(v)) {
          return nullptr;
      }
      return v;
    }
    return nullptr;
  }

  BlockingQueue<Variable>* GetBlockingQueue(const std::string& name) const {
    auto it = var_queues_.find(name);
    if (it != var_queues_.end()) {
      return it->second.get();
    }
    auto q = new BlockingQueue<Variable>(2);
    var_queues_.emplace(name, std::unique_ptr<BlockingQueue<Variable>>(q));
    return q;
  }

  // Scope in `kids_` are owned by this class.
  mutable std::list<Scope*> kids_;
  const Scope* parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(DataScope);

#ifndef PADDLE_ON_INFERENCE

 private:
  mutable RWLock kids_lock_;
  mutable RWLock vars_lock_;
#endif
};
}  // namespace framework
}  // namespace paddle
