/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace operators {

// cache engine repetitives
struct EngineCache {
  std::shared_ptr<ngraph::runtime::Executable> ngraph_handle;
  std::set<std::string> persistables;
  std::vector<std::string> var_in;
  std::vector<std::string> var_out;
  std::vector<size_t> var_in_updates;
  bool is_test = true;
};

template <class T, class Engine, int separator = 0>
class NgraphThreadCache {
 public:
  typedef decltype(Engine::getMutex()) mutex_type;
  typedef std::lock_guard<mutex_type> guard_type;
  typedef T& ref_type;
  enum class type_of_thread { unknown, forward, backward };

  template <class S>
  struct MetaInfo {
    std::thread::id owner_tid;   // owner of the cache, future use;
    type_of_thread worker_type;  // future use
    S real_content;
    MetaInfo()
        : owner_tid{std::this_thread::get_id()},
          worker_type{type_of_thread::unknown} {}
  };

  typedef std::unique_ptr<MetaInfo<T>> content_type;
  typedef std::list<content_type> storage_type;

 protected:
  static storage_type l;
  static mutex_type getMutex() { return Engine::getMutex(); }
  static void remove_from_list(const T* raw_ptr) {
    guard_type guard(getMutex());
    l.remove_if([raw_ptr](const content_type& sh) {
      return &(sh->real_content) == raw_ptr;
    });
  }

  template <class TRaw>
  struct TLSDescriptor {
    TRaw* raw_ptr;
    TLSDescriptor() : raw_ptr{nullptr} {}
    ~TLSDescriptor() {
      // if thread die
      NgraphThreadCache::remove_from_list(raw_ptr);

      /* TODO : Parallel executor swap */
      // FastMultiThreadCache::keep_alive_for_backward_thread(raw_ptr);
    }
  };

 public:
  NgraphThreadCache() = delete;
  NgraphThreadCache(const NgraphThreadCache& copy) = delete;

  static T& fetch() {
    thread_local TLSDescriptor<T> tls;
    if (!tls.raw_ptr) {
      using elem_type = typename content_type::element_type;
      content_type _p(new elem_type());
      if (!_p) PADDLE_THROW("Cannot alloc memory for thread-cache ");
      guard_type guard(getMutex());
      l.push_back(std::move(_p));
      tls.raw_ptr = &l.back()->real_content;
    }
    return *(tls.raw_ptr);
  }
  auto getSize() -> decltype(l.size()) {
    guard_type guard(getMutex());
    return l.size();
  }

  template <class F>
  void for_each_cache(F f) {
    guard_type guard(getMutex());
    std::for_each(l.begin(), l.end(), f);
  }
};

template <class T, class Engine, int separator>
typename NgraphThreadCache<T, Engine, separator>::storage_type
    NgraphThreadCache<T, Engine, separator>::l;

// perform graph build through bridge and execute computation
class NgraphEngine {
 public:
  explicit NgraphEngine(const framework::Scope& scope,
                        const platform::Place& place,
                        const framework::ExecutionContext& ctx);

  void Run(const framework::Scope& scope, const platform::Place& place) const;

  static bool is_training;
  static const framework::BlockDesc* p_bdesc;
  static std::vector<std::string> feed_vars, fetch_vars;

  static void FuseNgraphOps(
      const framework::BlockDesc& prog,
      std::vector<std::unique_ptr<framework::OperatorBase>>* ops);

  static std::recursive_mutex& getMutex() {
    static std::recursive_mutex mx;
    return mx;
  }

 private:
  template <class T>
  using ThCache =
      NgraphThreadCache<std::unordered_map<std::string, T>, NgraphEngine>;

  using main_engine_cache = ThCache<EngineCache>;
  using main_t_in_cache =
      ThCache<std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>;

  static framework::Variable* pre_var_ptr;

  const framework::Scope& scope_;
  const platform::Place& place_;
  std::vector<std::shared_ptr<framework::OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::set<std::string> persistables_;
  std::unordered_set<std::string> post_op_inputs_;
  bool is_test_{true};
  std::string func_cache_key_;

  // ngraph backend eg. CPU
  static std::shared_ptr<ngraph::runtime::Backend> backend_;
  // var_name of inputs
  std::vector<std::string> var_in_;
  // var_name of outputs from  fetch in order
  std::vector<std::string> var_out_;
  // non-persitable var_in
  std::vector<size_t> var_in_updates_;
  // map input vars to nodes
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_in_node_map_;
  // map each var name with a ngraph node
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_node_map_;
  // prepare info for ngraph engine need
  void Prepare(const framework::ExecutionContext& ctx);
  // get ngraph engine input and output list
  void BuildNgIO(const std::vector<framework::OpDesc*>& op_descs,
                 const std::vector<int>& interval);
  // get ngraph input and define ngraph input parameters
  void GetNgInputShape();
  // Call ngraph bridge to map ops
  void BuildNgNodes();
  // build ngraph function call
  std::shared_ptr<ngraph::Function> BuildNgFunction(
      const framework::ExecutionContext& ctx);
  // clear ngraph engine cache and t_in cache
  void ClearNgCache();
  // Check cache for ngraph function or otherwise build the function
  void GetNgFunction(const framework::ExecutionContext& ctx);
};

}  // namespace operators
}  // namespace paddle
