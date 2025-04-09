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

#pragma once
#ifdef PADDLE_WITH_DNNL
#include <memory>
#include <mutex>     // NOLINT
#include "dnnl.hpp"  // NOLINT
#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/utils/test_macros.h"

namespace phi {

using TensorNameMap = std::map<std::string, std::vector<std::string>>;

class OneDNNContextThreadLocals {
  // default onednn session id

  typedef OneDNNContextThreadLocals self;
  struct Body {
    bool said_once = false;
    size_t cur_mkldnn_session_id;
    // Current data input shape string.
    // - For fixed-shape, it's a null string in default.
    // - For dynamic-shape, it's user specific.
    std::string cur_input_shape_str;
    // the cache capacity of different input shapes for OneDNN.
    // Default 1 means fixed input shape, not dynamic shape.
    int cur_input_shape_cache_capacity;
    // Recently registered data_format. This is needed to
    // know for converting MKL-DNN Tensor to non MKL-DNN
    DataLayout cur_paddle_data_layout;
    // MKL-DNN stream used for execution of primitives (per-thread)
    dnnl::engine cur_engine;
    dnnl::stream cur_stream;
    std::string key_suffix;  // Key identifying current Executor
    bool key_attach_thread_id = true;
    void* exec_ptr_ = nullptr;

    Body();
    ~Body();
    void set_cur_mkldnn_session_id(size_t sid);
    size_t get_cur_mkldnn_session_id(void);
    void set_cur_input_shape_str(std::string input_shape_str);
    void set_cur_input_shape_cache_capacity(int input_shape_cache_capacity);
    TEST_API void set_cur_paddle_data_layout(DataLayout dl);
    DataLayout get_cur_paddle_data_layout(void);
    void log_lib_version(void);
    const dnnl::engine& get_engine(void) { return cur_engine; }
    dnnl::stream& get_stream(void) { return cur_stream; }
    void set_key_suffix(const std::string& suffix) { key_suffix = suffix; }
    const std::string& get_key_suffix(void) const { return key_suffix; }
    void disable_tid_in_key(void) { key_attach_thread_id = false; }
    bool is_tid_used_in_key(void) const { return key_attach_thread_id; }
    void set_curr_exec(void* exec_ptr) { exec_ptr_ = exec_ptr; }
    void* get_curr_exec(void) const { return exec_ptr_; }
  };
  OneDNNContextThreadLocals() = default;
  OneDNNContextThreadLocals(const OneDNNContextThreadLocals& c) = delete;

 public:
  // default onednn session id
  static constexpr size_t kMKLDNNSessionID_Default = 0;
  // onednn session id for cache clearing mode
  static constexpr size_t kMKLDNNSessionID_CacheClearing = -1;
  TEST_API static Body& fetch();
};

class OneDNNContext : public CPUContext {
 public:
  template <class T>
  using BlobPtr_t = std::shared_ptr<T>;
  template <class P1, class P2>
  using umap_value_smart_t = std::unordered_map<P1, BlobPtr_t<P2>>;
  template <class T>
  using umap_key_string_t = umap_value_smart_t<std::string, T>;

  // Following three maps are used to cache OneDNN primitives.
  // There relations are:
  // - BlobMap = Map<cur_thread_id, ShapeBlob>
  // - ShapeBlob = Map<cur_input_shape_str, KeyBlob>
  // - KeyBlob  = Map<blob_name, blob>

  using KeyBlob = umap_key_string_t<void>;
  using ShapeBlob = umap_key_string_t<KeyBlob>;
  using BlobMap = umap_value_smart_t<int, ShapeBlob>;

  // Auxiliary two-level structure (shape, executor) to easier control
  // clearing cache objects related to specific executor

  using ExecKey = void*;
  using ExecMapCacheIterPair = std::pair<BlobPtr_t<KeyBlob>, KeyBlob::iterator>;
  using ExecMap =
      std::unordered_map<ExecKey, std::vector<ExecMapCacheIterPair>>;
  using ExecShape = std::unordered_map<std::string, std::shared_ptr<ExecMap>>;

  explicit OneDNNContext(const Place& place);
  ~OneDNNContext();
  /* \brief  Get the active engine */
  const dnnl::engine& GetEngine() const { return tls().get_engine(); }

  // Remove all entries from the blob map
  TEST_API void ResetBlobMap(void* ptr);

  // Prevent next ResetBlobMap()
  void BlockNextCacheClearing();

  // Get the ShapeBlob size in cur_mkldnn_session_id.
  size_t GetShapeBlobSize() const;

  // Set data to blob (i.e. name/data pair). Create blob if not existing
  void SetBlob(const std::string& name, std::shared_ptr<void> data) const;

  // Calculate number of oneDNN objects cached
  TEST_API unsigned int GetCachedObjectsNumber(void) const;

  // Find a saved blob. Return nullptr if not found
  std::shared_ptr<void> GetBlob(const std::string& name) const;

  static auto tls() -> decltype(OneDNNContextThreadLocals::fetch()) {
    return OneDNNContextThreadLocals::fetch();
  }

  // Several methods for adapting ONEDNN-specific attributes and inputs
  bool HasDnnAttr(const std::string& attr_name) const;
  const Attribute& GetDnnAttr(const std::string& attr_name) const;
  void SetDnnAttr(const std::string& attr_name, Attribute attr);

  bool HasDnnInput(const std::string& input_name) const;
  const DenseTensor* GetDnnInput(const std::string& input_name) const;
  void SetDnnInput(const std::string& input_name, const DenseTensor* input);

  void ClearDnnAttr();

  void SetInputsName(const TensorNameMap& inputs_name);

  void SetOutputsName(const TensorNameMap& outputs_name);

  const std::vector<std::string>& GetInputsName(const std::string& input) const;

  const std::vector<std::string>& GetOutputsName(
      const std::string& output) const;

  static const char* name();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
#endif
