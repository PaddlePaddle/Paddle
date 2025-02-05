//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_context.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/flat_hash_map.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/expect.h"

#include "glog/logging.h"

namespace phi {

OneDNNContextThreadLocals::Body::Body()
    : cur_engine(dnnl::engine::kind::cpu, 0), cur_stream(cur_engine) {
  cur_mkldnn_session_id = kMKLDNNSessionID_Default;
  cur_input_shape_str = "";
  cur_input_shape_cache_capacity = 1;
  cur_paddle_data_layout = DataLayout::kNCHW;
}

// When Thread finish we clear oneDNN cache
// This is needed when we have one executor used by many threads
// e.g. test_analyzer_detect. Thread ID is not part of caching key
// (for naive executor) so we need to clear cache when one thread finish
// and other is to start inference
// TODO(jczaja): Ideally it would be good to clear only part of cache
// related to thread that is to be terminated
OneDNNContextThreadLocals::Body::~Body() {  // NOLINT
  auto cpu_place = phi::CPUPlace();
  // TODO(YuanRisheng): we need remove the dependency on fluid device context
  // here
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  OneDNNContext* dev_ctx = static_cast<OneDNNContext*>(pool.Get(cpu_place));
  dev_ctx->ResetBlobMap(exec_ptr_);
}

void OneDNNContextThreadLocals::Body::set_cur_mkldnn_session_id(size_t sid) {
  cur_mkldnn_session_id = sid;
}
size_t OneDNNContextThreadLocals::Body::get_cur_mkldnn_session_id() {
  return cur_mkldnn_session_id;
}

void OneDNNContextThreadLocals::Body::set_cur_input_shape_str(
    std::string input_shape_str) {
  cur_input_shape_str = input_shape_str;
}
void OneDNNContextThreadLocals::Body::set_cur_input_shape_cache_capacity(
    int input_shape_cache_capacity) {
  cur_input_shape_cache_capacity = input_shape_cache_capacity;
}

void OneDNNContextThreadLocals::Body::set_cur_paddle_data_layout(
    DataLayout dl) {
  cur_paddle_data_layout = dl;
}

DataLayout OneDNNContextThreadLocals::Body::get_cur_paddle_data_layout() {
  return cur_paddle_data_layout;
}

void OneDNNContextThreadLocals::Body::log_lib_version() {
  if (!said_once) {
    said_once = true;
    auto dv = dnnl::version();
    LOG(INFO) << "oneDNN v" << dv->major << "." << dv->minor << "."
              << dv->patch;
  }
}

OneDNNContextThreadLocals::Body& OneDNNContextThreadLocals::fetch() {
  thread_local Body b;
  return b;
}

struct OneDNNContext::Impl {
  Impl() : p_blobmap_() {
    p_blobmap_.reset(new BlobMap());
    p_exec_items_.reset(new ExecShape());
    p_mutex_.reset(new std::mutex());
  }

  ~Impl() = default;

  void ResetBlobMap(void* ptr) {
    VLOG(4) << OneDNNContext::tls().get_curr_exec() << " " << ptr;
    std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
    if (block_next_cache_clearing_ == 0) {
      VLOG(3) << "Clearing DNNL cache.";
      // If no specific executor pointer then clear
      // everything. For executor pointer then clear only
      // objects allocated when using given executor
      if (ptr == nullptr) {
        p_blobmap_->clear();
      } else {
        // Iterate through all shapes and release
        // for each shape and active executor all entries
        // of this executor
        for (auto& s : *p_exec_items_) {
          for (auto& v : (*s.second)[ptr]) {
            (v.first)->erase(v.second);
          }
          s.second->erase(ptr);
        }
      }
      // Reset paddle layout to NCHW
      VLOG(3) << "Resetting Paddle data layout to NCHW.";
      OneDNNContext::tls().set_cur_paddle_data_layout(DataLayout::kNCHW);
    } else {
      --block_next_cache_clearing_;
      VLOG(3) << "Prevented Clearing DNNL cache. Updated "
                 "block_next_cache_clearing_ : "
              << block_next_cache_clearing_;
      PADDLE_ENFORCE_GE(block_next_cache_clearing_,
                        0,
                        common::errors::InvalidArgument(
                            "Cache clearing mark should be non-negative "
                            ". But received %d.",
                            block_next_cache_clearing_));
    }
  }

  // Register object to currently used executor's map
  void LinkEntryWithExecutor(BlobPtr_t<KeyBlob> pblob,
                             KeyBlob::iterator it) const {
    // Take current input shape from TLS
    // Take current executor address from TLS
    // and for this executor's items add the one defined with arguments
    auto key_it =
        p_exec_items_
            ->insert(std::make_pair(OneDNNContext::tls().cur_input_shape_str,
                                    std::make_shared<ExecMap>()))
            .first;
    (*key_it->second)[OneDNNContext::tls().get_curr_exec()].push_back(
        std::make_pair(pblob, it));

    VLOG(3) << "LinkEntryWithExecutor, shapes: " << p_exec_items_->size()
            << " curr exec size: "
            << (*key_it->second)[OneDNNContext::tls().get_curr_exec()].size()
            << "\n";
  }

  void RemoveShapeEntriesWithExecutor() const {
    p_exec_items_->erase(p_exec_items_->begin());
  }

  void BlockNextCacheClearing() {
    std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
    ++block_next_cache_clearing_;
    VLOG(3) << "Next DNNL cache clearing has been blocked. Updated "
               "block_next_cache_clearing_ : "
            << block_next_cache_clearing_;
  }

  size_t GetShapeBlobSize() const {
    std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);
    BlobMap* pMap = p_blobmap_.get();
    auto map_it =
        pMap->find(OneDNNContext::tls().cur_mkldnn_session_id);  // NOLINT
    if (map_it == pMap->end()) {
      PADDLE_THROW(common::errors::NotFound(
          "OneDNNContext don't find cur_mkldnn_session_id: %d.",
          OneDNNContext::tls().cur_mkldnn_session_id));
    }
    return map_it->second->size();
  }

  void SetBlob(const std::string& name, BlobPtr_t<void> data) const {
    BlobMap* pMap = p_blobmap_.get();
    BlobPtr_t<ShapeBlob> sBlob = nullptr;
    BlobPtr_t<KeyBlob> pBlob = nullptr;

    int sid = OneDNNContext::tls().get_cur_mkldnn_session_id();  // NOLINT

    std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);

    // Find ShapeBlob for current onednn session id.
    auto map_it = pMap->find(sid);

    if (map_it == pMap->end()) {
      // 1st time to set blob in current thread
      sBlob = std::make_shared<ShapeBlob>();
      (*pMap)[sid] = sBlob;
      VLOG(2) << "SetBlob: sid=" << sid << ", add new sid\n";
    } else {
      sBlob = map_it->second;
    }

    // Find KeyBlob for current input shape
    auto key_it = sBlob->find(OneDNNContext::tls().cur_input_shape_str);

    if (key_it == sBlob->end()) {
      // In cache clearing mode, cur_input_shape_cache_capacity defines
      // max pblob capacity
      if ((static_cast<size_t>(sid) ==
           OneDNNContextThreadLocals::kMKLDNNSessionID_CacheClearing) &&
          !sBlob->empty() &&
          (sBlob->size() >=
           static_cast<size_t>(
               OneDNNContext::tls().cur_input_shape_cache_capacity))) {
        VLOG(2) << "sid=" << sid
                << ", remove all blobs of shape: " << sBlob->begin()->first;
        sBlob->erase(sBlob->begin()->first);
        RemoveShapeEntriesWithExecutor();
      }
      pBlob = std::make_shared<KeyBlob>();
      (*sBlob)[OneDNNContext::tls().cur_input_shape_str] = pBlob;
    } else {
      pBlob = key_it->second;
    }

    // Find Blob via name
    auto blob_it = pBlob->find(name);
    if (blob_it == pBlob->end()) {
      auto el =
          pBlob->insert(std::make_pair(name, data));  //  (*pBlob)[name] = data;
      // Register new element in per executor map
      // to have easily erased when executor terminated
      LinkEntryWithExecutor(pBlob, el.first);
    } else {
      blob_it->second = data;  // set data to existing blob
    }
    VLOG(2) << "SetBlob: sid=" << sid << ", add blob=" << name << "\n";
    // lock will be automatically released when out of scope
    return;
  }

  unsigned int GetCachedObjectsNumber() const {
    unsigned int num_entries = 0;
    for (auto const& l3 : *p_blobmap_) {
      for (auto const& l2 : *(l3.second)) {
        num_entries += (l2.second)->size();
      }
    }
    return num_entries;
  }

  OneDNNContext::BlobPtr_t<void> GetBlob(const std::string& name) const {
    BlobMap* pMap = p_blobmap_.get();
    BlobPtr_t<ShapeBlob> sBlob = nullptr;
    BlobPtr_t<KeyBlob> pBlob = nullptr;

    int sid = OneDNNContext::tls().get_cur_mkldnn_session_id();  // NOLINT

    std::lock_guard<decltype(*p_mutex_)> lock(*p_mutex_);

    // Find ShapeBlob for current onednn session id firstly
    auto map_it = pMap->find(sid);
    // (jczaja): After first iteration of model's execution we
    // should have all elements cached (mostly) so failures are unlikely (less
    // likely for dynamic shapes)
    if (unlikely(map_it == pMap->end())) {
      VLOG(2) << "GetBlob: sid=" << sid << ", miss sid\n";
      return nullptr;
    }
    sBlob = map_it->second;

    // Find KeyBlob for current input shape secondly
    auto sBlob_it = sBlob->find(OneDNNContext::tls().cur_input_shape_str);
    if (unlikely(sBlob_it == sBlob->end())) {
      VLOG(2) << "GetBlob: sid=" << OneDNNContext::tls().cur_input_shape_str
              << ", miss input_shape_str\n";
      return nullptr;
    }
    pBlob = sBlob_it->second;

    // Find Blob via name
    auto key_it = pBlob->find(name);

    if (unlikely(key_it == pBlob->end())) {
      VLOG(2) << "GetBlob sid=" << sid << ", miss blob=" << name << "\n";
      return nullptr;
    }

    VLOG(2) << "GetBlob sid=" << sid << ", get blob=" << name << "\n";
    // lock will be automatically released when out of scope
    return key_it->second;
  }

  bool HasDnnAttr(const std::string& attr_name) const {
    return dnn_attrs_.count(attr_name) != 0UL;
  }
  const Attribute& GetDnnAttr(const std::string& attr_name) const {
    auto iter = dnn_attrs_.find(attr_name);
    PADDLE_ENFORCE_NE(iter,
                      dnn_attrs_.end(),
                      common::errors::NotFound(
                          "Attribute `%s` is not found in OneDNNContext."));
    return iter->second;
  }

  void SetDnnAttr(const std::string& attr_name, Attribute attr) {
    dnn_attrs_[attr_name] = attr;
  }

  void ClearDnnAttr() { dnn_attrs_.clear(); }

  bool HasDnnInput(const std::string& input_name) const {
    return dnn_inputs_.count(input_name) != 0UL;
  }

  const DenseTensor* GetDnnInput(const std::string& input_name) const {
    auto iter = dnn_inputs_.find(input_name);
    PADDLE_ENFORCE_NE(
        iter,
        dnn_inputs_.end(),
        common::errors::NotFound(
            "Input DenseTensor `%s` is not found in OneDNNContext."));
    return iter->second;
  }

  void SetDnnInput(const std::string& input_name, const DenseTensor* input) {
    dnn_inputs_[input_name] = input;
  }

  void SetInputsName(const TensorNameMap& inputs_name) {
    inputs_name_ = inputs_name;
  }

  void SetOutputsName(const TensorNameMap& outputs_name) {
    outputs_name_ = outputs_name;
  }

  const std::vector<std::string>& GetInputsName(
      const std::string& input) const {
    auto it = inputs_name_.find(input);
    PADDLE_ENFORCE_NE(it,
                      inputs_name_.end(),
                      common::errors::NotFound(
                          "OneDnnContext does not have the input %s.", input));
    return it->second;
  }

  const std::vector<std::string>& GetOutputsName(
      const std::string& output) const {
    auto it = outputs_name_.find(output);
    PADDLE_ENFORCE_NE(
        it,
        outputs_name_.end(),
        common::errors::NotFound("OneDnnContext does not have the output %s.",
                                 output));
    return it->second;
  }

  std::shared_ptr<BlobMap> p_blobmap_;
  // Map key is pointer of executor and value is a data(iterator in map) needed
  // to erase
  std::shared_ptr<ExecShape> p_exec_items_;
  std::shared_ptr<std::mutex> p_mutex_;
  // 0 - clearing is allowed. x > 0 do not clear.
  unsigned int block_next_cache_clearing_ = 0;

  // Holds some attributes only used by the onednn kernel calculation
  // Since original onednn op kernel directly adds the operations that require
  // fusion to the native kernel operations, and uses the attribute `fuse_xxx`
  // to control, for onednn, there will be some attributes that seem to be
  // independent of the device are also saved here.
  // Here, the operation of fusion needs to be implemented separately as
  // a fusion op and kernel, instead of patching it to a basic operation.
  // Because DeviceContext is a global singleton, you need to ensure thread
  // safety, use the thread_local variable
  static thread_local AttributeMap dnn_attrs_;
  // For onednn, in addition to extra attrs, there are also extra inputs,
  // but the number is small. Hope that the implementation can be optimized
  // to remove this member in the future.
  static thread_local paddle::flat_hash_map<std::string, const DenseTensor*>
      dnn_inputs_;

  // Onednn need get input and output's name in current Kernel for generating
  // unique_key.
  static thread_local TensorNameMap inputs_name_;
  static thread_local TensorNameMap outputs_name_;
};

thread_local AttributeMap OneDNNContext::Impl::dnn_attrs_ = {};
thread_local paddle::flat_hash_map<std::string, const DenseTensor*>
    OneDNNContext::Impl::dnn_inputs_ = {};
thread_local TensorNameMap OneDNNContext::Impl::inputs_name_ = {};
thread_local TensorNameMap OneDNNContext::Impl::outputs_name_ = {};

OneDNNContext::OneDNNContext(const Place& place)
    : CPUContext(place), impl_(std::make_unique<Impl>()) {}

OneDNNContext::~OneDNNContext() = default;

void OneDNNContext::ResetBlobMap(void* ptr) { impl_->ResetBlobMap(ptr); }

void OneDNNContext::BlockNextCacheClearing() {
  impl_->BlockNextCacheClearing();
}

size_t OneDNNContext::GetShapeBlobSize() const {
  return impl_->GetShapeBlobSize();
}

void OneDNNContext::SetBlob(const std::string& name,
                            BlobPtr_t<void> data) const {
  impl_->SetBlob(name, data);
}

unsigned int OneDNNContext::GetCachedObjectsNumber() const {
  return impl_->GetCachedObjectsNumber();
}

OneDNNContext::BlobPtr_t<void> OneDNNContext::GetBlob(
    const std::string& name) const {
  return impl_->GetBlob(name);
}

bool OneDNNContext::HasDnnAttr(const std::string& attr_name) const {
  return impl_->HasDnnAttr(attr_name);
}

const Attribute& OneDNNContext::GetDnnAttr(const std::string& attr_name) const {
  return impl_->GetDnnAttr(attr_name);
}

void OneDNNContext::SetDnnAttr(const std::string& attr_name, Attribute attr) {
  return impl_->SetDnnAttr(attr_name, std::move(attr));
}

void OneDNNContext::ClearDnnAttr() { return impl_->ClearDnnAttr(); }

bool OneDNNContext::HasDnnInput(const std::string& input_name) const {
  return impl_->HasDnnInput(input_name);
}

const DenseTensor* OneDNNContext::GetDnnInput(
    const std::string& input_name) const {
  return impl_->GetDnnInput(input_name);
}

void OneDNNContext::SetDnnInput(const std::string& input_name,
                                const DenseTensor* input) {
  return impl_->SetDnnInput(input_name, input);
}

void OneDNNContext::SetInputsName(const TensorNameMap& inputs_name) {
  impl_->SetInputsName(inputs_name);
}

void OneDNNContext::SetOutputsName(const TensorNameMap& outputs_name) {
  impl_->SetOutputsName(outputs_name);
}

const std::vector<std::string>& OneDNNContext::GetInputsName(
    const std::string& input) const {
  return impl_->GetInputsName(input);
}

const std::vector<std::string>& OneDNNContext::GetOutputsName(
    const std::string& output) const {
  return impl_->GetOutputsName(output);
}

const char* OneDNNContext::name() { return "OneDNNContext"; }

}  // namespace phi
#endif
