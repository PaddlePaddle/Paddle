// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/infer_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#ifdef PADDLE_WITH_XPU
#include "xpu/runtime.h"
#endif
#include "glog/logging.h"

namespace paddle {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
InferGPUContext::InferGPUContext(const phi::Place& place)
    : phi::GPUContext(place, false) {}
#endif

#ifdef PADDLE_WITH_XPU
InferXPUContext::InferXPUContext(const phi::Place& place, int context_gm_size)
    : phi::XPUContext(place) {
  if (context_gm_size >= 0) {
    x_context()->set_option("XPUAPI_DEFAULT_SIZE",
                            std::to_string(context_gm_size).c_str());
  } else {
    x_context()->set_option("XPUAPI_DEFAULT_SIZE", "");
  }
}

void* InferXPUContext::Alloc(phi::TensorBase* tensor,
                             phi::DataType dtype,
                             size_t requested_size,
                             bool pinned,
                             bool fake_alloc) const {
  size_t size = tensor->numel() * phi::SizeOf(tensor->dtype());
  if (l3_autotune_size_ > 0 && holder_map_.empty()) {
    void* data_ptr =
        DeviceContext::Alloc(tensor, dtype, requested_size, pinned, fake_alloc);
    phi::XPUL3CacheBlock* l3_block = nullptr;
    phi::Allocation* holder =
        reinterpret_cast<phi::DenseTensor*>(tensor)->Holder().get();
    if (holder_l3_blocks_.count(holder) == 0) {
      l3_block = new phi::XPUL3CacheBlock();
      holder_l3_blocks_[holder] = l3_block;
      l3_blocks_.push_back(l3_block);
    } else {
      l3_block = holder_l3_blocks_[holder];
    }
    l3_block->Record(size);
    return data_ptr;
  } else if (l3_autotune_size_ > 0 && !holder_map_.empty()) {
    phi::Allocation* holder =
        reinterpret_cast<phi::DenseTensor*>(tensor)->Holder().get();
    auto holder_iter = holder_map_.find(holder);
    if (holder_iter != holder_map_.end()) {
      auto& holder_pair = holder_iter->second;
      auto* swap_holder = holder_pair.first;
      bool& swap_holder_is_l3 = holder_pair.second;
      if (swap_holder_is_l3 && swap_holder->size() >= size) {
        swap(*holder, *swap_holder);
        swap_holder_is_l3 = false;
      } else if (!swap_holder_is_l3 && holder->size() < size) {
        swap(*holder, *swap_holder);
        swap_holder_is_l3 = true;
      }
    }
    return DeviceContext::Alloc(
        tensor, dtype, requested_size, pinned, fake_alloc);
  } else {
    return DeviceContext::Alloc(
        tensor, dtype, requested_size, pinned, fake_alloc);
  }
}

void InferXPUContext::SetXContext(xpu::Context* x_context) {
  auto* old_x_context = this->x_context();
  if (old_x_context != x_context) {
    if (l3_owned_ && l3_size_ > 0 &&
        (x_context->_l3_mgr.get_size() != l3_size_ ||
         x_context->_l3_mgr.get_ptr() != l3_ptr_)) {
      xpu_free(l3_ptr_);
    }
    old_x_context->_l3_mgr.set(nullptr, 0);
    l3_size_ = x_context->_l3_mgr.get_size();
    l3_ptr_ = x_context->_l3_mgr.get_ptr();
    l3_owned_ = false;
    phi::XPUContext::SetXContext(x_context);
  }
}

void InferXPUContext::SetL3Info(size_t l3_size,
                                void* l3_ptr,
                                size_t l3_autotune_size,
                                const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());
  if (l3_ptr == nullptr) {
    if (l3_size_ != l3_size) {
      if (l3_owned_) {
        xpu_free(l3_ptr_);
      }
      if (l3_size > 0) {
        xpu_malloc(&l3_ptr_, l3_size, XPU_MEM_L3);
        if (l3_ptr_ != nullptr) {
          VLOG(3) << "remalloc l3(" << l3_size << ") success.";
          l3_size_ = l3_size;
          l3_owned_ = true;
          l3_autotune_size_ = l3_autotune_size;
        } else {
          VLOG(3) << "malloc l3(" << l3_size << ") failed. No l3 will be used.";
          l3_size_ = 0;
          l3_owned_ = false;
          l3_autotune_size_ = 0;
        }
      }
    }
  } else {
    if (l3_owned_) {
      xpu_free(l3_ptr_);
    }
    l3_ptr_ = l3_ptr;
    l3_size_ = l3_size;
    l3_autotune_size_ = l3_autotune_size;
  }
  if (l3_autotune_size_ == 0) {
    x_context()->_l3_mgr.set(l3_ptr_, l3_size_);
  }
}

void InferXPUContext::SetConvAutotuneInfo(std::string conv_autotune_file,
                                          int conv_autotune_level,
                                          bool conv_autotune_file_writeback,
                                          const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());

  VLOG(5) << "XPU conv autotune level:" << conv_autotune_level;
  VLOG(5) << "XPU conv autotune file:" << conv_autotune_file;
  VLOG(5) << "XPU conv autotune file writeback:"
          << conv_autotune_file_writeback;

  if (!conv_autotune_file.empty()) {
    int ret;
    ret = x_context()->set_option("XPU_CONV_AUTOTUNE_FILE",
                                  conv_autotune_file.c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::Unavailable("Failed to set XPU conv autotune file %s.",
                                    conv_autotune_file));
  }
  if (conv_autotune_level > 0) {
    int ret;
    ret = x_context()->set_option(
        "XPU_CONV_AUTOTUNE", (std::to_string(conv_autotune_level)).c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::Unavailable("Failed to set XPU conv autotune  %d.",
                                    conv_autotune_level));
  }
  if (conv_autotune_file_writeback) {
    int ret;
    ret = x_context()->set_option(
        "XPU_AUTOTUNE_WRITEBACK",
        (std::to_string(conv_autotune_file_writeback)).c_str());
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::Unavailable(
                          "Failed to set XPU conv autotune writeback %d.",
                          conv_autotune_file_writeback));
  }
}
void InferXPUContext::SetContextOption(const char* name, const char* value) {
  phi::backends::xpu::XPUDeviceGuard guard(GetPlace().GetDeviceId());
  VLOG(5) << "XPU Set Option name:" << name << " value:" << value;
  int ret;
  ret = x_context()->set_option(name, value);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      common::errors::Unavailable("Failed to set XPU option %s.", name));
}

void InferXPUContext::SetFcAutotuneInfo(std::string fc_autotune_file,
                                        int fc_autotune_level,
                                        bool fc_autotune_file_writeback,
                                        const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());

  VLOG(5) << "XPU fc autotune level:" << fc_autotune_level;
  VLOG(5) << "XPU fc autotune file:" << fc_autotune_file;
  VLOG(5) << "XPU fc autotune file writeback:" << fc_autotune_file_writeback;

  if (!fc_autotune_file.empty()) {
    int ret;
    ret = x_context()->set_option("XPU_FC_AUTOTUNE_FILE",
                                  fc_autotune_file.c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::Unavailable("Failed to set XPU fc autotune file %s.",
                                    fc_autotune_file));
  }
  if (fc_autotune_level > 0) {
    int ret;
    ret = x_context()->set_option("XPU_FC_AUTOTUNE",
                                  (std::to_string(fc_autotune_level)).c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        common::errors::Unavailable("Failed to set XPU fc autotune  %d.",
                                    fc_autotune_level));
  }
  if (fc_autotune_file_writeback) {
    int ret;
    ret = x_context()->set_option(
        "XPU_FC_AUTOTUNE_WRITEBACK",
        (std::to_string(fc_autotune_file_writeback)).c_str());
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      common::errors::Unavailable(
                          "Failed to set XPU fc autotune writeback %d.",
                          fc_autotune_file_writeback));
  }
}

void InferXPUContext::L3CacheAutotune() {
  if (l3_autotune_size_ == 0) return;
  if (holder_map_.empty()) {
    bool ret = l3_plan_.RunAutotune(l3_blocks_, l3_size_);
    if (!ret) {
      return;
    }
    auto* plan = l3_plan_.plan();
    int8_t* cur_l3_ptr = reinterpret_cast<int8_t*>(l3_ptr_);
    for (size_t i = 0; i < l3_blocks_.size(); i++) {
      size_t block_size = plan->at(i);
      if (block_size > 0) {
        l3_blocks_[i]->Set(cur_l3_ptr, block_size);
        cur_l3_ptr += block_size;
      }
    }
    x_context()->_l3_mgr.set(
        reinterpret_cast<int8_t*>(l3_ptr_) + l3_size_ - plan->back(),
        plan->back());

    for (auto holder_l3_block : holder_l3_blocks_) {
      auto* l3_block = holder_l3_block.second;
      if (l3_block->size() > 0) {
        auto* holder = holder_l3_block.first;
        auto place = holder->place();
        phi::Allocation* l3_holder =
            new phi::Allocation(l3_block->data(), l3_block->size(), place);
        holder_map_[holder] = std::make_pair(l3_holder, true);

        if (output_holder_set_.find(holder) != output_holder_set_.end()) {
          VLOG(4) << "Insert output tensor's l3 holder:" << l3_holder->ptr();
          SetOutHolder(l3_holder);
        }
      }
    }
  } else {
    for (auto& holders : holder_map_) {
      auto* holder = holders.first;
      auto& holder_pair = holders.second;
      if (!holder_pair.second &&
          output_holder_set_.find(holder) == output_holder_set_.end()) {
        swap(*holder, *(holder_pair.first));
        holder_pair.second = true;
      }
    }
  }
}

void InferXPUContext::SetOutHolder(phi::Allocation* holder) {
  output_holder_set_.insert(holder);
}
#endif

}  // namespace paddle
