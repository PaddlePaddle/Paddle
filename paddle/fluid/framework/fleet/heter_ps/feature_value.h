/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_HETERPS

#include <iostream>
#include <sstream>
#include <unordered_map>
#include "paddle/fluid/distributed/ps/table/ctr_dymf_accessor.h"


namespace paddle {
namespace framework {
#define MF_DIM 8

typedef uint64_t FeatureKey;

class FeatureValueAccessor {
 public:
  __host__ __device__  FeatureValueAccessor() {}
  __host__ __device__ ~FeatureValueAccessor() {}

  __host__ __device__ virtual int Configure(std::unordered_map<std::string, float> config) {
    _config = config;
    Initialize();
    return 0;
  }
  __host__ __device__  virtual int Initialize() = 0;

 protected:
  std::unordered_map<std::string, float> _config;
};

// adagrad: embed_sgd_dim=1, embedx_sgd_dim=1,embedx_dim=n
// adam std:  embed_sgd_dim=4, embedx_sgd_dim=n*2+2,embedx_dim=n
// adam shared:  embed_sgd_dim=4, embedx_sgd_dim=4,embedx_dim=n
class CommonFeatureValueAccessor : public FeatureValueAccessor {
 public:
  struct CommonFeatureValue {
    /*
      uint64_t cpu_ptr;
      float delta_score;
      float show;
      float click;
      float embed_w;
      std::vector<float> embed_g2sum;
      float slot;
      float mf_dim
      float mf_size
      std::vector<float> embedx_g2sum;
      std::vector<float> embedx_w;
       */

    __host__ __device__ int Dim() { return 9 + embed_sgd_dim + embedx_sgd_dim + embedx_dim; } // has cpu_ptr(2)
    __host__ __device__ int DimSize(size_t dim, int embedx_dim) { return sizeof(float); }
    __host__ __device__ int Size() { return Dim() * sizeof(float); } // cpu_ptr:uint64=2float
    __host__ __device__ int EmbedDim() { return embed_sgd_dim;}
    __host__ __device__ int EmbedXDim() { return embedx_sgd_dim;}
    __host__ __device__ int EmbedWDim() { return embedx_dim;}
    __host__ __device__ int CpuPtrIndex() {return 0; } // cpuprt uint64
    __host__ __device__ int DeltaScoreIndex() { return CpuPtrIndex() + 2; } 
    __host__ __device__ int ShowIndex() { return DeltaScoreIndex() + 1; }
    __host__ __device__ int ClickIndex() { return ShowIndex() + 1; }
    __host__ __device__ int EmbedWIndex() { return ClickIndex() + 1; }
    __host__ __device__ int EmbedG2SumIndex() { return EmbedWIndex() + 1; }
    __host__ __device__ int SlotIndex() { return EmbedG2SumIndex() + embed_sgd_dim; }
    __host__ __device__ int MfDimIndex() { return SlotIndex() + 1; }
    __host__ __device__ int MfSizeIndex() { return MfDimIndex() + 1; } // actual mf size (ex. 0)
    __host__ __device__ int EmbedxG2SumIndex() { return MfSizeIndex() + 1; }
    __host__ __device__ int EmbedxWIndex() { return EmbedxG2SumIndex() + embedx_sgd_dim; }
    

    // 根据mf_dim计算的总长度
    __host__ __device__ int Dim(int& mf_dim) {
      int tmp_embedx_sgd_dim = 1;
      if (optimizer_type_ == 3) {//adam
        tmp_embedx_sgd_dim = mf_dim * 2 + 2;
      } else if (optimizer_type_ == 4) { //shared_adam
        tmp_embedx_sgd_dim = 4;
      }
      return 9 + embed_sgd_dim + tmp_embedx_sgd_dim + mf_dim;
    }

    // 根据mf_dim 计算的总byte数
    __host__ __device__ int Size(int& mf_dim) {
      return Dim(mf_dim) * sizeof(float); // cpu_ptr:2float
    }

    // 根据mf_dim 计算的 mf_size byte数
    __host__ __device__ int MFSize(int& mf_dim) {
      int tmp_embedx_sgd_dim = 1;
      if (optimizer_type_ == 3) { //adam
        tmp_embedx_sgd_dim = mf_dim * 2 + 2;
      } else if (optimizer_type_ == 4) { //shared_adam
        tmp_embedx_sgd_dim = 4;
      }
      return (tmp_embedx_sgd_dim + mf_dim) * sizeof(float);
    }

    __host__ __device__ int EmbedxG2SumOffsetIndex() { return 0; }
    __host__ __device__ int EmbedxWOffsetIndex(float* val) {
      // has mf 
      int tmp_embedx_sgd_dim = 1;
      if (int(MfSize(val)) > 0) {
        if (optimizer_type_ == 3) {//adam
          tmp_embedx_sgd_dim = int(MfDim(val)) * 2 + 2;
        } else if (optimizer_type_ == 4) { //shared_adam
          tmp_embedx_sgd_dim = 4;
        }
        return EmbedxG2SumIndex() + tmp_embedx_sgd_dim; 
      } else {
        // no mf
        return 0;
      }
    }


    __host__ __device__ uint64_t CpuPtr(float* val) {return *(reinterpret_cast<uint64_t*>(val)); }
    __host__ __device__ float& DeltaScore(float* val) { return val[DeltaScoreIndex()]; }
    __host__ __device__ float& Show(float* val) { return val[ShowIndex()]; }
    __host__ __device__ float& Click(float* val) { return val[ClickIndex()]; }
    __host__ __device__ float& Slot(float* val) { return val[SlotIndex()]; }
    __host__ __device__ float& MfDim(float* val) { return val[MfDimIndex()]; }
    __host__ __device__ float& MfSize(float* val) { return val[MfSizeIndex()]; }
    __host__ __device__ float& EmbedW(float* val) { return val[EmbedWIndex()]; }
    __host__ __device__ float& EmbedG2Sum(float* val) { return val[EmbedG2SumIndex()]; }
    __host__ __device__ float& EmbedxG2Sum(float* val) { return val[EmbedxG2SumIndex()]; }
    __host__ __device__ float& EmbedxW(float* val) { return val[EmbedxWIndex()]; }

    int embed_sgd_dim;
    int embedx_dim;
    int embedx_sgd_dim;
    int optimizer_type_;
  };

  struct CommonPushValue {
    /*
       float slot;
       float show;
       float click;
       float mf_dim;
       float embed_g;
       std::vector<float> embedx_g;
       */

    __host__ __device__ int Dim(int embedx_dim) { return 5 + embedx_dim; }

    __host__ __device__ int DimSize(int dim, int embedx_dim) { return sizeof(float); }
    __host__ __device__ int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    __host__ __device__ int SlotIndex() { return 0; }
    __host__ __device__ int ShowIndex() { return CommonPushValue::SlotIndex() + 1; }
    __host__ __device__ int ClickIndex() { return CommonPushValue::ShowIndex() + 1; }
    __host__ __device__ int MfDimIndex() { return CommonPushValue::ClickIndex() + 1; }
    __host__ __device__ int EmbedGIndex() { return CommonPushValue::MfDimIndex() + 1; }
    __host__ __device__ int EmbedxGIndex() { return CommonPushValue::EmbedGIndex() + 1; }
    __host__ __device__ float& Slot(float* val) {
      return val[CommonPushValue::SlotIndex()];
    }
    __host__ __device__ float& Show(float* val) {
      return val[CommonPushValue::ShowIndex()];
    }
    __host__ __device__ float& Click(float* val) {
      return val[CommonPushValue::ClickIndex()];
    }
    __host__ __device__ float& MfDim(float* val) {
      return val[CommonPushValue::MfDimIndex()];
    }
    __host__ __device__ float& EmbedG(float* val) {
      return val[CommonPushValue::EmbedGIndex()];
    }
    __host__ __device__ float* EmbedxG(float* val) {
      return val + CommonPushValue::EmbedxGIndex();
    }
  };

  struct CommonPullValue {
    /*
       float show;
       float click;
       float embed_w;
       std::vector<float> embedx_w;
       */

    __host__ __device__ static int Dim(int embedx_dim) { return 3 + embedx_dim; }
    __host__ __device__ int DimSize(size_t dim) { return sizeof(float); }
    __host__ __device__ int Size(int embedx_dim) { return Dim(embedx_dim) * sizeof(float); }
    __host__ __device__ int ShowIndex() { return 0; }
    __host__ __device__ int ClickIndex() { return 1; }
    __host__ __device__ int EmbedWIndex() { return 2; }
    __host__ __device__ int EmbedxWIndex() { return 3; }
    __host__ __device__ float& Show(float* val) {
      return val[CommonPullValue::ShowIndex()];
    }
    __host__ __device__ float& Click(float* val) {
      return val[CommonPullValue::ClickIndex()];
    }
    __host__ __device__ float& EmbedW(float* val) {
      return val[CommonPullValue::EmbedWIndex()];
    }
    __host__ __device__ float* EmbedxW(float* val) {
      return val + CommonPullValue::EmbedxWIndex();
    }
  };

 
  __host__ __device__ CommonFeatureValueAccessor() {}
  __host__ __device__ ~CommonFeatureValueAccessor() {}

  __host__ __device__ virtual int Initialize() {
    int optimizer_type = (_config.find("optimizer_type") == _config.end())
                                 ? 1
                                 : int(_config["optimizer_type"]);
    int sparse_embedx_dim = (_config.find("embedx_dim") == _config.end())
                                ? 8
                                : int(_config["embedx_dim"]);
    if (optimizer_type == 3) { //adam
      common_feature_value.embed_sgd_dim = 4;
      common_feature_value.embedx_sgd_dim = sparse_embedx_dim * 2 + 2;
    } else if (optimizer_type == 4) { //shared_adam
      common_feature_value.embed_sgd_dim = 4;
      common_feature_value.embedx_sgd_dim = 4;
    } else {
      common_feature_value.embed_sgd_dim = 1;
      common_feature_value.embedx_sgd_dim = 1;
    }
    common_feature_value.optimizer_type_ = optimizer_type;
    common_feature_value.embedx_dim = sparse_embedx_dim;

    return 0;
  }


// build阶段从cpu_val赋值给gpu_val
__host__ __device__ void BuildFill(float* gpu_val, 
                                float* cpu_val,
                                paddle::distributed::CtrDymfAccessor* cpu_table_accessor,
                                int mf_dim,
                                size_t cpu_fv_dim) {

  gpu_val[common_feature_value.DeltaScoreIndex()] =
      cpu_val[cpu_table_accessor->common_feature_value.DeltaScoreIndex()];
  gpu_val[common_feature_value.ShowIndex()] =
      cpu_val[cpu_table_accessor->common_feature_value.ShowIndex()];
  gpu_val[common_feature_value.ClickIndex()] = 
      cpu_val[cpu_table_accessor->common_feature_value.ClickIndex()];
  gpu_val[common_feature_value.SlotIndex()] = 
      cpu_val[cpu_table_accessor->common_feature_value.SlotIndex()];
  gpu_val[common_feature_value.EmbedWIndex()] = 
      cpu_val[cpu_table_accessor->common_feature_value.EmbedWIndex()];
  for (int i = 0; i < common_feature_value.EmbedDim(); i++) {
    gpu_val[common_feature_value.EmbedG2SumIndex() + i] =
      cpu_val[cpu_table_accessor->common_feature_value.EmbedG2SumIndex() + i];
  }

  cpu_val[cpu_table_accessor->common_feature_value.MfDimIndex()] = float(mf_dim);
  gpu_val[common_feature_value.MfDimIndex()] = mf_dim;
  if (cpu_fv_dim > cpu_table_accessor->GetAccessorInfo().dim - 
          cpu_table_accessor->GetAccessorInfo().mf_size / sizeof(float)) {
    gpu_val[common_feature_value.MfSizeIndex()] = 
        common_feature_value.MFSize(mf_dim) / sizeof(float);

    for (int x = 0; x < int(common_feature_value.MFSize(mf_dim) / sizeof(float));
            x++) {
      gpu_val[common_feature_value.EmbedxG2SumIndex() + x]  =
        cpu_val[cpu_table_accessor->common_feature_value.EmbedxG2SumIndex() + x];
    }
  } else {
    gpu_val[common_feature_value.MfSizeIndex()] = 0;
    for (int x = common_feature_value.EmbedxG2SumIndex(); 
          x < int(common_feature_value.Size(mf_dim) / sizeof(float)); x++){
      gpu_val[x] = 0;
    }
  }
}


// dump_to_cpu阶段从gpu_val赋值给cpu_val
__host__ __device__ void DumpFill(float* cpu_val, 
                                float* gpu_val,
                                paddle::distributed::CtrDymfAccessor* cpu_table_accessor,
                                int mf_dim) {

  cpu_val[cpu_table_accessor->common_feature_value.DeltaScoreIndex()] =
      gpu_val[common_feature_value.DeltaScoreIndex()];
  cpu_val[cpu_table_accessor->common_feature_value.ShowIndex()] = 
      gpu_val[common_feature_value.ShowIndex()];
  cpu_val[cpu_table_accessor->common_feature_value.ClickIndex()] = 
      gpu_val[common_feature_value.ClickIndex()];
  cpu_val[cpu_table_accessor->common_feature_value.EmbedWIndex()] = 
      gpu_val[common_feature_value.EmbedWIndex()];
  cpu_val[cpu_table_accessor->common_feature_value.SlotIndex()] = 
      gpu_val[common_feature_value.SlotIndex()];

  for (int i = 0; i < common_feature_value.EmbedDim(); i++) {
    cpu_val[cpu_table_accessor->common_feature_value.EmbedG2SumIndex() + i] = 
      gpu_val[common_feature_value.EmbedG2SumIndex() + i];
  }

  if (gpu_val[common_feature_value.MfSizeIndex()] > 0) {
    
    for (int x = 0; x < int(common_feature_value.MFSize(mf_dim) / sizeof(float));
              x++) {
        cpu_val[cpu_table_accessor->common_feature_value.EmbedxG2SumIndex() + x]  = 
          gpu_val[common_feature_value.EmbedxG2SumIndex() + x];
    }
  }
}


// dy_mf_fill_dvals_kernel, dy_mf_search_kernel 阶段 gpukernel 中从src_val赋值给dest_val
__host__ __device__ void FeatureValueFill(float* dest_val, 
                                float* src_val,
                                int mf_dim) {
  *(reinterpret_cast<uint64_t*>(dest_val + common_feature_value.CpuPtrIndex())) =
    *(reinterpret_cast<uint64_t*>(src_val + common_feature_value.CpuPtrIndex()));
  dest_val[common_feature_value.DeltaScoreIndex()] = src_val[common_feature_value.DeltaScoreIndex()];
  dest_val[common_feature_value.ShowIndex()] = src_val[common_feature_value.ShowIndex()];
  dest_val[common_feature_value.ClickIndex()] = src_val[common_feature_value.ClickIndex()];
  dest_val[common_feature_value.EmbedWIndex()] = src_val[common_feature_value.EmbedWIndex()];
  for (int i = 0; i < common_feature_value.EmbedDim(); i++) {
    dest_val[common_feature_value.EmbedG2SumIndex() + i] = 
      src_val[common_feature_value.EmbedG2SumIndex() + i];
  }
  dest_val[common_feature_value.SlotIndex()] = src_val[common_feature_value.SlotIndex()];
  dest_val[common_feature_value.MfDimIndex()] = mf_dim;
  dest_val[common_feature_value.MfSizeIndex()] = src_val[common_feature_value.MfSizeIndex()];

  for (int x = common_feature_value.EmbedxG2SumIndex();
          x < int(common_feature_value.Size(mf_dim) / sizeof(float)); x++){
    dest_val[x] = src_val[x];
  }
}


// dy_mf_fill_shard_grads_kernel,update_one 阶段 gpukernel 中从src_val赋值给dest_val
__host__ __device__ void PushValueFill(float* dest_val, 
                                const float* src_val) {
  dest_val[common_push_value.SlotIndex()] = src_val[common_push_value.SlotIndex()];
  dest_val[common_push_value.ShowIndex()] = src_val[common_push_value.ShowIndex()];
  dest_val[common_push_value.ClickIndex()] = src_val[common_push_value.ClickIndex()];
  dest_val[common_push_value.MfDimIndex()] = src_val[common_push_value.MfDimIndex()];
  dest_val[common_push_value.EmbedGIndex()] = src_val[common_push_value.EmbedGIndex()];

  for (int x = 0; x < int(src_val[common_push_value.MfDimIndex()]); x++) {
    dest_val[common_push_value.EmbedxGIndex() + x] =  src_val[common_push_value.EmbedxGIndex() + x];
  }
}

// update_basic 阶段 gpukernel 中从src_val赋值给dest_val
__host__ __device__ void PushValueFillBasic(float* dest_val, 
                                const float* src_val) {
  dest_val[common_push_value.SlotIndex()] = src_val[common_push_value.SlotIndex()];
  dest_val[common_push_value.ShowIndex()] = src_val[common_push_value.ShowIndex()];
  dest_val[common_push_value.ClickIndex()] = src_val[common_push_value.ClickIndex()];
  dest_val[common_push_value.MfDimIndex()] = src_val[common_push_value.MfDimIndex()];
  dest_val[common_push_value.EmbedGIndex()] = src_val[common_push_value.EmbedGIndex()];

}


// merge_one 阶段 gpukernel 中 PushValue 从src_val赋值给dest_val
__host__ __device__ void MergePushValue(float* dest_val, 
                                const float* src_val) {
  dest_val[common_push_value.ShowIndex()] +=  src_val[common_push_value.ShowIndex()];
  dest_val[common_push_value.ClickIndex()] += src_val[common_push_value.ClickIndex()];
  dest_val[common_push_value.EmbedGIndex()] += src_val[common_push_value.EmbedGIndex()];
  for (int j = 0; j < int(dest_val[common_push_value.MfDimIndex()]); j++) {
    dest_val[common_push_value.EmbedxGIndex() + j] += src_val[common_push_value.EmbedxGIndex() + j];
  }
}


// merge_basic 阶段 gpukernel 中 PushValue 从src_val赋值给dest_val
__host__ __device__ void MergePushValueBasic(float* dest_val, 
                                const float* src_val) {
  dest_val[common_push_value.ShowIndex()] +=  src_val[common_push_value.ShowIndex()];
  dest_val[common_push_value.ClickIndex()] += src_val[common_push_value.ClickIndex()];
  dest_val[common_push_value.EmbedGIndex()] += src_val[common_push_value.EmbedGIndex()];
}

// PullCopy 阶段 gpukernel 中  FeatureValue回填到PullValue
__host__ __device__ void Select(float* dest_val, 
                                float* src_val,
                                uint64_t* key,
                                int mf_dim) {
  if (*key == 0) {
      *(dest_val + common_pull_value.ShowIndex()) = 0;
      *(dest_val + common_pull_value.ClickIndex()) = 0;
      *(dest_val + common_pull_value.EmbedWIndex()) = 0;
    } else {
      *(dest_val + common_pull_value.ShowIndex()) = src_val[common_feature_value.ShowIndex()];
      *(dest_val + common_pull_value.ClickIndex()) = src_val[common_feature_value.ClickIndex()];
      *(dest_val + common_pull_value.EmbedWIndex()) = src_val[common_feature_value.EmbedWIndex()];
    }

    if (src_val[common_feature_value.MfSizeIndex()] == 0 || *key == 0) {
      for (int j = 0; j < mf_dim; j++) {
        *(dest_val + common_pull_value.EmbedxWIndex() + j) = 0;
      }
    } else {
      for (int j = 0; j < mf_dim; j++) {
        *(dest_val + common_pull_value.EmbedxWIndex() + j) = 
            src_val[common_feature_value.EmbedxWOffsetIndex(src_val) + j];
      }
    }
}


  __host__ __device__ std::string ParseToString(const float* v, int param_size) {
    /*
        uint64_t cpu_ptr; // 2float
        float delta_score;
        float show;
        float click;
        float embed_w;
        std::vector<float> embed_g2sum;
        float slot;
        float mf_dim
        float mf_size
        std::vector<float> embedx_g2sum;
        std::vector<float> embedx_w;
    */
    std::stringstream os;
    os << "cpuptr: " << common_feature_value.CpuPtr(const_cast<float*>(v)) << " delta_score: " << v[2] 
        << " show: " << v[3] << " click: " << v[4] 
        << " embed_w:" << v[5] << " embed_g2sum:";
    for (int i = common_feature_value.EmbedG2SumIndex();
        i < common_feature_value.SlotIndex(); i++) {
      os << " " << v[i];
    }
    int mf_dim = int(common_feature_value.MfDim(const_cast<float*>(v)));
    os << " slot: " << common_feature_value.Slot(const_cast<float*>(v)) 
      << " mf_dim: " << mf_dim
      << " mf_size: " << common_feature_value.MfSize(const_cast<float*>(v))
      << " mf: ";
    if (param_size > common_feature_value.EmbedxG2SumIndex()) {
      for (auto i = common_feature_value.EmbedxG2SumIndex();
          i < common_feature_value.Dim(mf_dim); ++i) {
        os << " " << v[i];
      }
    }
    return os.str();
  }

 public:
  CommonFeatureValue common_feature_value;
  CommonPushValue common_push_value;
  CommonPullValue common_pull_value;
};


struct FeatureValue {
  float delta_score;
  float show;
  float clk;
  int slot;
  float lr;
  float lr_g2sum;
  int mf_size;
  int mf_dim;
  uint64_t cpu_ptr;
  float mf[0];

  friend std::ostream& operator<<(std::ostream& out, FeatureValue& val) {
    out << "show: " << val.show << " clk: " << val.clk << " slot: " << val.slot
        << " lr: " << val.lr << " mf_dim: " << val.mf_dim
        << "cpuptr: " << val.cpu_ptr << " mf_size: " << val.mf_size << " mf:";
    for (int i = 0; i < val.mf_dim + 1; ++i) {
      out << " " << val.mf[i];
    }
    return out;
  }
  __device__ __forceinline__ void operator=(const FeatureValue& in) {
    delta_score = in.delta_score;
    show = in.show;
    clk = in.clk;
    slot = in.slot;
    lr = in.lr;
    lr_g2sum = in.lr_g2sum;
    mf_size = in.mf_size;
    mf_dim = in.mf_dim;
    cpu_ptr = in.cpu_ptr;
    for (int i = 0; i < mf_dim + 1; i++) {
      mf[i] = in.mf[i];
    }
  }
};

struct FeaturePushValue {
  float show;
  float clk;
  int slot;
  float lr_g;
  int mf_dim;
  float mf_g[0];

  __device__ __forceinline__ FeaturePushValue
  operator+(const FeaturePushValue& a) const {
    FeaturePushValue out;
    out.slot = a.slot;
    out.mf_dim = a.mf_dim;
    out.show = a.show + show;
    out.clk = a.clk + clk;
    out.lr_g = a.lr_g + lr_g;
    // out.mf_g = a.mf_g;
    for (int i = 0; i < out.mf_dim; ++i) {
      out.mf_g[i] = a.mf_g[i] + mf_g[i];
    }
    return out;
  }
  __device__ __forceinline__ void operator=(const FeaturePushValue& in) {
    show = in.show;
    clk = in.clk;
    slot = in.slot;
    lr_g = in.lr_g;
    mf_dim = in.mf_dim;
    for (int i = 0; i < mf_dim; i++) {
      mf_g[i] = in.mf_g[i];
    }
  }
};

}  // end namespace framework
}  // end namespace paddle
#endif
