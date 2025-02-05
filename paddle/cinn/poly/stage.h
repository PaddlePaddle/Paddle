// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <isl/cpp.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/poly/domain.h"
#include "paddle/cinn/poly/map.h"

namespace cinn {
namespace poly {
using ir::DeviceAPI;

struct ComputeAtRelation;

enum class ScopeKind {
  kLocal = 0,
  kShared = 1,
  kGlobal = 2,
};

struct StageForloopInfo {
  StageForloopInfo() = default;
  StageForloopInfo(ir::ForType for_type, ir::DeviceAPI device, uint8_t offset)
      : for_type(for_type), device(device), offset(offset) {}

  ir::ForType for_type;
  //! The offset in the \p for_type. e.g. for GPUBlock, 0 represents blockIdx.x,
  //! 1 is blockIdx.y, 2 is blockIdx.z.
  uint8_t offset;
  ir::DeviceAPI device;
};

//! Store the information about some other tensor `compute_at` this tensor.
struct ComputeAtInfo {
  ComputeAtInfo(const std::string& consumer_tensor_name,
                const std::string& producer_tensor_name,
                const std::vector<int>& adjusted_producer_shape,
                const std::vector<int>& preceding_offset_for_producer_load,
                int level)
      : consumer_tensor_name(consumer_tensor_name),
        producer_tensor_name(producer_tensor_name),
        adjusted_producer_shape(adjusted_producer_shape),
        preceding_offset_for_producer_load(preceding_offset_for_producer_load),
        level(level) {}

  std::string consumer_tensor_name;
  std::string producer_tensor_name;
  //! The shape of the buffer belong to the producer tensor after compute_at.
  //! NOTE this doesn't support dynamic dimension yet.
  std::vector<int> adjusted_producer_shape;
  //! The preceding offsets for the indice in the Loads for the producers, the
  //! offset will make the minimum indice to be 0, size of this should equal to
  //! level+1.
  std::vector<int> preceding_offset_for_producer_load;
  //! the level of the consumer tensor's transformed range.
  int level{-1};
};

/**
 * Meta information for tensor.
 */
struct TensorScheduleMeta {
  //! Store the information of all the other producer tensors `compute_at` this
  //! tensor.
  std::vector<ComputeAtInfo> compute_at_infos;

  bool compute_inline{false};

  //! Name of the tensors those share buffer with `this` tensor.
  std::set<std::string> tensors_to_share_buffer_with;
};

/**
 * Stage is the basic element of polyhedral which represents a stage in CINN.
 * It supports multiple transforms such as tile, split and so on.
 */
class Stage : public Object {
 public:
  static Shared<Stage> New(const isl::set& domain,
                           Expr expr = Expr(),
                           ir::_Tensor_* tensor = nullptr);

  TensorScheduleMeta meta;

  /**
   * The id of this element, should be unique across the transform.
   */
  const char* id() const;

  //! Expression contained in this stage.
  const Expr& expr() const { return expr_; }
  //! Change this stage's domain to be consistent with other's domain.
  void ChangeDomain(Stage* other, int level);
  //! Add for loop in this stage's transform and replace this tensor's index in
  //! this tensor's compute body.
  void ChangeIndex(Stage* other);
  //! Get the i-th axis.
  Iterator axis(int i) const;
  //! Get the axis named \p i.
  Iterator axis(const std::string& i) const;
  //! Get the original reduce axis names.
  std::vector<std::string> origin_reduce_axis_names();

  std::vector<std::string> axis_names() const;

  ir::_Tensor_* tensor() { return tensor_; }

  /**
   * Mark this stage to expand inplace in all the usages.
   */
  void ComputeInline();
  void DisableComputeInline();

  bool inlined() const { return meta.compute_inline; }

  /**
   * Mark this buffer should share buffer with \p other.
   */
  void ShareBufferWith(Stage* other);

  /**
   * Split the loop level of into two new loop levels.
   * @param level the level to split.
   * @param factor the extent(size) of the inner loop created after splitting.
   * @return the new outer and inner iterators.
   */
  // @{
  std::tuple<Iterator, Iterator>  //
  Split(const Iterator& level, int factor);
  std::tuple<Iterator, Iterator>  //
  Split(const std::string& level, int factor);
  std::tuple<Iterator, Iterator>  //
  Split(int level, int factor);
  // @}

  /**
   * Split the loop level of into two new loop levels.
   * @param level the level to split.
   * @param nparts the extent(size) of the outer loop created after splitting.
   * @return the new outer and inner iterators.
   */
  // @{
  std::tuple<Iterator, Iterator>  //
  SplitOuter(const Iterator& level, int nparts);
  std::tuple<Iterator, Iterator>  //
  SplitOuter(const std::string& level, int nparts);
  std::tuple<Iterator, Iterator>  //
  SplitOuter(int level, int nparts);
  // @}

  /**
   * Reorder the iterators.
   * @param order the order of all the iterators.
   */
  void Reorder(const std::vector<Iterator>& order);
  void Reorder(const std::vector<int>& order);

  /**
   * Tile the two loop levels \p level0 and \p level1 with rectangular tiling.
   * @param level0 the first level.
   * @param level1 the second level.
   * @param factor0 tiling size of the first level.
   * @param factor1 tiling size of the second level.
   * @return the new iterators.
   */
  std::tuple<Iterator, Iterator, Iterator, Iterator>  //
  Tile(const Iterator& level0,
       const Iterator& level1,
       int factor0,
       int factor1);
  std::tuple<Iterator, Iterator, Iterator, Iterator>  //
  Tile(int level0, int level1, int factor0, int factor1);

  int GetDimRange(int level);

  /**
   * Vectorize the stage in \p level.
   * @param level
   */
  void Vectorize(int level, int factor);
  void Vectorize(const std::string& axis, int factor);
  void Vectorize(const Iterator& axis, int factor);

  /**
   * Parallel a for-loop.
   * @param level
   */
  void Parallel(int level);
  void Parallel(const std::string& axis);
  void Parallel(const Iterator& axis);

  /**
   * Unroll a for-loop.
   */
  void Unroll(int level);
  void Unroll(const std::string& level);
  void Unroll(const Iterator& level);

  void Bind(int level, const std::string& axis);

  enum ComputeAtKind {
    kComputeAtAuto,
    kComputeAtBefore,
    kComputeAtAfter,
  };

  /**
   * Apply loop skewing on the loop levels \p i and \p j with a skewing factor
   * of \p factor.
   * TODO(Superjomn) Refine this transform.
   */
  std::tuple<Iterator, Iterator>  //
  Skew(const Iterator& i, const Iterator& j, int factor);

  //! Add a control dependency link to \p t.
  void CtrlDepend(const ir::Tensor& t);
  //! Get the tensors control depend on.
  const std::set<ir::Tensor>& ctrl_depends() const;

  /**
   * Set the memory type of this stage's tensor.
   * @param memory_type the memory type of this tensor. For example,
   * memory_type="shared".
   */
  void SetBuffer(const std::string& memory_type);

  /**
   * Given two stages already satisfy ComputeAtRelation.IsCompatible, set
   * compute_ats_ for them.
   * @param other the other stage to set compute_ats_.
   * @param level the level of ComputeAtRelation.
   */
  void SimpleComputeAt(Stage* other, int level);

  /**
   * \brief Mark the stage compute at the level of some other stage. Usually
   * used when there is no access relation between two tensors.
   *
   * The difference between ComputeAt2 and ComputeAt is that ComputeAt2 can be
   * used when there is no access relation between two tensors.
   *
   * @param other the target stage to compute at.
   * @param level the level of \p other's forloop to compute at
   */
  void ComputeAt2(Stage* other, int level);

  // Do ComputeAt2 except for setting the ComputeAt level, which is moving the
  // computations together.
  void ComputeAt3(Stage* other, int level);

  /**
   * \brief Mark the stage compute at the level of some other stage.
   *
   * NOTE This can only be called after all transformations are preformed, and
   * once called, no further transform can perform for that if the iterators are
   * changed, the original `ComputeAt` level will become invalid.
   *
   * @param other the target stage to compute at.
   * @param level the level of \p other's forloop to compute at
   */
  void ComputeAt(Stage* other, int level);

  void ShowISL() const;

  void AddForLoopInTransform(
      std::vector<std::vector<Expr>>& indices);  // NOLINT

  /**
   * Set thread scope.
   */
  void SetScope(ScopeKind scope) { scope_ = scope; }

  /**
   * Get thread scope.
   */
  ScopeKind scope() const { return scope_; }

  /**
   * \brief Fuse two forloop levels and return the new level.
   * @param level0 the first level.
   * @param level1 the second level.
   * @return the new level.
   */
  Iterator Fuse(const Iterator& level0, const Iterator& level1);
  Iterator Fuse(const std::vector<Iterator>& levels);
  Iterator Fuse(int level0, int level1);
  Iterator Fuse(const std::vector<int>& levels);
  Iterator FuseDirect(const std::vector<int>& levels);
  Iterator Fuse(const std::string& level0, const std::string& level1);
  const isl::set& domain() const { return domain_; }
  const isl::map& transform() const { return transform_; }
  isl::set transformed_domain() const;

  // Dealing with the `ComputeAt` transform.
  std::vector<ComputeAtRelation> compute_ats() const;

  //! Get the level-th dimensional name.
  std::string ith_dim_name(int level);
  //! Get the i-th iterator.
  Iterator ith_iterator(int level);

  /** Get the final level after all the transforms.
   * The level will be affected by some schedule like ComputeAt, this will
   * return the right level.
   *
   * @param level the level in schedule.
   */
  int GetTransformedLevel(int level);

  //! Get the statements.
  std::vector<std::string> input_statements() const;

  virtual const char* type_info() const { return __type_info__; }

  inline const ir::VectorizeInfo& vectorize_info() const {
    return vectorize_info_;
  }
  inline const std::set<int>& unroll_info() const { return unroll_info_; }
  inline const std::set<int>& parallel_info() const { return parallel_info_; }
  inline std::map<std::string, ComputeAtRelation>& GetComputeAts() {
    return compute_ats_;
  }
  inline void SetComputeAts(
      const std::map<std::string, ComputeAtRelation>& compute_ats) {
    compute_ats_ = compute_ats;
  }

  /*
  const std::set<std::string>& extra_depend_stages() const { return
  extra_depend_stages_; } void set_extra_depend_stages(const
  std::set<std::string>& x) { extra_depend_stages_ = x; } void
  add_extra_depend_stage(const std::string& statement) {
  extra_depend_stages_.insert(statement); }
   */

  const std::map<int /*level*/, StageForloopInfo>& forloop_infos() const {
    return forloop_infos_;
  }

  bool has_expression() const;

  Stage() = default;

  void ComputeAtSchedule(Stage* other,
                         int level,
                         ComputeAtKind kind = kComputeAtAuto);

  ir::Tensor LookupCtrlDepend(const std::string& tensor_name) const;

  //! Get number of transform output dimensions, this equals to the number of
  //! forloops in generated code.
  inline int n_in_dims() const {
    return isl_map_dim(transform_.get(), isl_dim_in);
  }
  //! Get number of transform output dimensions, this equals to the number of
  //! dimensions of corresponding tensor.
  inline int n_out_dims() const {
    return isl_map_dim(transform_.get(), isl_dim_out);
  }

  //! Copy other stage's transform.
  //! For example, if the target_transform is `Split(0,1)`,
  //! this api will apply `Split(0,1)` on itself.
  void CopyTransform(Stage* other, int level = -1);
  //! Edit temp tensor's shape, its buffer's shape and index when doing
  //! ComputeAt2.
  void EditTempTensor(Stage* other, int level);
  //! Copy other stage's LoopInfo.
  //! For example, if the target_forloop_infos is `Bind(0,"threadIdx.x")`,
  //! this api will apply `Bind(0,"threadIdx.x")` on itself.
  void CopyLoopInfo(Stage* other);
  //! Set stage's transform_
  void SetTransform(isl::map new_transform) { transform_ = new_transform; }
  //! Set stage's forloop_infos_
  void SetForloopInfo(std::map<int, StageForloopInfo> forloop_infos) {
    forloop_infos_ = forloop_infos;
  }
  void AddForloopInfo(int level, const StageForloopInfo& info);
  bool IfCudaBind() { return cuda_bind_info_; }

 private:
  explicit Stage(const isl::set& domain,
                 Expr expr = Expr(),
                 ir::_Tensor_* tensor = nullptr);

  /**
   * Initialize with an identity schedule.
   */
  void InitTransform();

  //! Lock the \p level-th axis and disallow the further schedules on this axis.
  void LockAxis(uint32_t level);
  //! Unlock the \p level-th axis.
  void UnlockAxis(uint32_t level);
  //! Tell if the \p level -th axis is locked.
  bool is_axis_locked(uint32_t level) const;
  //! Assert that the axis is not locked, abort if fail.
  void AssertAxisIsNotLocked(uint32_t level);

  static constexpr char* __type_info__ = "Stage";

 private:
  isl::set domain_;
  isl::map transform_;
  Expr expr_;
  // this compute_at some other stages.
  std::map<std::string, ComputeAtRelation> compute_ats_;
  ir::VectorizeInfo vectorize_info_;
  //! The for-loop levels to unroll.
  std::set<int> unroll_info_;
  //! The for-loop levels to parallel.
  std::set<int> parallel_info_;
  //! Record some forloop levels' information.
  std::map<int /*level*/, StageForloopInfo> forloop_infos_;
  //! A weak reference to the tensor.
  ir::_Tensor_* tensor_{};
  //! Thread scope.
  ScopeKind scope_{ScopeKind::kGlobal};
  std::set<ir::Tensor> ctrl_depends_;

  std::set<int> locked_axis_;
  bool cuda_bind_info_{false};

  friend isl_map* __isl_give GatherAccesses(Stage* stage,
                                            const std::string& tensor_name);
  friend class PolyGroupScheduler;
};

std::vector<std::pair<std::string, std::string>> ExtractExtraDepLinksFromStages(
    const std::vector<Stage*>& stages);

//! This stage compute_at some other stage.
struct ComputeAtRelation {
  //! the other stage.
  Shared<Stage> stage;
  int level{-1};

  //! Check whether the stage \p self is compatible with \p stage.
  bool IsCompatible(Stage* self);
};

//! Return the corresponding inner iterator name.
inline std::string InnerName(const std::string& name);
inline std::string InnerName(const Iterator& iterator);
//! Return the corresponding inner iterator name.
inline std::string OuterName(const std::string& name);
inline std::string OuterName(const Iterator& iterator);

inline Iterator DefaultIterator(int i) {
  return Iterator(cinn::common::axis_name(i));
}

/**
 * Collect the access to a tensor named \p tensor_name in \p stage.
 */
std::vector<isl::map> GatherAccesses(const Stage* stage,
                                     const std::string& tensor_name);

class _StageMap_ : public Object {
 public:
  /**
   * Get a stage from the stage map.
   * NOTE The stage should exists, or it will abort.
   */
  // @{
  Stage* operator[](const ir::Tensor& tensor);
  const Stage* operator[](const ir::Tensor& tensor) const;
  Stage* operator[](const ir::_Tensor_* tensor);
  const Stage* operator[](const ir::_Tensor_* tensor) const;
  // @}

  //! Insert a stage into the map, it will replace if an older one exists.
  Stage* Insert(const ir::Tensor& key, Stage* stage);
  //! Insert a stage only if not exists.
  Stage* InsertLazily(const ir::Tensor& key);

  Stage* InsertLazily(const ir::Tensor& key, Stage* stage);

  //! Lookup a tensor from the map, return nullptr if not exists.
  Stage* Lookup(const std::string& name) const;

  inline size_t size() const { return data_.size(); }

  const char* type_info() const override { return __type_info__; }

  static constexpr const char* __type_info__ = "StageMap";

 private:
  absl::flat_hash_map<std::string, Shared<Stage>> data_;
};
}  // namespace poly
}  // namespace cinn
