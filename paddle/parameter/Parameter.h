/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>

#include "ParameterConfig.pb.h"
#include "TrainerConfig.pb.h"

#include "ParameterUpdaterHook.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/ThreadLocal.h"
#include "paddle/utils/Util.h"

namespace paddle {

class SparsePrefetchRowCpuMatrix;

class Parameter;
typedef std::function<void(Parameter* param)> UpdateCallback;
typedef std::function<void(int paramId, Parameter* param)> ParamInitCallback;

class Parameter;
typedef std::shared_ptr<Parameter> ParameterPtr;

class Parameter {
public:
  Parameter(const ParameterConfig& config, bool useGpu, bool doInit = true);
  const std::string& getName() const { return config_.name(); }

  size_t getSize() const { return config_.size(); }

  bool isFullSize() const {
    return this->getSize() == bufs_[PARAMETER_VALUE]->getSize();
  }

  inline bool useGpu() const { return useGpu_; }

  int getDeviceId() const { return deviceId_; }

  void setDevice(int deviceId) { deviceId_ = deviceId; }

  /// The id ranges from 0 to the_total_number_of_parameters - 1
  size_t getID() const { return config_.para_id(); }

  /// ID is a implict value created until neural network is built.
  void setID(size_t id) { config_.set_para_id(id); }

  bool isStatic() const { return config_.is_static(); }

  enum MatType {
    MAT_NORMAL,
    /// both value and grad are shared
    MAT_NORMAL_SHARED,

    /// Now used in BatchNorm in CPU mode
    MAT_VALUE_SHARED,

    /// sparse matrix, which has full size parameter
    MAT_SPARSE_ROW_IDS,
    /// sparse matrix, parameter size scale by sparse rates.
    MAT_SPARSE_ROW_AUTO_GROW,
    MAT_CACHE_ROW,
    MAT_SPARSE_ROW,

    /// sparse matrix for prefetching parameter from pserver
    MAT_SPARSE_ROW_PREFETCH,
    /// same as above, but parameter has full size for saving parameter in local
    MAT_SPARSE_ROW_PREFETCH_FULL_SIZE,
  };

  void enableSparseParameter() {
    if (config_.is_sparse()) {
      if (config_.format() == "csr") {
        size_t height = config_.dims(0);
        size_t nnz = config_.size();
        enableIntType(PARAMETER_ROWS, height + 1);
        enableIntType(PARAMETER_COLS, nnz);
        format_ = SPARSE_CSR;
      } else {
        size_t width = config_.dims(1);
        size_t nnz = config_.size();
        enableIntType(PARAMETER_COLS, width + 1);
        enableIntType(PARAMETER_ROWS, nnz);
        format_ = SPARSE_CSC;
      }
    }
  }

  /// allocate buffer for the give type
  void enableType(ParameterType type, MatType matType = MAT_NORMAL) {
    if (bufs_[type] || mats_[type]) {
      return;
    }
    SetDevice device(deviceId_);
    if (config_.dims_size() == 2) {
      if (matType == MAT_NORMAL || matType == MAT_NORMAL_SHARED ||
          matType == MAT_SPARSE_ROW_PREFETCH_FULL_SIZE ||
          matType == MAT_VALUE_SHARED || matType == MAT_SPARSE_ROW_IDS) {
        bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
        bufs_[type]->zeroMem();
      } else {
        CHECK(isGradSparseUpdate());
      }
      if (config_.is_sparse() && type == PARAMETER_VALUE) {
        enableSparseParameter();
      }
      setMat(type, matType);
    } else {
      bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
      bufs_[type]->zeroMem();
    }
  }

  void enableBufType(ParameterType type) {
    if (bufs_[type]) return;
    bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
    bufs_[type]->zeroMem();
  }

  void enableIntType(ParameterType type, size_t intStoreSize = 0) {
    if (!intBufs_[type]) {
      SetDevice device(deviceId_);
      size_t size = intStoreSize ? intStoreSize : config_.size();
      intBufs_[type] = IVector::create(size, useGpu_);
      intBufs_[type]->zeroMem();
    }
  }

  void enableSharedType(ParameterType type,
                        VectorPtr vec,
                        MatrixPtr mat = nullptr) {
    if (!bufs_[type] && !mats_[type]) {
      bufs_[type] = vec;
      mats_[type] = mat;
    }
  }

  /// for batchGradientMachine: blockNum is number of partitions of the matrix.
  bool isGradShared(size_t* blockNum = NULL);

  bool isValueShared();

  // for AsgdSparseGradientMachine & SgdSparseGradientMachine:
  // and MultiGradientMachine
  bool isGradSparseUpdate() const;

  bool isSparseRemoteUpdate() const {
    return config_.sparse_remote_update() && !useGpu();
  }

  const ParameterConfig& getConfig() const { return config_; }

  ParameterConfig& getConfig() { return config_; }

  bool hasType(ParameterType pType) const {
    return bufs_[pType] || mats_[pType];
  }

  const VectorPtr& getBuf(ParameterType pType) const {
    return this->bufs_[pType];
  }

  const VectorPtr* getBufs() const { return bufs_; }

  const MatrixPtr& getMat(ParameterType pType) const { return mats_[pType]; }

  void setValueUpdated() { updated_ = true; }

  void clearValueUpdated() { updated_ = false; }

  bool isValueUpdated() const { return updated_; }

  /**
   * Save parameter value to a file
   */
  bool save(const std::string& filename) const;

  /**
   * Save parameter to ostream
   */
  bool save(std::ostream& s) const;

  /**
   * Load parameter value from a file
   */
  bool load(const std::string& filename);

  /**
   * Load parameter from istream
   */
  bool load(std::istream& is);

  void incShared() { sharedCount_++; }

  /**
   * After one of the parameter's gradient is merged
   * You should call this function to do some additional processing,
   */
  void incUpdate(const UpdateCallback& callbacks = NULL);

  void clearGradient() {
    auto& mat = getMat(PARAMETER_GRADIENT);
    if (mat) {
      // zeroMem will also clear rows for SparseRowCpuMatrix
      mat->zeroMem();
    } else {
      auto& gradBuf = getBuf(PARAMETER_GRADIENT);
      if (gradBuf) gradBuf->zeroMem();
    }
  }

  void initialize();

  /**
   * Initialize the value according to config_: initial_mean,
   * initial_std and initial_strategy.
   */
  void randomize();
  static void randomize(const VectorPtr& value, const ParameterConfig& config);

  /// Initialize the value to 0
  void zeroMem();

  static const int kFormatVersion = 0;
  /// file header structure
  struct Header {
    int32_t version;     // = 0, file format version
    uint32_t valueSize;  // = sizeof(real)
    uint64_t size;       // = getSize()
  };

  /**
   * @brief  Parameter Update Hook.
   *
   * The parameter's update hook before ParameterUpdater::updateImpl
   * It could modify gradient/momentum/etc here. Such as drop some gradient,
   * etc.
   */
  void updateHook() {
    for (auto& hook : updaterHooks_) {
      hook->update(this);
    }
  }

  /**
   * @brief  Initialize all updater hook.
   *
   * This method should be invoked in ParameterUpdater::init() only.
   */
  void initHook() {
    for (auto& hook : updaterHooks_) {
      hook->init(this);
    }
  }

protected:
  /**
   * @brief create matrix to matType.
   *
   * used by gradient machine which needs specify matrix type,
   * instead of creating in weights.cpp.
   *
   * @note  pType should be enabled already.
   */
  void setMat(ParameterType pType, int matType);

  bool isUpdatable() { return (updateCounter_ == sharedCount_); }

  void clearUpdate() { updateCounter_ = 0; }

protected:
  ParameterConfig config_;

  bool useGpu_;

  int deviceId_;

  /**
   * @brief bufs_ stores parameter value and gradient.
   *
   * Layer should use bufs_[PARAMETER_VALUE] to form weight matrix for
   * calculation and stores gradient to bufs_[PARAMETER_GRADIENT].
   */
  VectorPtr bufs_[NUM_PARAMETER_TYPES];

  /**
   * @brief Weight matrix for bufs_.
   *
   * It's helpfull when parameter shared by multi-layers.
   * Caller should check, if mats exist, do not create it again.
   */
  MatrixPtr mats_[NUM_PARAMETER_TYPES];

  /// Int vectors, used in some User defined parameter types
  IVectorPtr intBufs_[NUM_PARAMETER_TYPES];

  int sharedCount_;
  int updateCounter_;

  bool updated_;
  SparseFormat format_;

  std::vector<std::shared_ptr<IParameterUpdaterHook>> updaterHooks_;

public:
  void setSharedCount(int cnt) { sharedCount_ = cnt; }
  int getSharedCount() { return sharedCount_; }

  bool isSparse() { return config_.is_sparse(); }
  SparseFormat getFormat() { return format_; }

  static const std::string kMissParameterFail;
  static const std::string kMissParameterRand;
  static const std::string kMissParameterZero;
};

typedef std::map<std::string, ParameterPtr> ParameterMap;

}  // namespace paddle
