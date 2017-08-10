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

#include "Parameter.h"
#include <gflags/gflags.h>
#include <fstream>
#include "AverageOptimizer.h"
#include "FirstOrderOptimizer.h"
#include "OptimizerFunctions.h"
#include "OptimizerWithRegularizer.h"
#include "ParameterUpdateFunctions.h"
#include "ThreadLocalBuffer.h"
#include "hl_gpu.h"
#include "paddle/math/CpuSparseMatrix.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/SparseRowMatrix.h"
#include "paddle/utils/Logging.h"

DEFINE_int32(enable_grad_share,
             (100 * 1024 * 1024),
             "threshold for enable gradient parameter share for batch "
             "multi-cpu training");
DEFINE_int32(
    grad_share_block_num,
    64,
    "block number of gradient parameter share for batch multi-cpu training");

namespace paddle {

const std::string Parameter::kMissParameterFail = "fail";
const std::string Parameter::kMissParameterRand = "rand";
const std::string Parameter::kMissParameterZero = "zero";

Parameter::Parameter(const ParameterConfig& config, bool useGpu, bool doInit)
    : config_(config),
      useGpu_(useGpu),
      deviceId_(-1),
      sharedCount_(0),
      updateCounter_(0),
      updated_(false) {
  setID(-1); /* capture uninitialized id */
  if (useGpu_ && FLAGS_parallel_nn) {
    /* gpu environment is specified by device property */
    deviceId_ = config_.device();
    if (deviceId_ < 0) {
      useGpu_ = false;
    }
  }

  if (doInit) {
    initialize();
  }

  for (int i = 0; i < config.update_hooks_size(); ++i) {
    this->updaterHooks_.push_back(IParameterUpdaterHook::create(config, i));
  }
}

void Parameter::initialize() {
  SetDevice device(deviceId_);

  bufs_[PARAMETER_VALUE] =
      Vector::createParallelVector(config_.size(), useGpu_);
  bufs_[PARAMETER_VALUE]->zeroMem();

  if (config_.is_sparse()) {
    enableSparseParameter();
  }

  if (!isStatic()) {
    bufs_[PARAMETER_GRADIENT] =
        Vector::createParallelVector(config_.size(), useGpu_);
    bufs_[PARAMETER_MOMENTUM] =
        Vector::createParallelVector(config_.size(), useGpu_);

    bufs_[PARAMETER_GRADIENT]->zeroMem();
    bufs_[PARAMETER_MOMENTUM]->zeroMem();
  }
}

void Parameter::randomize(const VectorPtr& value,
                          const ParameterConfig& config) {
  if (PARAMETER_INIT_UNIFORM == config.initial_strategy()) {
    // initialize the parameter as uniform distribution
    real initial_min = config.initial_mean() - config.initial_std();
    real initial_max = config.initial_mean() + config.initial_std();
    value->uniform(initial_min, initial_max);
    VLOG(1) << config.name() << ": initial_min=" << initial_min
            << ", initial_max=" << initial_max;
  } else if (PARAMETER_INIT_NORMAL == config.initial_strategy()) {
    /* Initialize the parameters randomly */
    value->randnorm(config.initial_mean(), config.initial_std());
    VLOG(1) << config.name() << ": initial_mean=" << config.initial_mean()
            << ", initial_std=" << config.initial_std();
  } else {
    LOG(FATAL) << "not supported initial_strategy: "
               << config.initial_strategy();
  }
}

void Parameter::randomize() {
  if (!bufs_[PARAMETER_VALUE]) return;
  SetDevice device(deviceId_);
  Parameter::randomize(bufs_[PARAMETER_VALUE], config_);

  if (config_.is_sparse()) {
    if (format_ == SPARSE_CSC) {
      sparseRand(intBufs_[PARAMETER_COLS]->getData(),
                 intBufs_[PARAMETER_ROWS]->getData(),
                 config_.size(),
                 config_.dims(1) + 1,
                 config_.dims(0),
                 useGpu_);
    } else {
      sparseRand(intBufs_[PARAMETER_ROWS]->getData(),
                 intBufs_[PARAMETER_COLS]->getData(),
                 config_.size(),
                 config_.dims(0) + 1,
                 config_.dims(1),
                 useGpu_);
    }
  }
  setValueUpdated();
}

void Parameter::zeroMem() {
  if (!bufs_[PARAMETER_VALUE]) return;
  bufs_[PARAMETER_VALUE]->zeroMem();
  setValueUpdated();
  LOG(INFO) << getName() << " set to 0";
}

bool Parameter::isGradShared(size_t* blockNum) {
  if (!useGpu_ && !isStatic() && FLAGS_enable_grad_share > 0 &&
      !isGradSparseUpdate() &&
      this->getSize() > (size_t)FLAGS_enable_grad_share) {
    if (blockNum) {
      *blockNum = (size_t)FLAGS_grad_share_block_num;
    }
    return true;
  }
  return false;
}

bool Parameter::isValueShared() {
  return !useGpu_ && config_.is_shared() && FLAGS_trainer_count > 1;
}

bool Parameter::isGradSparseUpdate() const {
  return !useGpu_ && !isStatic() &&
         (config_.sparse_update() || config_.sparse_remote_update());
}

void Parameter::setMat(ParameterType pType, int matType) {
  CHECK(!mats_[pType]);

  if (config_.dims_size() == 0 && matType == MAT_NORMAL) {
    return;
  }

  CHECK_EQ((size_t)config_.dims_size(), 2LU);
  size_t height = config_.dims(0);
  size_t width = config_.dims(1);
  if (matType == MAT_NORMAL) {
    if (!config_.is_sparse()) {
      CHECK_EQ(height * width, bufs_[pType]->getSize());
      mats_[pType] =
          Matrix::create(bufs_[pType]->getMemoryHandle(), height, width);
    } else {
      size_t size = bufs_[pType]->getSize();
      CHECK_GE(height * width, size);
      if (format_ == SPARSE_CSR) {
        CHECK_EQ(height + 1, intBufs_[PARAMETER_ROWS]->getSize());
        CHECK_EQ(size, intBufs_[PARAMETER_COLS]->getSize());
      } else {
        CHECK_EQ(width + 1, intBufs_[PARAMETER_COLS]->getSize());
        CHECK_EQ(size, intBufs_[PARAMETER_ROWS]->getSize());
      }
      mats_[pType] =
          Matrix::createSparseMatrix(bufs_[pType]->getData(),
                                     intBufs_[PARAMETER_ROWS]->getData(),
                                     intBufs_[PARAMETER_COLS]->getData(),
                                     height,
                                     width,
                                     bufs_[pType]->getSize(),
                                     FLOAT_VALUE,
                                     format_,
                                     false,
                                     useGpu_);
    }
  } else if (matType == MAT_NORMAL_SHARED) {
    CHECK_EQ(height * width, bufs_[pType]->getSize());
    size_t blockNum = 0;
    CHECK(isGradShared(&blockNum));
    mats_[pType] = std::make_shared<SharedCpuMatrix>(
        blockNum,
        std::dynamic_pointer_cast<CpuMemoryHandle>(
            bufs_[pType]->getMemoryHandle()),
        height,
        width);
  } else if (matType == MAT_VALUE_SHARED) {
    CHECK_EQ(height * width, bufs_[pType]->getSize());
    mats_[pType] = std::make_shared<SharedCpuMatrix>(
        std::dynamic_pointer_cast<CpuMemoryHandle>(
            bufs_[pType]->getMemoryHandle()),
        height,
        width);
  } else if (matType == MAT_SPARSE_ROW_IDS) {
    CHECK_EQ(height * width, bufs_[pType]->getSize());
    mats_[pType] = std::make_shared<SparseRowIdsCpuMatrix>(
        std::dynamic_pointer_cast<CpuMemoryHandle>(
            bufs_[pType]->getMemoryHandle()),
        height,
        width);
  } else if (matType == MAT_SPARSE_ROW) {
    auto valueMat =
        std::dynamic_pointer_cast<SparseRowCpuMatrix>(mats_[PARAMETER_VALUE]);
    SparseRowCpuMatrix::IndexDictPtr indexDict(nullptr);
    if (pType != PARAMETER_VALUE) {
      CHECK(valueMat) << "The matrix for PARAMETER_VALUE must be set "
                      << " and its type must be MAT_SPARSE_ROW,"
                      << " MAT_SPARSE_ROW_PREFETCH or MAT_CACHE_ROW";
      indexDict = valueMat->getIndexDictHandle();
    }
    auto mat =
        std::make_shared<SparseRowCpuMatrix>(nullptr,
                                             height,
                                             width,
                                             // grad share index with value
                                             indexDict);
    mats_[pType] = mat;
  } else if (matType == MAT_CACHE_ROW) {
    CHECK(isGradSparseUpdate());
    auto mat = std::make_shared<CacheRowCpuMatrix>(height, width);
    mats_[pType] = mat;
  } else if (matType == MAT_SPARSE_ROW_PREFETCH_FULL_SIZE ||
             matType == MAT_SPARSE_ROW_PREFETCH) {
    auto mat = std::make_shared<SparsePrefetchRowCpuMatrix>(
        bufs_[pType] ? std::dynamic_pointer_cast<CpuMemoryHandle>(
                           bufs_[pType]->getMemoryHandle())
                     : nullptr,
        height,
        width,
        nullptr,  // indexDictHandle
        getGlobalSyncThreadPool());
    mats_[pType] = mat;
  } else if (matType == MAT_SPARSE_ROW_AUTO_GROW) {
    CHECK(isGradSparseUpdate());
    mats_[pType] = std::make_shared<SparseAutoGrowRowCpuMatrix>(height, width);
  } else {
    LOG(FATAL) << "Unsupported mat type" << matType;
  }
}

void Parameter::incUpdate(const UpdateCallback& callback) {
  // Static parameter is fixed, and does not need to be updated
  if (isStatic()) {
    return;
  }

  ++updateCounter_;
  if (isUpdatable()) {
    if (callback) callback(this);
    clearUpdate();
  }
}

bool Parameter::save(const std::string& filename) const {
  std::ofstream fs(filename, std::ios_base::binary);
  CHECK(fs) << "Fail to open " << filename;
  return save(fs);
}

bool Parameter::save(std::ostream& s) const {
  CpuVector vec(*bufs_[PARAMETER_VALUE].get());
  Header header;
  header.version = kFormatVersion;
  header.valueSize = sizeof(real);
  header.size = getSize();

  CHECK_EQ(header.size, vec.getSize());

  CHECK(s.write(reinterpret_cast<char*>(&header), sizeof(header)))
      << "Fail to write parameter " << getName();

  CHECK(s.write(reinterpret_cast<char*>(vec.getData()),
                header.size * sizeof(real)))
      << "Fail to write parameter " << getName();
  if (config_.is_sparse()) {
    CpuIVector rows(*intBufs_[PARAMETER_ROWS].get());
    CpuIVector cols(*intBufs_[PARAMETER_COLS].get());
    CHECK(s.write(reinterpret_cast<char*>(rows.getData()),
                  rows.getSize() * sizeof(int)))
        << "Fail to write parameter " << getName();
    CHECK(s.write(reinterpret_cast<char*>(cols.getData()),
                  cols.getSize() * sizeof(int)))
        << "Fail to write parameter " << getName();
  }

  return true;
}

/**
 * Load parameter value from a file
 */
bool Parameter::load(const std::string& filename) {
  std::ifstream fs(filename, std::ios_base::binary);
  if (!fs) {
    LOG(INFO) << "missing parameters [" << filename << "] while loading model.";
    if (kMissParameterFail == FLAGS_load_missing_parameter_strategy) {
      LOG(FATAL) << getName() << " missing, not allowed.";
      return false;
    }
    if (kMissParameterRand == FLAGS_load_missing_parameter_strategy) {
      LOG(INFO) << getName() << " missing, set to random.";
      randomize();
      return true;
    }
    if (kMissParameterZero == FLAGS_load_missing_parameter_strategy) {
      LOG(INFO) << getName() << " missing, set to zero.";
      zeroMem();
      return true;
    }
    LOG(FATAL) << "unsupported load_missing_parameter_strategy: "
               << FLAGS_load_missing_parameter_strategy;
    return false;
  }
  return load(fs);
}

bool Parameter::load(std::istream& s) {
  CpuVector vec(*bufs_[PARAMETER_VALUE].get());
  Header header;
  CHECK(s.read(reinterpret_cast<char*>(&header), sizeof(header)))
      << "Fail to read parameter " << getName();
  CHECK_EQ(header.version, kFormatVersion) << "Incorrect format version: "
                                           << header.version;
  CHECK_EQ(header.size, getSize())
      << "The size (" << header.size << ") in the file does not match the size "
      << "(" << getSize() << ") of the parameter: " << getName();
  CHECK_EQ(header.valueSize, sizeof(real))
      << "Unsupported valueSize " << header.valueSize << " at: " << getName();
  CHECK(s.read(reinterpret_cast<char*>(vec.getData()),
               header.size * sizeof(real)));

  auto& tmp = *bufs_[PARAMETER_VALUE].get();
  if (typeid(tmp) == typeid(GpuVector)) {
    bufs_[PARAMETER_VALUE]->copyFrom(vec);
  }

  if (config_.is_sparse() && config_.need_compact()) {
    // load from dense parameter with many zero
    CHECK_EQ(config_.dims_size(), 2);
    auto height = config_.dims(0);
    auto width = config_.dims(1);
    auto mat = Matrix::create(vec.getData(), height, width);
    CpuSparseMatrix sparseMat(height,
                              width,
                              0,
                              FLOAT_VALUE,
                              format_,
                              /*trans*/ false);
    sparseMat.copyFrom(*mat, HPPL_STREAM_DEFAULT);
    auto nnz = sparseMat.getElementCnt();
    size_t rowSize = (format_ == SPARSE_CSR) ? height + 1 : nnz;
    size_t colSize = (format_ == SPARSE_CSR) ? nnz : width + 1;

    intBufs_[PARAMETER_ROWS]->copyFrom(sparseMat.getRows(), rowSize);
    intBufs_[PARAMETER_COLS]->copyFrom(sparseMat.getCols(), colSize);
    bufs_[PARAMETER_VALUE]->resize(nnz);  // for setMat check
    bufs_[PARAMETER_VALUE]->copyFrom(sparseMat.getValue(), nnz);
    config_.set_size(nnz);
    LOG(INFO) << "compact nnz=" << (1. * nnz / (height * width))
              << " name=" << config_.name();
  } else if (config_.is_sparse()) {
    CpuIVector rows(*intBufs_[PARAMETER_ROWS].get());
    CpuIVector cols(*intBufs_[PARAMETER_COLS].get());
    size_t rowSize, colSize;
    CHECK_EQ(config_.dims_size(), 2);
    if (format_ == SPARSE_CSR) {
      rowSize = config_.dims(0) + 1;
      colSize = config_.size();
    } else {
      rowSize = config_.size();
      colSize = config_.dims(1) + 1;
    }
    CHECK(
        s.read(reinterpret_cast<char*>(rows.getData()), rowSize * sizeof(int)));
    CHECK(
        s.read(reinterpret_cast<char*>(cols.getData()), colSize * sizeof(int)));
    auto& paramRows = *intBufs_[PARAMETER_ROWS].get();
    if (typeid(paramRows) == typeid(GpuIVector)) {
      intBufs_[PARAMETER_ROWS]->copyFrom(rows);
    }
    auto& paramCols = *intBufs_[PARAMETER_COLS].get();
    if (typeid(paramCols) == typeid(GpuIVector)) {
      intBufs_[PARAMETER_COLS]->copyFrom(cols);
    }
  }

  setValueUpdated();

  return true;
}

}  // namespace paddle
