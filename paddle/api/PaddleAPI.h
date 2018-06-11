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

#include <stddef.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/GlobalConstants.h"

/// Import PaddlePaddle's enumeration into global namespace.
using namespace paddle::enumeration_wrapper;  // NOLINT

/**
 * @brief Initialize paddle.
 *
 * In python, this method should be invoked as
 * @code
 *  import sys
 *  import paddle
 *  paddle.initPaddle(sys.argv)
 *  or you can change arguments as any list of str.
 * @endcode
 */
void initPaddle(int argc, char** argv);

/// Return FLAGS_use_gpu
bool isUsingGpu();

/// Set the Flags_use_gpu to the given parameter
void setUseGpu(bool useGpu);

/// Return true if this py_paddle is compiled in GPU Version
bool isGpuVersion();

/// Return FLAGS_trainer_count
int getTrainerCount();

/// The Error of IO Operation. Such as file not found, etc.
class IOError {};

/// Out of range error
class RangeError {};

/// Not support Error, such as access GPU memory directly, etc.
class UnsupportError : public std::runtime_error {
 public:
  UnsupportError() : std::runtime_error(" ") {}
  explicit UnsupportError(const std::string& message)
      : std::runtime_error(message) {}
};

/// This type will map to python's list of float.
struct FloatArray {
  const float* buf;
  const size_t length;
  bool needFree;  // true if the buf is dynamic alloced.
  FloatArray(const float* b, const size_t l);
};

/// This type will map to python's list of int
struct IntArray {
  const int* buf;
  const size_t length;
  bool needFree;
  IntArray(const int* b, const size_t l, bool f = false);
};

/// This type will map to python's list of (int, float)
struct IntWithFloatArray {
  const float* valBuf;
  const int* idxBuf;
  const size_t length;
  bool needFree;
  IntWithFloatArray(const float* v, const int* i, size_t l, bool f = false);
};

enum SparseValueType { SPARSE_NON_VALUE = 0, SPARSE_VALUE = 1 };

enum SparseFormatType { SPARSE_CSR = 0, SPARSE_CSC = 1 };

/**
 * In Python, -1UL is hard to write. So define a const value used by python
 * side.
 */
const size_t NO_SPARSE_ID = -1UL;

struct MatrixPrivate;
class Matrix {
  Matrix();  // User Cannot Create Matrix.
  DISABLE_COPY(Matrix);
  static Matrix* createByPaddleMatrixPtr(void* sharedPtr);

 public:
  virtual ~Matrix();

  /**
   * Create A Matrix with height,width, which is filled by zero.
   */
  static Matrix* createZero(size_t height,
                            size_t width,
                            bool useGpu = isUsingGpu());

  /**
   * Create Sparse Matrix.
   *
   * After create sparse, sparseCopyFrom can be used to fill matrix.
   *
   * @param nnz  Number of non zero values.
   *
   * @note the default sparse type is SPARSE_CSR.
   */
  static Matrix* createSparse(size_t height,
                              size_t width,
                              size_t nnz,
                              bool isNonVal = true,
                              bool trans = false,
                              bool useGpu = isUsingGpu());

  /**
   * Create Dense Matrix.
   *
   * @param data  list of float should be passed in python.
   * @note        the value will be copy into a new matrix.
   */
  static Matrix* createDense(const std::vector<float>& data,
                             size_t height,
                             size_t width,
                             bool useGpu = isUsingGpu());

  static Matrix* createDenseFromNumpy(
      float* data,
      int dim1,
      int dim2,
      bool copy = true,
      bool useGpu = isUsingGpu()) throw(UnsupportError);

  /**
   *  Create Cpu Dense Matrix from numpy matrix, dtype=float32
   *
   *  @param data  a numpy matrix.
   *  @param dim1  dimension of data.
   *  @param dim2  dimension of data.
   *  @param copy  true if copy into a new matrix, false will create
   *               matrix inplace. copy = false should be used with extreme
   *               care because Matrix will share the memory with the given
   *               numpy array. If the numpy array object is no longer valid,
   *               the memory space will not be usable.
   */
  static Matrix* createCpuDenseFromNumpy(float* data,
                                         int dim1,
                                         int dim2,
                                         bool copy = true);

  /// Create Gpu Dense Matrix from numpy matrix, dtype=float32
  static Matrix* createGpuDenseFromNumpy(float* data, int dim1, int dim2);

  /**
   * Cast to numpy matrix.
   *
   * @note    This method take no parameter in python.
   * @note    This method in python will return a numpy matrix, not void.
   * @note    Only CpuDenseMatrix is supported.
   *
   * Example:
   * @code
   * import paddle
   * m = paddle.Matrix.createZero(10,2)
   * numpy_mat = m.toNumpyMat()
   * @endcode
   */
  void toNumpyMatInplace(float** view_data,
                         int* dim1,
                         int* dim2) throw(UnsupportError);

  /// Copy To numpy mat.
  void copyToNumpyMat(float** view_m_data,
                      int* dim1,
                      int* dim2) throw(UnsupportError);

  /// Copy From Numpy Mat
  void copyFromNumpyMat(float* data, int dim1, int dim2) throw(UnsupportError,
                                                               RangeError);

  /// return true if this matrix is sparse.
  bool isSparse() const;

  SparseValueType getSparseValueType() const throw(UnsupportError);

  SparseFormatType getSparseFormat() const throw(UnsupportError);

  IntArray getSparseRowCols(size_t i) const throw(UnsupportError, RangeError);

  IntWithFloatArray getSparseRowColsVal(size_t i) const
      throw(UnsupportError, RangeError);

  size_t getHeight() const;

  size_t getWidth() const;

  float get(size_t x, size_t y) const throw(RangeError);

  void set(size_t x, size_t y, float val) throw(RangeError, UnsupportError);

  /// return type is list of float
  FloatArray getData() const;

  /**
   * Copy from rows, cols, values.
   *
   * if sparse_nonvalue, the values should be []
   */
  void sparseCopyFrom(const std::vector<int>& rows,
                      const std::vector<int>& cols,
                      const std::vector<float>& values =
                          std::vector<float>()) throw(UnsupportError);

  bool isGpu() const;

 private:
  void* getSharedPtr() const;

  MatrixPrivate* m;
  friend class Trainer;
  friend class GradientMachine;
  friend class Arguments;
};

struct VectorPrivate;
class Vector {
  DISABLE_COPY(Vector);
  Vector();
  static Vector* createByPaddleVectorPtr(void* ptr);

  void* getSharedPtr();

 public:
  ~Vector();

  /// Create Vector filled with zero.
  static Vector* createZero(size_t sz, bool useGpu = isUsingGpu());

  /**
   * Create Vector from list of float.
   *
   * It will create a new vector, and copy data into it.
   */
  static Vector* create(const std::vector<float>& data,
                        bool useGpu = isUsingGpu());

  static Vector* createVectorFromNumpy(
      float* data,
      int dim,
      bool copy = true,
      bool useGpu = isUsingGpu()) throw(UnsupportError);
  /**
   * Create Cpu Vector from numpy array, which dtype=float32
   *
   * If copy is false, it will create vector inplace.
   */
  static Vector* createCpuVectorFromNumpy(float* data,
                                          int dim,
                                          bool copy = true);

  /// Create Gpu Vector from numpy array, which dtype=float32
  static Vector* createGpuVectorFromNumpy(float* data, int dim);

  /**
   * copy from another vector
   * throw(RangeError) if size of src vector is different from size of this
   * vector
   */
  void copyFrom(Vector* src) throw(RangeError);

  /// Cast to numpy array inplace.
  void toNumpyArrayInplace(float** view_data, int* dim1) throw(UnsupportError);

  /// Copy to numpy array.
  void copyToNumpyArray(float** view_m_data, int* dim1);

  /// Copy from numpy array.
  void copyFromNumpyArray(float* data, int dim);

  /// __getitem__ in python
  float get(const size_t idx) const throw(RangeError, UnsupportError);

  /// __setitem__ in python
  void set(const size_t idx, float val) throw(RangeError, UnsupportError);

  /// Return is GPU vector or not.
  bool isGpu() const;

  /// Return a list of float, the memory is alloced and copied.
  FloatArray getData() const;

  /// __len__ in python
  size_t getSize() const;

 private:
  VectorPrivate* m;

 private:
  friend class Parameter;
  friend class ParameterOptimizer;
  friend struct ParameterTraverseCallbackPrivate;
};

struct IVectorPrivate;
class IVector {
  IVector();
  DISABLE_COPY(IVector);
  static IVector* createByPaddleVectorPtr(void* ptr);

 public:
  /// Create IVector filled with zero
  static IVector* createZero(size_t sz, bool useGpu = isUsingGpu());

  /**
   * Create IVector from list of int.
   * It will create a new vector, and copy data into it.
   */
  static IVector* create(const std::vector<int>& data,
                         bool useGpu = isUsingGpu());

  static IVector* createVectorFromNumpy(
      int* data,
      int dim,
      bool copy = true,
      bool useGpu = isUsingGpu()) throw(UnsupportError);

  /**
   * Create Cpu IVector from numpy array, which dtype=int32
   *
   * If copy is false, it will create vector inplace
   */
  static IVector* createCpuVectorFromNumpy(int* data,
                                           int dim,
                                           bool copy = true);
  /**
   * Create Gpu IVector from numpy array, which dtype=int32
   */
  static IVector* createGpuVectorFromNumpy(int* data, int dim);

  /// Cast to numpy array inplace.
  void toNumpyArrayInplace(int** view_data, int* dim1) throw(UnsupportError);

  /// Copy to numpy array.
  void copyToNumpyArray(int** view_m_data, int* dim1);

  /// Copy from numpy array.
  void copyFromNumpyArray(int* data, int dim);

  virtual ~IVector();

  /// Return a list of int, the memory is alloced and copied.
  IntArray getData() const;

  /// This method will map to python [] method.
  int& operator[](const size_t idx) throw(RangeError, UnsupportError);

  const int& operator[](const size_t idx) const
      throw(RangeError, UnsupportError);

  inline int get(const size_t idx) const throw(RangeError, UnsupportError) {
    return (*this)[idx];
  }

  inline void set(const size_t idx, int val) throw(RangeError, UnsupportError) {
    (*this)[idx] = val;
  }

  /// Return true if it is gpu vector.
  bool isGpu() const;

  /// This method will map to python __len__();
  size_t getSize() const;

 private:
  void* getSharedPtr() const;

  friend class Arguments;
  IVectorPrivate* m;
};

struct ArgumentsPrivate;

/// The Arguments is actual a std::vector<paddle::Argument> in paddle.
class Arguments {
 private:
  Arguments();  // Internal Create.
  DISABLE_COPY(Arguments);

 public:
  /**
   * Create a arguments with size.
   * Note that it can be zero.
   */
  static Arguments* createArguments(size_t slotNum);

  void resize(size_t slotNum);

  virtual ~Arguments();

  /**
   * Return the slot number that aguments contains.
   *
   * It is actually the vector's size
   */
  size_t getSlotNum() const;

  /**
   * The get functions of Arguments
   *
   * the param idx is the slot id
   */
  Matrix* getSlotValue(size_t idx) const throw(RangeError);
  Matrix* getSlotGrad(size_t idx) const throw(RangeError);
  IVector* getSlotIds(size_t idx) const throw(RangeError);
  Matrix* getSlotIn(size_t idx) const throw(RangeError);
  IVector* getSlotSequenceStartPositions(size_t idx) const throw(RangeError);
  IVector* getSlotSubSequenceStartPositions(size_t idx) const throw(RangeError);
  IVector* getSlotSequenceDim(size_t idx) const throw(RangeError);
  // End Of get functions of Arguments

  int64_t getBatchSize(size_t idx = 0) const throw(RangeError);

  /**
   * The set functions of Arguments.
   *
   * The param idx is the slot id.
   * The other param is the input Matrix or vector.
   */
  void setSlotValue(size_t idx, Matrix* mat) throw(RangeError);
  void setSlotGrad(size_t idx, Matrix* mat) throw(RangeError);
  void setSlotIn(size_t idx, Matrix* mat) throw(RangeError);
  void setSlotIds(size_t idx, IVector* vec) throw(RangeError);
  void setSlotSequenceStartPositions(size_t idx,
                                     IVector* vec) throw(RangeError);
  void setSlotSubSequenceStartPositions(size_t idx,
                                        IVector* vec) throw(RangeError);
  void setSlotSequenceDim(size_t idx, IVector* vec) throw(RangeError);

  /**
   * Set the frame height of the idx-th Argument.
   *
   * @param ids The index of which Argument.
   * @param h The height value.
   */
  void setSlotFrameHeight(size_t idx, size_t h) throw(RangeError);

  /**
   * Set the frame height of the idx-th Argument.
   *
   * @param ids The index of which Argument.
   * @param h The height value.
   */
  void setSlotFrameWidth(size_t idx, size_t w) throw(RangeError);

  size_t getSlotFrameHeight(size_t idx = 0) const throw(RangeError);
  size_t getSlotFrameWidth(size_t idx = 0) const throw(RangeError);

  float sum() const;

 private:
  static Arguments* createByPaddleArgumentVector(void* ptr);
  static Arguments* createByPaddleArgument(const void* ptr);
  void* getInternalArgumentsPtr() const;

 private:
  ArgumentsPrivate* m;
  friend class Trainer;
  friend class GradientMachine;
  friend class SequenceGenerator;
};

enum GradientMatchineCreateMode {
  CREATE_MODE_NORMAL = paddle::GradientMachine::kNormal,
  CREATE_MODE_SGD_SPARSE_CPU_TRAINING =
      paddle::GradientMachine::kSgdSparseCpuTraining,
  CREATE_MODE_TESTING = paddle::GradientMachine::kTesting
};

struct ParameterConfigPrivate;
class ParameterConfig {
  DISABLE_COPY(ParameterConfig);
  ParameterConfig();

  /**
   * Internal methods
   */
  static ParameterConfig* createParameterConfigFromParameterSharedPtr(
      void* ptr);
  static ParameterConfig* createParameterConfigFromParameterPtr(void* ptr);
  void* getRawPtr();

 public:
  ~ParameterConfig();

  /**
   * return proto buf string.
   */
  std::string toProtoString() const;

 private:
  ParameterConfigPrivate* m;

 private:
  friend class Parameter;
  friend class ParameterOptimizer;
  friend struct ParameterTraverseCallbackPrivate;
};

struct OptimizationConfigPrivate;
class OptimizationConfig {
  DISABLE_COPY(OptimizationConfig);
  OptimizationConfig();

 public:
  static OptimizationConfig* createFromProtoString(const std::string& str);
  ~OptimizationConfig();

  /**
   * return protobuf string.
   */
  std::string toProtoString();

 private:
  OptimizationConfigPrivate* m;

  friend class TrainerConfig;
  friend class ParameterOptimizer;
  friend class ParameterUpdater;
  friend class Trainer;
};

struct ParameterPrivate;
class Parameter {
 private:
  Parameter();
  DISABLE_COPY(Parameter);

 public:
  virtual ~Parameter();

  /**
   * get parameter name
   */
  std::string getName() const;

  /**
   * get buf in Parameter
   */
  Vector* getBuf(ParameterType type);

  /**
   * get id
   */
  size_t getID() const;

  ParameterConfig* getConfig();
  void setValueUpdated();

  bool save(const std::string& filename) const;

  bool load(const std::string& filename) const;

  size_t getSize() const;

 private:
  static Parameter* createFromRawPtr(void* ptr);
  static Parameter* createFromSharedPtr(void* ptr);

 private:
  ParameterPrivate* m;
  friend class UpdateCallbackWrapper;
  friend class GradientMachine;
  friend class ParameterUpdater;
};

struct ModelConfigPrivate;
/**
 * You can only get model config from TrainerConfig.
 *
 * It is used by GradientMachine.
 */
class ModelConfig {
 private:
  ModelConfig();
  DISABLE_COPY(ModelConfig);

 public:
  virtual ~ModelConfig();

 private:
  ModelConfigPrivate* m;
  friend class TrainerConfig;
  friend struct TrainerConfigPrivate;
  friend class GradientMachine;
};

struct TrainerConfigPrivate;
/**
 * To get TrainerConfig from file.
 *
 * It is used by GradientMachine.
 */
class TrainerConfig {
 private:
  TrainerConfig();
  DISABLE_COPY(TrainerConfig);

 public:
  virtual ~TrainerConfig();

  static TrainerConfig* createFromTrainerConfigFile(
      const std::string& configPath);
  static TrainerConfig* createFromProtoString(const std::string& str);

  ModelConfig* getModelConfig() const;

  OptimizationConfig* getOptimizationConfig() const;

 private:
  TrainerConfigPrivate* m;
  friend class Trainer;
};

/**
 * The callback in backword.
 *
 * You can inherit this class in python.
 *
 * @code
 * class UpdateCallbackInPython(paddle.UpdateCallback):
 *   def __init__(self):
 *     paddle.UpdateCallback.__init__(self)
 *
 *   def apply(self, param):
 *     assert isinstance(param, paddle.Parameter)
 * @endcode
 */
class UpdateCallback {
 public:
  virtual ~UpdateCallback();
  virtual void apply(Parameter* p);
};

struct ParameterTraverseCallbackPrivate;
class ParameterTraverseCallback {
  DISABLE_COPY(ParameterTraverseCallback);
  ParameterTraverseCallback();

 public:
  ~ParameterTraverseCallback();

  void apply(const std::vector<Vector*>& vecs,
             const ParameterConfig& config,
             size_t sparseId);

 private:
  ParameterTraverseCallbackPrivate* m;
  friend class ParameterOptimizer;
};

/**
 * The ParameterOptimizer Wrapper Class.
 *
 * Basically same as common/ParameterOptimizer.h
 */
struct ParameterOptimizerPrivate;
class ParameterOptimizer {
  DISABLE_COPY(ParameterOptimizer);
  ParameterOptimizer();

 public:
  static ParameterOptimizer* create(OptimizationConfig* config);

  ~ParameterOptimizer();

  void init(size_t numRows, const ParameterConfig* config);

  void startPass();

  void finishPass();

  void startBatch(size_t numSamplesProcessed);

  void finishBatch();

  void update(const std::vector<Vector*>& vecs,
              const ParameterConfig& conf,
              size_t sparseId = NO_SPARSE_ID);

  std::vector<int> getParameterTypes() const;

  ParameterTraverseCallback* needSpecialTraversal(
      const ParameterConfig& config) const;

 private:
  ParameterOptimizerPrivate* m;
};

class SequenceGenerator;
class Evaluator;
struct GradientMachinePrivate;
class GradientMachine {
 private:
  GradientMachine();
  DISABLE_COPY(GradientMachine);

 public:
  virtual ~GradientMachine();

  /**
   * Create By ProtoStr.
   *
   * The ProtoStr can be generate by python's protobuf code.
   */
  static GradientMachine* createByConfigProtoStr(
      const std::string& protoStr,
      GradientMatchineCreateMode mode = CREATE_MODE_NORMAL,
      const std::vector<int>& parameterTypes = defaultParamTypes);

  /**
   * Create by ModelConfig object.
   *
   * To get ModelConfig, you can get TrainerConfig from config file, then get
   * model config by TrainerConfig
   */
  static GradientMachine* createByModelConfig(
      ModelConfig* conf,
      GradientMatchineCreateMode mode = CREATE_MODE_NORMAL,
      const std::vector<int>& parameterTypes = defaultParamTypes);

  /**
   * @brief finish
   */
  void finish();

  void start();

  /**
   * Prefetch row ids of sparse parameter.
   */
  void prefetch(const Arguments& inArgs);

  /**
   * Do some thing when train pass ended.
   */
  void onPassEnd();

  /**
   * The forward stage of GradientMachine.
   *
   * @note  the outArgs could be zero length arguemnts.
   * @note  THIS METHOD IS VERY USEFULL FOR PREDICT FROM TRAINED MODEL.
   */
  void forward(const Arguments& inArgs, Arguments* outArgs, PassType passType);

  /**
   * The backward stage of GradientMachine.
   *
   * @note  Currently the ParameterUpdater is not wrapped in SWIG, so backward
   * cannot actually train a network. But you can write a update callback to
   * change the parameter or implement a ParameterUpdater in python side.
   */
  void backward(const UpdateCallback& callback = UpdateCallback());

  /**
   * Combine forward/backward
   */
  void forwardBackward(const Arguments& inArgs,
                       Arguments* outArgs,
                       PassType passType,
                       const UpdateCallback& callback = UpdateCallback());

  void loadParameters(const std::string& path);

  size_t getParameterSize() const;
  Parameter* getParameter(size_t i) throw(RangeError);

  size_t getNonStaticParameterSize() const;
  Parameter* getNonStaticParameter(size_t i) throw(RangeError);

  void randParameters();

  Arguments* getLayerOutput(const std::string& layerName) const
      throw(UnsupportError);

  /**
   * Create a sequence generator.
   *
   * @note  It just like a paddle_gen_sequence.
   */
  SequenceGenerator* asSequenceGenerator(
      const std::vector<std::string>& dict = std::vector<std::string>(),
      size_t begin_id = 0UL,
      size_t end_id = 0UL,
      size_t max_length = 100UL,
      size_t beam_size = -1UL);

  Evaluator* makeEvaluator();

  void eval(Evaluator* evaluator);

 private:
  GradientMachinePrivate* m;

  static GradientMachine* createFromPaddleModelPtr(
      const void* confPtr,
      GradientMatchineCreateMode mode,
      const std::vector<int>& types);

  // Not to use c++ 11 init-list, so we use static var as function default arg.
  static std::vector<int> defaultParamTypes;
  friend class Trainer;
  friend class ParameterUpdater;
};

struct ParameterUpdaterPrivate;
class ParameterUpdater {
 private:
  ParameterUpdater();

 public:
  static ParameterUpdater* createLocalUpdater(OptimizationConfig* config);
  static ParameterUpdater* createRemoteUpdater(OptimizationConfig* config,
                                               int passCount,
                                               bool useSparseUpdater);
  static ParameterUpdater* createNewRemoteUpdater(
      OptimizationConfig* config,
      const std::string pserverSpec,
      const bool useEtcd) throw(UnsupportError);
  ~ParameterUpdater();

  /**
   * @brief initialize Parameter Updater by GradientMachine.
   * @param gm
   */
  void init(const GradientMachine& gm);

  /**
   * @brief begin of a training/testing of one pass.
   */
  void startPass();

  /**
   * @brief end of a traning/testing of one pass.
   */
  void finishPass();

  /**
   * @brief begin of a training/testing of one batch.
   * @param data batch's size
   * @return PassType, mostly will be training.
   */
  PassType startBatch(size_t batchSize);

  /**
   * @brief end of a traning/testing of one batch
   * @param cost current batch cost.
   */
  void finishBatch(float cost);

  /**
   * @brief update a parameter (by local optimizer or by cluster pserver)
   * @param param
   */
  void update(Parameter* param);

  /**
   * @breif only get required sparse rows by default.
   * @param fullSize: get full matrix parameter if *fullSize* set
   * @param apply: get PARAMETER_APPLY on pserver if *apply* set
   */
  void getParametersRemote(bool fullSize = false, bool apply = false);

  /**
   * @brief restore the average parameter.
   * @note It is only used in AverageOptimizer. Restore will get the current
   * PARAMETER_VALUE back.
   */
  void restore();

  /**
   * @brief apply. Store the average parameter.
   * @note It is only used in AverageOptimizer. Apply will store the current
   * PARAMETER_VALUE to buffer, calcaualte current Average Parameter, and save
   * it to PARAMETER_VALUE.
   */
  void apply();

  /**
   * @brief catchUpWith The Regularization will be delayed in many situations(
   * pserver, local sparse). Catch Up means catch the regularization up, apply
   * regularization to all params.
   */
  void catchUpWith();

 private:
  ParameterUpdaterPrivate* m;
};

struct EvaluatorPrivate;
class Evaluator {
 private:
  Evaluator();
  DISABLE_COPY(Evaluator);

 public:
  ~Evaluator();

  /**
   * @brief begin an evaluate stage.
   */
  void start();

  /**
   * @brief end an evaluate stage.
   */
  void finish();

  /**
   * @brief toString will get a evaluate result.
   *
   * __repr__ method in python
   */
  std::string toString();

  std::vector<std::string> getNames() const;

  double getValue(const std::string name) const;

 private:
  EvaluatorPrivate* m;

  friend class GradientMachine;
};

struct TrainerPrivate;
class Trainer {
 private:
  TrainerPrivate* m;
  Trainer();
  Trainer(TrainerConfig* optConfig, GradientMachine* gm);
  DISABLE_COPY(Trainer);

 public:
  virtual ~Trainer();

  /// Create A Trainer By TrainerConfig. using paddle command line.
  static Trainer* createByCommandLine() throw(IOError);

  static Trainer* create(TrainerConfig* optConfig,
                         GradientMachine* gm) throw(IOError);

  /// Start training
  void startTrain();

  /// Finish training
  void finishTrain();

  /// Start a pass.
  void startTrainPass();

  /// Finish a pass
  void finishTrainPass();

  /**
   * Train one batch,
   *
   * @return true if all batch finished.
   */
  bool trainOneBatch(size_t batchSize);

  void trainOneDataBatch(size_t batchSize, const Arguments& args);

  void startTestPeriod();
  void testOneDataBatch(size_t batchSize, const Arguments& args);
  void finishTestPeriod();

  void forwardOneBatch(size_t batchSize);

  Arguments* getForwardOutput();

  Arguments* getLayerOutput(const std::string& layerName) const;
};

/// the N-Best results generated from one input sequence.
class ISequenceResults {
 public:
  virtual ~ISequenceResults();

  /// Number of result.
  virtual size_t getSize() const = 0;

  /**
   * Get sentence from dictionary.
   *
   * @param id  the index of result.
   * @param split  if true, the return sentence will be splited with ' ' by
   *               each word. Default is false.
   */
  virtual std::string getSentence(size_t id, bool split = false) const
      throw(RangeError) = 0;
  virtual std::vector<int> getSequence(size_t id) const throw(RangeError) = 0;
  virtual float getScore(size_t id) const throw(RangeError) = 0;
};

struct SequenceGeneratorPrivate;
class SequenceGenerator {
  DISABLE_COPY(SequenceGenerator);
  SequenceGenerator();

 public:
  virtual ~SequenceGenerator();

  /**
   * Generate Sequence by input.
   *
   * @note  The inArgs is just one sequence of data.
   * @note  The return will get a N-best generate result by inArgs.
   *        Sort by score.
   */
  ISequenceResults* generateSequence(const Arguments& inArgs) const;

  void setDict(const std::vector<std::string>& dict);
  void setBos(size_t bos);
  void setEos(size_t eos);
  void setMaxLength(size_t maxlength);
  void setBeamSize(size_t beamSize);

 private:
  static SequenceGenerator* createByGradientMachineSharedPtr(void* ptr);
  friend class GradientMachine;

 private:
  SequenceGeneratorPrivate* m;
};
