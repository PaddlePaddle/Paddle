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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <utility>
#include <vector>
#include "boost/crc.hpp"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {

namespace framework {

class LoDTensor;

class Tensor;

template <class U>
struct TensorOutStreamer {
  typedef TensorOutStreamer<U> self;
  const char* name;
  const Tensor& tensor;
  size_t limit;
  TensorOutStreamer(const char* name, const Tensor& _tensor);
  self& setLimit(size_t _limit);
  const U* begin() const;
  const U* end() const;
  int checksum() const;
};

template <int version = 0>
class TensorDumpConfig {
 protected:
  std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  template <class F>
  void env_exe(const char* name, F f) {
    auto* env = std::getenv(name);
    if (env) {
      f(env);
    }
  }

  size_t limit_1;
  size_t limit_4;
  std::string filename;
  std::vector<std::string> ops;
  bool synchronized;
  TensorDumpConfig()
      : limit_1(128),
        limit_4(128),
        filename("/dev/stdout"),
        synchronized(false) {
    env_exe("TENSOR_DUMP_OPERATORS", [this](const char* value) {
      if (strlen(value)) {
        ops = split(value, ',');
      }
    });

    env_exe("TENSOR_DUMP_FILE",
            [this](const char* value) { filename = value; });

    env_exe("TENSOR_DUMP_LIMIT_SIZEOF_1", [this](const char* value) {
      std::stringstream ss;
      ss << value;
      ss >> limit_1;
    });

    env_exe("TENSOR_DUMP_LIMIT_SIZEOF_4", [this](const char* value) {
      std::stringstream ss;
      ss << value;
      ss >> limit_4;
    });

    env_exe("TENSOR_DUMP_SYNCHRONIZE",
            [this](const char* value) { synchronized = true; });
  }

 public:
  bool is_disabled() const { return !ops.size(); }
  bool hasOperator(const char* name) {
    return std::find_if(ops.begin(), ops.end(),
                        [name](const std::string& item) {
                          return item == name;
                        }) != ops.end();
  }

  bool is_synchronized() const { return synchronized; }
  const std::string& getFilename() { return filename; }
  std::ofstream& getOutputStream() {
    static std::unique_ptr<std::ofstream> ptr(
        new std::ofstream(filename.c_str()));
    return *ptr;
  }
  size_t getLimit_1() { return limit_1; }
  size_t getLimit_4() { return limit_4; }
  size_t getLimitViaSize(int size) { return (size == 1) ? limit_1 : limit_4; }
  static TensorDumpConfig<version>& get() {
    static TensorDumpConfig<version> inst;
    return inst;
  }
  static size_t NextRecord() {
    static size_t seq = 0;
    return seq++;
  }

  static std::mutex& getMutex() {
    static std::mutex mx;
    return mx;
  }
};

class Tensor {
#ifdef PADDLE_WITH_MKLDNN

 public:
  inline mkldnn::memory::format_tag format() const { return format_; }

  inline void set_format(const mkldnn::memory::format_tag format) {
    format_ = format;
  }

 protected:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */

  mkldnn::memory::format_tag format_ = mkldnn::memory::format_tag::undef;
#endif

 public:
  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenMatrix;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor() : type_(proto::VarType::FP32), offset_(0) {}

  explicit Tensor(const proto::VarType::Type&);

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  T* data();

  /*! Return a pointer to constant memory block. */
  template <typename T>
  const T* data() const;

  template <class T>
  bool hasType() const {
    return type_ == ::paddle::framework::DataTypeTrait<T>::DataType();
  }

  /*! Serialize tensor to file with label name */
  int dump(const char* name);

  inline bool IsInitialized() const;

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(const platform::Place& place, size_t requested_size = 0);

  void* mutable_data(const platform::Place& place, proto::VarType::Type type,
                     size_t requested_size = 0);

  void* mutable_data(const platform::Place& place, size_t requested_size = 0);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims           The dimensions of the memory block.
   * @param[in] place          The place of the memory block.
   * @param[in] requested_size The size of the block in bytes.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(const DDim& dims, const platform::Place& place,
                  size_t requested_size = 0);

  /*! Return the dimensions of the memory block. */
  const DDim& dims() const;

  /*! Return the numel of the memory block. */
  int64_t numel() const;

  /*! Resize the dimensions of the memory block. */
  Tensor& Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  Tensor& ShareDataWith(const Tensor& src);

  /**
   * @brief  Return a sub-tensor of the given tensor.
   *
   * @param[in] begin_idx   The index of the start row(inclusive) to slice.
   *                        The index number begins from 0.
   * @param[in] end_idx     The index of the end row(exclusive) to slice.
   *                        The index number begins from 0.
   */
  Tensor Slice(int64_t begin_idx, int64_t end_idx) const;

  const platform::Place& place() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::place() is called.");
    return holder_->place();
  }

  proto::VarType::Type type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::type() is called.");
    return type_;
  }

  // memory size returns the holding memory size in byte.
  size_t memory_size() const;

  void check_memory_size() const;

  DataLayout layout() const { return layout_; }

  void set_layout(const DataLayout layout) { layout_ = layout; }

  void clear() {
    holder_ = nullptr;
    offset_ = 0;
  }

  void ShareBufferWith(const Tensor& tensor) {
    holder_ = tensor.holder_;
    offset_ = tensor.offset_;
  }

  bool IsSharedBufferWith(const Tensor& src) const {
    return holder_ && holder_ == src.Holder();
  }

  const std::shared_ptr<memory::Allocation>& Holder() const { return holder_; }
  size_t offset() const { return offset_; }

  std::shared_ptr<memory::Allocation> MoveMemoryHolder() {
    return std::move(holder_);
  }

  void ResetHolder(std::shared_ptr<memory::Allocation> holder);

  void ResetHolderWithType(std::shared_ptr<memory::Allocation> holder,
                           const proto::VarType::Type type);

 private:
  /*! holds the memory block if allocated. */
  std::shared_ptr<memory::Allocation> holder_;
  proto::VarType::Type type_;
  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /**
   * @brief the layout of memory block, default is NHWC.
   *
   * @note the memory allocation order, describe how weight/data is stored
   *       For example, in 4-D Tensor(rank=4), there are three commonly
   *       used layout. They are
   *            NCHW, NHWC, CHWN.
   *       N,C,H,W for respectively the batch size, the number of
   *       feature maps, the height.
   */
  // Fix me: here just change the default layout to kNCHW
  // it doesn't fix the real issue, i.e. feeder should set up tensor layout
  // according to actual input data
  DataLayout layout_ = DataLayout::kNCHW;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really begins.
   */
  size_t offset_;
};

template <class I, class F>
void for_each_no_more(I b, I e, std::size_t count, F f) {
  if (count) {
    for (; count && b != e; ++b, --count) {
      f(*b);
    }
  } else {
    for (; b != e; ++b) {
      f(*b);
    }
  }
}

template <class U>
TensorOutStreamer<U>::TensorOutStreamer(const char* _name,
                                        const Tensor& _tensor)
    : name(_name), tensor(_tensor), limit(0) {}

template <class U>
TensorOutStreamer<U>& TensorOutStreamer<U>::setLimit(size_t _limit) {
  limit = _limit;
  return *this;
}

template <class U>
const U* TensorOutStreamer<U>::begin() const {
  return tensor.data<U>();
}

template <class U>
const U* TensorOutStreamer<U>::end() const {
  return begin() + tensor.memory_size() / sizeof(U);
}

template <class U>
int TensorOutStreamer<U>::checksum() const {
  boost::crc_32_type result;
  result.process_bytes(reinterpret_cast<const unsigned char*>(begin()),
                       tensor.memory_size());
  return result.checksum();
}

template <class U>
struct type_desc;

template <>
struct type_desc<signed char> {
  enum { pad = 2, break_line = 256 };
  static const char* name() { return "signed_char"; }
  static std::ostream& format(std::ostream& out, signed char v) {
    out << std::setw(pad) << std::setfill('0') << std::hex
        << (static_cast<int>(v) & 0xFF);
    return out;
  }
};

template <>
struct type_desc<unsigned char> {
  enum { pad = 2, break_line = 256 };
  static const char* name() { return "unsigned_char"; }
  static std::ostream& format(std::ostream& out, unsigned char v) {
    out << std::setw(pad) << std::setfill('0') << std::hex
        << (static_cast<int>(v) & 0xFF);
    return out;
  }
};

template <>
struct type_desc<float> {
  enum { pad = 8, break_line = 32 };
  static const char* name() { return "float"; }
  static std::ostream& format(std::ostream& out, float v) {
    out << std::setw(pad) << std::setfill(' ') << v << " ";
    return out;
  }
};

template <class U>
std::ostream& operator<<(std::ostream& out, const TensorOutStreamer<U>& ts) {
  auto& tensor = ts.tensor;
  auto& dims = tensor.dims();
  auto& conf = TensorDumpConfig<>::get();
  out << std::setw(8) << std::setfill(' ') << (TensorDumpConfig<>::NextRecord())
      << ") type=[" << type_desc<U>::name() << "]  => " << ts.name
      << " crc32=" << std::hex << ts.checksum() << std::dec
      << "  elem=" << tensor.numel() << "  "
      << " dims=" << dims.size() << "=>";

  for (decltype(dims.size()) i = 0; i < dims.size(); ++i) {
    out << "[" << dims.at(i) << "]";
  }

  out << std::endl;
  std::size_t br = 0;

  for_each_no_more(ts.begin(), ts.end(), conf.getLimitViaSize(sizeof(U)),
                   [&out, &br](U unit) {
                     if (type_desc<U>::break_line == br++) {
                       br = 0;
                       out << std::endl;
                     }
                     type_desc<U>::format(out, unit) << std::dec;
                   });
  out << std::endl;
  return out;
}

template <class T>
struct TupleExtractor;

template <class First, class... Tail>
struct TupleExtractor<std::tuple<First, Tail...>> {
  typedef First head;
  typedef std::tuple<Tail...> tail;
};

template <>
struct TupleExtractor<std::tuple<>> {};

template <class First, class... Tail>
struct DumpComposit {
  typedef std::tuple<First, Tail...> args_tuple;
  enum { size = std::tuple_size<args_tuple>::value };
  typedef typename std::tuple_element<size - 1, args_tuple>::type last;
  static_assert(std::is_same<last, float>::value,
                "Last one must be float type!");
  static_assert(std::is_arithmetic<First>::value,
                "Incorrect type - is not an arithmetic type!");

  static void execute(const char* name, const Tensor& _tensor) {
    if (_tensor.hasType<First>()) {
      if (TensorDumpConfig<>::get().is_synchronized()) {
        /* in case of parallel executor , io must be synchronized */
        std::lock_guard<decltype(TensorDumpConfig<>::getMutex())> l(
            TensorDumpConfig<>::getMutex());
        TensorDumpConfig<>::get().getOutputStream()
            << TensorOutStreamer<First>(name, _tensor);
      } else {
        TensorDumpConfig<>::get().getOutputStream()
            << TensorOutStreamer<First>(name, _tensor);
      }
    } else {
      DumpComposit<Tail...>::execute(name, _tensor);
    }
  }
};

template <>
struct DumpComposit<float> {
  static void execute(const char* name, const Tensor& _tensor) {
    if (TensorDumpConfig<>::get().is_synchronized()) {
      /* in case of parallel executor , io must be synchronized */
      std::lock_guard<decltype(TensorDumpConfig<>::getMutex())> l(
          TensorDumpConfig<>::getMutex());
      TensorDumpConfig<>::get().getOutputStream()
          << TensorOutStreamer<float>(name, _tensor);
    } else {
      TensorDumpConfig<>::get().getOutputStream()
          << TensorOutStreamer<float>(name, _tensor);
    }
  }
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/tensor_impl.h"
