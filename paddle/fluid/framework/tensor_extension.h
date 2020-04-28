// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cstdint>
#include <cstring>
#include <dnnl.hpp>
#include "boost/crc.hpp"

namespace paddle {
namespace framework {

// ########################

namespace colors {

constexpr const char* RESET() { return "\033[0m"; }

constexpr const char* RED() { return "\033[0;31m"; }

constexpr const char* GREEN() { return "\033[1;32m"; }

constexpr const char* CYAN() { return "\033[0;36m"; }

#define msg_color(c, x) \
  std::cout << c << "[tensor-dump]: " << x << colors::RESET() << std::endl;
#define msg_green(x) msg_color(colors::GREEN(), x)
}  // namespace colors

template <DataLayout LayoutType>
struct ConvPaddleLayout2Mkldnn;

template <>
struct ConvPaddleLayout2Mkldnn<DataLayout::kNCHW> {
  static const dnnl::memory::format_tag value = dnnl::memory::format_tag::nchw;
};

template <>
struct ConvPaddleLayout2Mkldnn<DataLayout::kNHWC> {
  static const dnnl::memory::format_tag value = dnnl::memory::format_tag::nhwc;
};

template <class L>
dnnl::memory::format_tag tensor_layout2_mkldnn(L layout) {
  using tag = dnnl::memory::format_tag;

  if (layout == DataLayout::kNHWC) return tag::nhwc;

  return tag::nchw;
}

template <class T>
struct type2mkldnntype;

template <>
struct type2mkldnntype<float> {
  static const dnnl::memory::data_type value = dnnl::memory::data_type::f32;
};

template <>
struct type2mkldnntype<uint8_t> {
  static const dnnl::memory::data_type value = dnnl::memory::data_type::u8;
};

template <>
struct type2mkldnntype<int8_t> {
  static const dnnl::memory::data_type value = dnnl::memory::data_type::s8;
};

template <class T, class TensorType>
void reorder_via_mkldnn(void* out, TensorType* tensor_obj, DataLayout layout) {
  dnnl::engine::kind engine_kind = dnnl::engine::kind::cpu;
  dnnl::engine engine(engine_kind, 0);
  // Create dnnl::stream.
  dnnl::stream engine_stream(engine);
  dnnl::memory::dims src_dims;
  auto& paddle_dims = tensor_obj->dims();

  for (decltype(paddle_dims.size()) i = 0; i < paddle_dims.size(); ++i)
    src_dims.push_back(paddle_dims.at(i));
  auto current_layout = tensor_obj->layout();
  /*
  std::cout << "Reorder from=" << DataLayoutToString(current_layout)
            << " to=" << DataLayoutToString(layout) << std::endl;
  */
  auto src_md = dnnl::memory::desc(src_dims, type2mkldnntype<T>::value,
                                   tensor_layout2_mkldnn(current_layout));
  auto dst_md = dnnl::memory::desc(src_dims, type2mkldnntype<T>::value,
                                   tensor_layout2_mkldnn(layout));

  auto src_mem = dnnl::memory(src_md, engine, tensor_obj->raw_data());
  auto dst_mem = dnnl::memory(dst_md, engine, out);

  // Create primitive descriptor.
  auto reorder_pd =
      dnnl::reorder::primitive_desc(engine, src_md, engine, dst_md);
  // Create the primitive.
  auto reorder_prim = dnnl::reorder(reorder_pd);
  // Primitive arguments.
  std::unordered_map<int, dnnl::memory> reorder_args;
  reorder_args.insert({DNNL_ARG_SRC, src_mem});
  reorder_args.insert({DNNL_ARG_DST, dst_mem});
  // Primitive execution: reorder with scaled sum.
  reorder_prim.execute(engine_stream, reorder_args);
  // Wait for the computation to finalize.
  engine_stream.wait();
}

template <class U, class TensorType>
struct TensorOutStreamer {
  typedef TensorOutStreamer<U, TensorType> self;
  const char* label;
  const char* name;
  const TensorType& tensor;
  size_t limit;
  DataLayout layout;
  TensorOutStreamer(const char* label, const char* name,
                    const TensorType& _tensor, DataLayout layout);
  self& setLimit(size_t _limit);
  const U* begin() const;
  const U* end() const;
  int checksum() const;
};

template <int version = 0>
class TensorDumpConfig {
 public:
  static DataLayout getDefaultDataLayout() { return DataLayout::kAnyLayout; }

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

  struct OperatorDetails {
    const std::string name;
    DataLayout layout;
    OperatorDetails(const std::string& _name,
                    DataLayout _layout = getDefaultDataLayout())
        : name{_name}, layout{_layout} {}
    const std::string getLayoutString() { return DataLayoutToString(layout); }
  };

  // std::vector<std::string> ops;
  std::vector<OperatorDetails> ops;

  bool synchronized;
  TensorDumpConfig()
      : limit_1(128),
        limit_4(128),
        filename("/dev/stdout"),
        synchronized(false) {
    env_exe("TENSOR_DUMP_OPERATORS", [this](const char* value) {
      auto tmp_ops = split(value, ',');
      auto it = std::back_inserter(ops);
      for (auto& _op : tmp_ops) {
        auto key_value = split(_op, '=');
        auto size = key_value.size();
        if (!size || size > 2) {
          throw std::runtime_error("incorrect key=value");
        }
        if (size == 1) {
          *it++ = {key_value[0]};
          continue;
        }
        *it++ = {key_value[0], StringToDataLayout(key_value[1])};
      }

      for (auto& item : ops) {
        msg_green("config for operator " << item.name << ","
                                         << item.getLayoutString());
      }
    });

    env_exe("TENSOR_DUMP_FILE", [this](const char* value) {
      filename = value;
      msg_green("output file " << filename);
    });

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
                        [name](const OperatorDetails& item) {
                          return item.name == name;
                        }) != ops.end();
  }

  auto fetchOperator(const char* name) -> decltype(ops.begin()) {
    return std::find_if(
        ops.begin(), ops.end(),
        [name](const OperatorDetails& item) { return item.name == name; });
  }

  auto fetchOperatorEnd() -> decltype(ops.end()) { return ops.end(); }

  DataLayout getLayoutForOperator(const char* name) {
    auto it = std::find_if(
        ops.begin(), ops.end(),
        [name](const OperatorDetails& item) { return item.name == name; });

    if (it != ops.end()) {
      return it->layout;
    }
    return getDefaultDataLayout();
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

template <int version>
std::ostream& operator<<(
    std::ostream& out,
    const typename TensorDumpConfig<version>::OperatorDetails& c) {
  out << "OperatorDetails{ " << c.name << "," << c.getLayoutString() << " }"
      << std::endl;
  return out;
}

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

template <class U, class TensorType>
TensorOutStreamer<U, TensorType>::TensorOutStreamer(const char* _label,
                                                    const char* _name,
                                                    const TensorType& _tensor,
                                                    DataLayout _layout)
    : label(_label), name(_name), tensor(_tensor), limit(0), layout(_layout) {}

template <class U, class TensorType>
TensorOutStreamer<U, TensorType>& TensorOutStreamer<U, TensorType>::setLimit(
    size_t _limit) {
  limit = _limit;
  return *this;
}

template <class U, class TensorType>
const U* TensorOutStreamer<U, TensorType>::begin() const {
  // return tensor.data<U>();
  return reinterpret_cast<const U*>(tensor.raw_data());
}

template <class U, class TensorType>
const U* TensorOutStreamer<U, TensorType>::end() const {
  return begin() + tensor.memory_size() / sizeof(U);
}

template <class U, class TensorType>
int TensorOutStreamer<U, TensorType>::checksum() const {
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
  enum { pad = 8, break_line = 16 };
  static const char* name() { return "float"; }
  static std::ostream& format(std::ostream& out, float v) {
    out << std::setw(pad) << std::setfill(' ') << v << " ";
    return out;
  }
};

template <class U, class TensorType>
std::ostream& operator<<(std::ostream& out,
                         const TensorOutStreamer<U, TensorType>& ts) {
  auto& tensor = ts.tensor;
  auto& dims = tensor.dims();
  auto& conf = TensorDumpConfig<>::get();

  out << std::setw(8) << std::setfill(' ') << (TensorDumpConfig<>::NextRecord())
      << ") type=[" << type_desc<U>::name() << "]  => label=" << ts.label
      << " name=" << ((ts.name) ? ts.name : "undef") << " crc32=" << std::hex
      << ts.checksum() << std::dec << " elem=" << tensor.numel()
      << " dims=" << dims.size() << "=>";

  for (decltype(dims.size()) i = 0; i < dims.size(); ++i) {
    out << "[" << dims.at(i) << "]";
  }

  out << " layout_default=" << DataLayoutToString(tensor.layout())
      << " layout_desired=" << DataLayoutToString(ts.layout);
  out << std::endl;
  std::size_t br = 0;

  if (ts.layout != DataLayout::kAnyLayout) {
    /* Reorder layout */
    std::vector<U> vout(tensor.memory_size() / sizeof(U));
    reorder_via_mkldnn<U>(vout.data(), &tensor, ts.layout);

    for_each_no_more(vout.begin(), vout.end(), conf.getLimitViaSize(sizeof(U)),
                     [&out, &br](U unit) {
                       if (type_desc<U>::break_line == br++) {
                         br = 0;
                         out << std::endl;
                       }
                       type_desc<U>::format(out, unit) << std::dec;
                     });

  } else {
    for_each_no_more(ts.begin(), ts.end(), conf.getLimitViaSize(sizeof(U)),
                     [&out, &br](U unit) {
                       if (type_desc<U>::break_line == br++) {
                         br = 0;
                         out << std::endl;
                       }
                       type_desc<U>::format(out, unit) << std::dec;
                     });
  }

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

template <int v>
struct Int2Type {};
template <class First, class... Tail>
struct DumpComposit {
  typedef std::tuple<First, Tail...> args_tuple;
  enum { size = std::tuple_size<args_tuple>::value };
  typedef typename std::tuple_element<size - 1, args_tuple>::type last;
  static_assert(std::is_same<last, float>::value,
                "Last one must be float type!");
  static_assert(std::is_arithmetic<First>::value,
                "Incorrect type - is not an arithmetic type!");
  template <class TensorType>
  static void execute(const char* label, const char* name,
                      const TensorType& _tensor, DataLayout layout) {
    static_assert(
        std::is_member_function_pointer<decltype(&TensorType::type)>::value,
        "Incorrect Tensor type");
    if (_tensor.type() ==
        ::paddle::framework::DataTypeTrait<First>::DataType()) {
      if (TensorDumpConfig<>::get().is_synchronized()) {
        /* in case of parallel executor , io must be synchronized */
        std::lock_guard<decltype(TensorDumpConfig<>::getMutex())> l(
            TensorDumpConfig<>::getMutex());
        TensorDumpConfig<>::get().getOutputStream()
            << TensorOutStreamer<First, TensorType>(label, name, _tensor,
                                                    layout);
      } else {
        TensorDumpConfig<>::get().getOutputStream()
            << TensorOutStreamer<First, TensorType>(label, name, _tensor,
                                                    layout);
      }
    } else {
      DumpComposit<Tail...>::execute(label, name, _tensor, layout);
    }
  }
};

template <>
struct DumpComposit<float> {
  template <class TensorType>
  static void execute(const char* label, const char* name,
                      const TensorType& _tensor, DataLayout layout) {
    if (TensorDumpConfig<>::get().is_synchronized()) {
      /* in case of parallel executor , io must be synchronized */
      std::lock_guard<decltype(TensorDumpConfig<>::getMutex())> l(
          TensorDumpConfig<>::getMutex());
      TensorDumpConfig<>::get().getOutputStream()
          << TensorOutStreamer<float, TensorType>(label, name, _tensor, layout);
    } else {
      TensorDumpConfig<>::get().getOutputStream()
          << TensorOutStreamer<float, TensorType>(label, name, _tensor, layout);
    }
  }
};

}  // namespace framework
}  // namespace paddle
