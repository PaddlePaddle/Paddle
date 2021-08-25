/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <memory>  // NOLINT
#include <mutex>   // NOLINT
#include <string>  // NOLINT
#include <thread>  // NOLINT
#include <vector>  // NOLINT
#include "paddle/fluid/framework/archive.h"

#ifdef HOST
#undef HOST
#endif
#include <mpi.h>

namespace paddle {
namespace framework {
template <class T>
struct mpi_type_trait {};

#define DECLARE_MPI_DATA_TYPE(cpp_type, mpi_type)   \
  template <>                                       \
  struct mpi_type_trait<cpp_type> {                 \
    static MPI_Datatype type() { return mpi_type; } \
  };
DECLARE_MPI_DATA_TYPE(double, MPI_DOUBLE)
DECLARE_MPI_DATA_TYPE(float, MPI_FLOAT)
DECLARE_MPI_DATA_TYPE(int32_t, MPI_INT)
DECLARE_MPI_DATA_TYPE(uint32_t, MPI_UNSIGNED)
DECLARE_MPI_DATA_TYPE(int64_t, MPI_LONG_LONG)
DECLARE_MPI_DATA_TYPE(uint64_t, MPI_UNSIGNED_LONG_LONG)
DECLARE_MPI_DATA_TYPE(long long, MPI_LONG_LONG)                    // NOLINT
DECLARE_MPI_DATA_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG)  // NOLINT
#undef DECLARE_MPI_DATA_TYPE

typedef MPI_Datatype CommDataType;
typedef decltype(MPI_COMM_WORLD) CommRole;

template <class Derived>
class MessagePassingInterface {
 public:
  virtual ~MessagePassingInterface() {}
  virtual void Initialize(int argc, char** args) = 0;
  virtual void Finalizer() = 0;
  virtual size_t Rank(CommRole comm_role = MPI_COMM_WORLD) = 0;
  virtual size_t Size(CommRole comm_role = MPI_COMM_WORLD) = 0;
  virtual int Split(CommRole* newcomm, CommRole comm_role = MPI_COMM_WORLD,
                    int color = 0, int key = 0) = 0;
  virtual void Barrier(CommRole comm_role) = 0;

  template <class T>
  T AllReduce(T x, MPI_Op op, CommRole comm_role = MPI_COMM_WORLD) {
    T tot;
    auto* derived = (Derived*)(this);  // NOLINT
    derived->AllReduceImpl(&x, &tot, 1, mpi_type_trait<T>::type(), op,
                           comm_role);
    return tot;
  }

  template <class T>
  void Bcast(T* p, int count, int root, CommRole comm_role = MPI_COMM_WORLD) {
    BinaryArchive ar;
    int len = 0;
    if (Rank(comm_role) == root) {
      for (int i = 0; i < count; i++) {
        ar << p[i];
      }
      len = boost::lexical_cast<int>(ar.Length());
    }
    auto* derived = (Derived*)(this);  // NOLINT
    derived->BcastImpl(&len, 1, MPI_INT, root, comm_role);
    ar.Resize(len);
    ar.SetCursor(ar.Buffer());
    derived->BcastImpl(ar.Buffer(), len, MPI_BYTE, root, comm_role);

    for (int i = 0; i < count; i++) {
      ar >> p[i];
    }
  }

  template <class T>
  void CheckConsistency(const T* p, int count,
                        CommRole comm_role = MPI_COMM_WORLD) {
    BinaryArchive ar;
    for (int i = 0; i < count; i++) {
      ar << p[i];
    }

    size_t hash_code = boost::hash_range(ar.Buffer(), ar.Finish());
    size_t root_hash_code = hash_code;
    auto* derived = (Derived*)(this);  // NOLINT
    derived->BcastImpl(&root_hash_code, 1, mpi_type_trait<size_t>::type(), 0,
                       comm_role);
    CHECK(root_hash_code == hash_code);
    Barrier(comm_role);
  }
};

// 略trick, 由于paddle的MPi在外层通过mpi4py已经初始化
// 因此C++内无法重新初始化，因此传入mpi4py的Comm对象指针
// 并构造与mpi4py相同的Comm结构，从而实现反解MPI原始Comm对象(pb_mpi)
class MPI4pyComm {
 public:
  ssize_t obj_cnt;
  void* ob_type;
  CommRole ob_mpi;
  unsigned int flags;
};

class CommonMPI : public MessagePassingInterface<CommonMPI> {
 public:
  virtual void Initialize(int argc, char** args);
  virtual void Finalizer();
  virtual size_t Rank(CommRole comm_role = MPI_COMM_WORLD);
  virtual size_t Size(CommRole comm_role = MPI_COMM_WORLD);
  virtual int Split(CommRole* newcomm, CommRole comm_role = MPI_COMM_WORLD,
                    int color = 0, int key = 0);
  virtual void Barrier(CommRole comm_role);
  template <class T>
  void AllReduceImpl(T* input, T* output, size_t data_num,
                     CommDataType data_type, MPI_Op op, CommRole comm_role) {
    MPI_Allreduce(input, output, data_num, data_type, op, comm_role);
  }
  template <class T>
  void BcastImpl(T* input, size_t data_num, CommDataType data_type, int root,
                 CommRole comm_role) {
    MPI_Bcast(input, data_num, data_type, root, comm_role);
  }
};

}  // namespace framework
}  // namespace paddle
