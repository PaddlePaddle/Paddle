#include "paddle/framework/tensor_array.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {

class TensorArrayTester : public ::testing::Test {
 protected:
  void SetUp() override {
    LoDTensor source;
    source.Resize(make_ddim({batch_size, dim}));
    int* data = source.mutable_data<int>(platform::CPUPlace());
    for (int i = 0; i < 16 * 32; i++) {
      data[i] = i;
    }
    ta.Unstack(source);
  }

  TensorArray ta;
  const int batch_size{16};
  const int dim{32};
};

TEST_F(TensorArrayTester, Read) {
  for (int i = 0; i < batch_size; i++) {
    const auto& tensor = ta.Read(i);
    ASSERT_EQ(tensor.dims()[0], 1);
    ASSERT_EQ(tensor.dims()[1], dim);
  }
}

TEST_F(TensorArrayTester, Write) {
  LoDTensor source;
  source.Resize(make_ddim({1, dim}));
  for (int i = 0; i < dim; i++) {
    *(source.mutable_data<int>(platform::CPUPlace()) + i) = i;
  }

  ta.Write(2, source);

  const auto& tensor = ta.Read(2);
  for (int i = 0; i < dim; i++) {
    EXPECT_EQ(*(tensor.data<int>() + i), *(source.data<int>() + i));
  }
}

TEST_F(TensorArrayTester, WriteShared) {
  LoDTensor source;
  source.Resize(make_ddim({1, dim}));
  for (int i = 0; i < dim; i++) {
    *(source.mutable_data<int>(platform::CPUPlace()) + i) = i;
  }

  ta.WriteShared(2, source);

  const auto& tensor = ta.Read(2);
  for (int i = 0; i < dim; i++) {
    EXPECT_EQ(*(tensor.data<int>() + i), *(source.data<int>() + i));
  }

  EXPECT_EQ(source.data<int>(), tensor.data<int>());
}

class TensorArrayPackTester : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    lod.push_back(std::vector<size_t>{0, 2, 9, 13});

    source.set_lod(lod);
    source.Resize(make_ddim({13, 128}));
    source.mutable_data<int>(platform::CPUPlace());

    // content of each setence: 0 1 2 3 4
    const auto& level = lod.front();
    for (size_t i = 0; i < level.size() - 1; i++) {
      size_t begin = level[i];
      size_t end = level[i + 1];
      for (size_t j = begin; j < end; j++) {
        auto record = source.Slice<int>(j, j + 1);
        for (int dim = 0; dim < 128; dim++) {
          record.mutable_data<int>(platform::CPUPlace())[dim] = j - begin;
        }
      }
    }

    // unpack
    meta = ta.Unpack(source, 0, true);
  }

  LoD lod;
  TensorArray ta;
  LoDTensor source;
  std::vector<DySeqMeta> meta;
};

TEST_F(TensorArrayPackTester, Unpack) {
  ASSERT_EQ(ta.size(), 7UL);

  const auto& t0 = ta.Read(0);
  const auto& t1 = ta.Read(1);

  ASSERT_EQ(t0.data<int>()[0], int(0));
  ASSERT_EQ(t1.data<int>()[0], int(1));
}

TEST_F(TensorArrayPackTester, Pack) {
  LoDTensor packed = ta.Pack(0, meta, lod);
}

TEST_F(TensorArrayTester, size) { ASSERT_EQ(ta.size(), (size_t)batch_size); }

}  // namespace framework
}  // namespace paddle
