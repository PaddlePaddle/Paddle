// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/ops/_local_scalar_dense.h>
#include <ATen/ops/_nnz.h>
#include <ATen/ops/_values.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/all.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/any.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/coalesce.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/dsplit.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/expand.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/hsplit.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/is_coalesced.h>
#include <ATen/ops/item.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/rename.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/select.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_csr_tensor.h>
#include <ATen/ops/split.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/std.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/t.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/to.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/unflatten.h>
#include <ATen/ops/unsafe_split.h>
#include <ATen/ops/unsafe_split_with_sizes.h>
#include <ATen/ops/unsqueeze.h>
#include <ATen/ops/view.h>
#include <ATen/ops/view_as.h>
#include <ATen/ops/vsplit.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
