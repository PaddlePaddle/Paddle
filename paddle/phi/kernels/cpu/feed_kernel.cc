// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/framework/feed_fetch_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

const phi::FeedType& CheckAndGetFeedItem(const phi::FeedList* x, int col) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    common::errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Feed') of current feeding variable to be "
                        "no less than 0. But received column index = %d.",
                        col));
  const auto feed_list = x;
  PADDLE_ENFORCE_LT(
      static_cast<size_t>(col),
      feed_list->size(),
      common::errors::InvalidArgument(
          "The column index of current feeding variable is expected to be "
          "less than the length of feeding list. But received column index = "
          "%d, the length of feeding list = %d",
          col,
          feed_list->size()));

  return feed_list->at(static_cast<size_t>(col));
}

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      common::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item =
      CheckAndGetFeedItem(reinterpret_cast<const phi::FeedList*>(&x), col);
  const auto& in_tensor = static_cast<DenseTensor>(feed_item);
  const auto& place = dev_ctx.GetPlace();
  if (phi::is_same_place(in_tensor.place(), place)) {
    out->ShareDataWith(in_tensor);
  } else {
    phi::Copy(dev_ctx, in_tensor, place, false, out);
  }

  out->set_lod(in_tensor.lod());
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(feed,
                                         ALL_LAYOUT,
                                         phi::FeedDenseTensorKernel) {}
