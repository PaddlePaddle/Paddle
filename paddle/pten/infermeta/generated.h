#pragma once

#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

namespace pten {

void EmptyInferMeta(const ScalarArray& shape, DataType dtype, Backend place, DataLayout layout, MetaTensor* out);

void Empty_likeInferMeta(const MetaTensor& x, DataType dtype, Backend place, DataLayout layout, MetaTensor* out);

void FullInferMeta(const ScalarArray& shape, const Scalar& value, DataType dtype, Backend place, DataLayout layout, MetaTensor* out);

void Full_likeInferMeta(const MetaTensor& x, const Scalar& value, DataType dtype, Backend place, DataLayout layout, MetaTensor* out);

void ScaleInferMeta(const MetaTensor& x, const Scalar& scale, float bias, bool bias_after_scale, MetaTensor* out);

}  // namespace pten
