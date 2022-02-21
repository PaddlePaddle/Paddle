
#include "paddle/pten/infermeta/generated.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"

namespace pten {

void EmptyInferMeta(const ScalarArray& shape, DataType dtype, Backend place, DataLayout layout, MetaTensor* out) {
  CreateInferMeta(shape, dtype, layout, out);
}

void Empty_likeInferMeta(const MetaTensor& x, DataType dtype, Backend place, DataLayout layout, MetaTensor* out) {
  CreateLikeInferMeta(x, dtype, layout, out);
}

void FullInferMeta(const ScalarArray& shape, const Scalar& value, DataType dtype, Backend place, DataLayout layout, MetaTensor* out) {
  CreateInferMeta(shape, dtype, layout, out);
}

void Full_likeInferMeta(const MetaTensor& x, const Scalar& value, DataType dtype, Backend place, DataLayout layout, MetaTensor* out) {
  CreateLikeInferMeta(x, dtype, layout, out);
}

void ScaleInferMeta(const MetaTensor& x, const Scalar& scale, float bias, bool bias_after_scale, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

}  // namespace pten

PT_REGISTER_INFER_META_FN(add, pten::ElementwiseInferMeta);
PT_REGISTER_INFER_META_FN(cast, pten::CastInferMeta);
PT_REGISTER_INFER_META_FN(concat, pten::ConcatInferMeta);
PT_REGISTER_INFER_META_FN(conj, pten::UnchangedInferMeta);
PT_REGISTER_INFER_META_FN(divide, pten::ElementwiseInferMeta);
PT_REGISTER_INFER_META_FN(dot, pten::DotInferMeta);
PT_REGISTER_INFER_META_FN(empty, pten::EmptyInferMeta);
PT_REGISTER_INFER_META_FN(empty_like, pten::Empty_likeInferMeta);
PT_REGISTER_INFER_META_FN(flatten, pten::FlattenInferMeta);
PT_REGISTER_INFER_META_FN(full, pten::FullInferMeta);
PT_REGISTER_INFER_META_FN(full_like, pten::Full_likeInferMeta);
PT_REGISTER_INFER_META_FN(matmul, pten::MatmulInferMeta);
PT_REGISTER_INFER_META_FN(mean, pten::ReduceInferMeta);
PT_REGISTER_INFER_META_FN(multiply, pten::ElementwiseInferMeta);
PT_REGISTER_INFER_META_FN(reshape, pten::ReshapeInferMeta);
PT_REGISTER_INFER_META_FN(scale, pten::ScaleInferMeta);
PT_REGISTER_INFER_META_FN(sign, pten::UnchangedInferMeta);
PT_REGISTER_INFER_META_FN(subtract, pten::ElementwiseInferMeta);
PT_REGISTER_INFER_META_FN(sum, pten::SumInferMeta);