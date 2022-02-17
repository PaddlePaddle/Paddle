#ifdef PTEN_TENSOR
#else
#define PTEN_TENSOR

include "paddle/infrt/dialect/pten/infrt_pten_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"

def PTEN_DenseTensorDialect : Dialect {
  let name = "pten_dt";

  let description = [{
    The PTEN DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::pten";
}

// PTEN DenseTensor related Op.
class PDT_Op<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_DenseTensorDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

class CreateUninitTensorOp<string dtype>
      : PDT_Op<"create_uninit_tensor." # dtype, [NoSideEffect]> {
  let summary = "pdt.create_uninit_tensor operation";

  let description = [{
      An operation that creates an uninitialized tensor.
  }];

  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs TensorType:$output);
}

class CreateInitedTensorOp<string dtype, Attr array_attr>
      : PDT_Op<"create_inited_tensor." #dtype, [NoSideEffect]> {
  let summary = "pdt.create_inited_tensor operation";

  let description = [{
      An operation that creates an tensor with shape and values assigned.
  }];

  let arguments = (ins I64ArrayAttr:$shape, array_attr:$values);
  let results = (outs TensorType:$output);
}

def PrintTensorOp : PDT_Op<"print_tensor"> {
  let summary = "pdt.print_tensor operation";

  let description = [{
    An operation that prints a tensor.
  }];

  let arguments = (ins TensorType:$input);
  let results = (outs);
  let assemblyFormat = "`(` $input `:` type($input) `)` attr-dict";
}

class FillTensor<string dtype, Attr attr_type> :
      PDT_Op<"fill_tensor." # dtype> {
  let summary = "dt.fill_tensor operation";

  let description = [{
      An operation that fills an input tensor with a values.
  }];

  let arguments = (ins
      TensorType:$input,
      attr_type:$value
  );
  let results = (outs);

  let assemblyFormat = "`(` $input `:` type($input) `)`  attr-dict";
}

class FillTensorWithConstantOp<string dtype> :
      PDT_Op<"fill_tensor_with_constant." # dtype> {
  let summary = "dt.fill_tensor_with_constant operation";

  let description = [{
      An operation that fills an input tensor with a single value.
  }];

  let arguments = (ins
      TensorType:$input,
      AnyAttr:$value
  );
  let results = (outs);

  let assemblyFormat = "`(` $input `:` type($input) `)`  attr-dict";
}

foreach dtype = ["ui8", "ui16", "ui32", "ui64", "i32", "f32", "f64", "i64"] in {
    def PDT_CreateUninitTensorOp_#dtype : CreateUninitTensorOp<dtype>;
    def PDT_FillTensorWithConstantOp_#dtype : FillTensorWithConstantOp<dtype>;
}

def PDT_FillTensor_f32: FillTensor<"f32", F32ArrayAttr>;
def PDT_FillTensor_i32: FillTensor<"i32", I32ArrayAttr>;
def PDT_CreateInitedTensorOp_f32 : CreateInitedTensorOp<"f32", F32ArrayAttr>;
def PDT_CreateInitedTensorOp_i32 : CreateInitedTensorOp<"i32", I32ArrayAttr>;

#endif
