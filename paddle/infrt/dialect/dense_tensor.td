#ifdef DT_OPS
#else
#define DT_OPS

include "paddle/infrt/dialect/infrt_base.td"
include "paddle/infrt/dialect/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DT_Dialect : Dialect {
  let name = "dt";

  let description = [{
      The DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::dt";
}

class DT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<DT_Dialect, mnemonic, traits>;

class CreateUninitTensorOp<string dtype>
      : DT_Op<"create_uninit_tensor." # dtype, [NoSideEffect]> {
  let summary = "dt.create_uninit_tensor operation";

  let description = [{
      An operation that creates an uninitialized tensor.
  }];

  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs TensorType:$output);

  let parser  = [{ return infrt::dt::parseCreateUninitTensorOp(parser, result); }];
  let printer = [{ return infrt::dt::printCreateUninitTensorOp(p, *this); }];
}


def ShallowCopyTensorOp
      : DT_Op<"shallow_copy_tensor", [NoSideEffect]> {
  let summary = "dt.shallow_copy_tensor operation";

  let description = [{
      An operation that copy a tensor shallowly.
  }];

  let arguments = (ins TensorType:$input);
  let results = (outs TensorType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}


class FillTensorWithConstantOp<string dtype> :
      DT_Op<"fill_tensor_with_constant." # dtype> {
  let summary = "dt.fill_tensor_with_constant operation";

  let description = [{
      An operation that fills an input tensor with a value.
  }];

  let arguments = (ins
      TensorType:$input,
      AnyAttr:$value
  );
  let results = (outs);

  // TODO: can be removed?
  //let parser  = [{ return infrt::dt::parseFillTensorWithConstantOp(parser, result); }];
  //let printer = [{ return infrt::dt::printFillTensorWithConstantOp(p, *this); }];
  let assemblyFormat = "`(` $input `:` type($input) `)`  attr-dict";
}

def PrintTensorOp : DT_Op<"print_tensor"> {
  let summary = "dt.print_tensor operation";

  let description = [{
    An operation that prints a tensor.
  }];

  let arguments = (ins TensorType:$input);
  let results = (outs);
  let assemblyFormat = "`(` $input `:` type($input) `)` attr-dict";
}

class SetTensorOp<string dtype> :
      DT_Op<"set_tensor_with_constant_values." # dtype> {
  let summary = "dt.set_tensor_with_constant_values operation";

  let description = [{
    An operation that sets an input tensor with given values.
  }];

  let arguments = (ins TensorType);
  let results = (outs);

  let parser  = [{ return infrt::dt::parseSetTensorOp(parser, result); }];
  let printer = [{ return infrt::dt::printSetTensorOp(p, *this); }];
}

def LoadParamsOp : DT_Op<"load_params", [NoSideEffect]> {
  let summary = "dt.load_params operation";

  let description = [{
    An operation that can load tensors to TensorMap.
  }];

  // input path of model params.
  let arguments = (ins StringType:$path);
  let results = (outs TensorMapType);

  let assemblyFormat = "`(` operands `)` attr-dict";
  let verifier = ?;
}

def TensorMapGetTensorOp : DT_Op<"tensor_map_get_tensor", [NoSideEffect]> {
  let summary = "dt.tensor_map_get_tensor operation";

  let description = [{
    An operation that can get a tensor from a TensorMap.
  }];

  // input path of model params.
  let arguments = (ins
          TensorMapType:$map,
          StringType:$name
          );
  let results = (outs TensorType:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `->` type($output)";
  let verifier = ?;
}

def TensorMapGetSizeOp : DT_Op<"tensor_map_get_size", [NoSideEffect]> {
  let summary = "ddt.tensor_map_get_size operation";

  let description = [{
    An operation that get the size of a TensorMap.
  }];

  let arguments = (ins TensorMapType:$map);
  let results = (outs I32:$size);
  let assemblyFormat = "`(` $map `)` attr-dict `->` type($size)";
}

def GetTensorShapeOp : DT_Op<"get_tensor_shape", [NoSideEffect]> {
  let summary = "dt.get_tensor_shape operation";

  let description = [{
      An operation that returns the shape of the input tensor.
  }];

  let arguments = (ins TensorType:$input);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

class NaiveElementwiseAddOp<string dtype> :
    DT_Op<"naive_elementwise_add." # dtype, [NoSideEffect]> {
  let summary = "dt.naive_elementwise_add operation";

  let description = [{
    Naive elementwise_add operation.
    Just for testing.
  }];
  let arguments = (ins TensorType:$a, TensorType:$b);
  let results = (outs TensorType:$output);
  let assemblyFormat = "`(` $a `,` $b `)` attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($output)";
}

class NaiveMatmulOp<string dtype> :
    DT_Op<"naive_matmul." # dtype, [NoSideEffect]> {
  let summary = "dt.naive_matmul operation";

  let description = [{
    Naive matmul operation.
    Just for testing.
  }];
  let arguments = (ins TensorType:$x, TensorType:$w);
  let results = (outs TensorType:$output);
  let assemblyFormat = "`(` $x `,` $w `)` attr-dict `:` `(` type($x) `,` type($w) `)` `->` type($output)";
}

foreach dtype = ["ui8", "ui16", "ui32", "ui64", "i32", "f32", "f64", "i64"] in {
  def DT_CreateUninitTensorOp_#dtype : CreateUninitTensorOp<dtype>;
  def DT_FillTensorOp_#dtype : FillTensorWithConstantOp<dtype>;
  def DT_SetTensorOp_#dtype : SetTensorOp<dtype>;
  def DT_NaiveElementwiseAddOp_#dtype : NaiveElementwiseAddOp<dtype>;
  def DT_NaiveMatmulOp_#dtype : NaiveMatmulOp<dtype>;
}

#endif  // DT_OPS
