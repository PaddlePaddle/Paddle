#ifdef PTEN_TENSOR
#else
#define PTEN_TENSOR

include "paddle/infrt/dialect/pten/infrt_pten_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt_base.td"

// PTEN DenseTensor related Op.
class PDT_Op<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_DenseTensorDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op needs to provide all of a printer, parser and verifier.
  let printer = [{ return infrt::dialect::print(p, *this); }];
  let verifier = [{ return infrt::dialect::verify(*this); }];
  let parser = [{ return infrt::dialect::parse$cppClass(parser, result); }];
}


class CreateUninitTensorOp<string dtype>
      : PDT_Op<"create_uninit_tensor." # dtype, [NoSideEffect]> {
  let summary = "pdt.create_uninit_tensor operation";

  let description = [{
      An operation that creates an uninitialized tensor.
  }];

  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs TensorType:$output);

  let parser  = [{ return infrt::pten::parseCreateUninitTensorOp(parser, result); }];
  let printer = [{ return infrt::pten::printCreateUninitTensorOp(p, *this); }];
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

class FillTensorWithConstantOp<string dtype> :
      PDT_Op<"fill_tensor_with_constant." # dtype> {
  let summary = "dt.fill_tensor_with_constant operation";

  let description = [{
      An operation that fills an input tensor with a value.
  }];

  let arguments = (ins
      TensorType:$input,
      AnyAttr:$value
  );
  let results = (outs);

  let assemblyFormat = "`(` $input `:` type($input) `)`  attr-dict";
}

#endif
