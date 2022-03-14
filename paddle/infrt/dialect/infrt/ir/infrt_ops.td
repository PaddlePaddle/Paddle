include "paddle/infrt/dialect/infrt/ir/infrt_base.td"

// Op definition
class Infrt_Op<string mnemonic, list<OpTrait> traits = []> : Op<Infrt_Dialect, mnemonic, traits> {

  // Each registered op needs to provide all of a printer, parser and verifier.
  // let printer = [{ return infrt::print(p, *this); }];
  // let verifier = [{ return infrt::verify(*this); }];
  // let parser = [{ return infrt::parse$cppClass(parser, result); }];
}

def Infrt_KernelOp : Infrt_Op<"kernel", [NoSideEffect]> {
  let summary = "kernel op";
  let description = [{kernel op!}];
  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$name,
                       OptionalAttr<DictionaryAttr>:$attrs);
  let results = (outs Variadic<AnyType>);
}

def Infrt_ReturnOp : Infrt_Op<"return", [Terminator]> {
  let summary = "host executor return operation";
  let description = [{
      The "infrt.return" operation represents a return operation within a function.

        func @foo() : (i32, f8) {
        infrt.return %0, %1 : i32, f8
        }
    }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Infrt_CallOp : Infrt_Op<"call"> {
  let summary = "call a host operation";
  let description = [{
      The "infrt.call" operation represents a direct call to a function. The operands and result types of the call must match the specified function type.

          %2 = infrt.call @add(%0, %1) : (f32, f32) -> f32
    }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  //let extraClassDeclaration = [{
  //    mlir::StringRef getCallee() { return callee(); }
  //    mlir::FunctionType getCalleeType();
  //  }];
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Infrt_CvtTensorOp : Infrt_Op<"cvt_tensor", [NoSideEffect]> {
  let summary = "convert tensor type op";
  let description = [{convert tensor type op!}];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
}

def Infrt_TensorListGetTensorOp : Infrt_Op<"tensor_list_get_tensor", [NoSideEffect]> {
  let summary = "infrt.tensor_list_get_tensor operation";

  let description = [{
    An operation that can get a tensor from a TensorList.
  }];

  let arguments = (ins
          DenseTensorList:$l,
          I32Attr:$id
          );
  let results = (outs DenseTensor:$output);
  let verifier = ?;
}

def TensorListGetSizeOp : Infrt_Op<"tensor_list_get_size", [NoSideEffect]> {
  let summary = "infrt.tensor_list_get_size operation";

  let description = [{
    An operation that get the size of a TensorList.
  }];

  let arguments = (ins DenseTensorList:$map);
  let results = (outs I32:$size);
  // let assemblyFormat = "`(` $map `)` attr-dict `->` type($size)";
}