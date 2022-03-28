include "mlir/Interfaces/InferTypeOpInterface.td"
include "paddle/infrt/dialect/core/ir/core_base.td"

// Op definition
class Core_Op<string mnemonic, list<OpTrait> traits = []> : Op<Core_Dialect, mnemonic, traits> {
}

def Core_GraphOp : Core_Op<"graph", [SingleBlockImplicitTerminator<"core::ReturnOp">]> {
  let summary = "paddle graph Op";
  let description = [{
    Describe a paddle graph or subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
}

def Core_KernelOp : Core_Op<"kernel", [NoSideEffect]> {
  let summary = "kernel op";
  let description = [{kernel op!}];
  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$name,
                       OptionalAttr<DictionaryAttr>:$attrs);
  let results = (outs Variadic<AnyType>);
}

def Core_ReturnOp : Core_Op<"return", [Terminator]> {
  let summary = "host executor return operation";
  let description = [{
      The "core.return" operation represents a return operation within a function.

        func @foo() : (i32, f8) {
        core.return %0, %1 : i32, f8
        }
    }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Core_CallOp : Core_Op<"call"> {
  let summary = "call a host operation";
  let description = [{
      The "core.call" operation represents a direct call to a function. The operands and result types of the call must match the specified function type.

          %2 = core.call @add(%0, %1) : (f32, f32) -> f32
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

def Core_TensorCastOp : Core_Op<"tensor_cast", [NoSideEffect]> {
  let summary = "cast tensor type op";
  let description = [{cast tensor type op!}];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
}

def Core_ConstantOp : Core_Op<"constant", [NoSideEffect, ConstantLike, DeclareOpInterfaceMethods<InferTypeOpInterface>, AllTypesMatch<["value", "output"]>]> {
  let summary = "constant Op";
  let description = [{}];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyType:$output);
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value)>,
  ];
}
