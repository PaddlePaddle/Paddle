def PD_FeedOp : PD_Op<"feed", [NoSideEffect]> {
  let summary = "Feed Op";

  let description = [{
    Feed a tensor into the model.
  }];

  let arguments = (ins);
  let results = (outs PD_Tensor:$out);

  let assemblyFormat = [{
      `(` `)` attr-dict `:` type($out)
  }];
}

def PD_FetchOp : PD_Op<"fetch", [Terminator]> {
  let summary = "fetch Op";

  let description = [{
    Fetch tensor from the graph.
  }];

  let arguments = (ins Variadic<PD_Tensor>:$inputs);
}

def PD_GraphOp : PD_Op<"graph", [SingleBlockImplicitTerminator<"FetchOp">]> {
  let summary = "paddle graph Op";
  let description = [{
    Describe a paddle graph or subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<PD_Tensor>:$inputs);
  let results = (outs Variadic<PD_Tensor>:$outputs);
}

def PD_ConstantOp : PD_Op<"Constant", [NoSideEffect, ConstantLike, DeclareOpInterfaceMethods<InferTypeOpInterface>, AllTypesMatch<["value", "output"]>]> {
  let summary = "constant Op";
  let description = [{}];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs PD_Tensor:$output);
  let hasFolder = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Attribute value">,
  ];
}
