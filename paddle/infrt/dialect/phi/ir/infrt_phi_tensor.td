#ifdef PHI_TENSOR
#else
#define PHI_TENSOR

include "paddle/infrt/dialect/phi/ir/infrt_phi_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"

def PHI_DenseTensorDialect : Dialect {
  let name = "phi_dt";

  let description = [{
    The PHI DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

// PHI DenseTensor related Op.
class PDT_Op<string mnemonic, list<OpTrait> traits = []> : Op<PHI_DenseTensorDialect,
  mnemonic, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove])> {}

class CreateDenseTensorOp<string target>
      : PDT_Op<"create_dense_tensor." # target, [NoSideEffect]> {
  let arguments = (ins Context:$context, I64ArrayAttr:$dims, 
    LayoutAttr:$layout, I64ArrayAttr:$lod, PrecisionAttr:$precision);
  let results = (outs DenseTensor:$output);
}

def CreateHostInitedDenseTensorOp : PDT_Op<"create_host_inited_dense_tensor.f32", [NoSideEffect]> {
  let arguments = (ins
    Context:$context,
    I64ArrayAttr:$dims,
    LayoutAttr:$layout,
    I64ArrayAttr:$lod,
    F32ArrayAttr:$values,
    DefaultValuedAttr<BoolAttr, "true">:$run_once
  );
  let results = (outs DenseTensor:$output);
}

def CreateInitedGpuFLOAT32DenseTensorOp
      : PDT_Op<"create_inited_dense_tensor.gpu.f32", [NoSideEffect]> {
  let arguments = (ins Context:$context, I64ArrayAttr:$dims,
    LayoutAttr:$layout, I64ArrayAttr:$lod, F32Attr:$value);
  let results = (outs DenseTensor:$output);
}

def CreateInitedCpuFLOAT32DenseTensorOp
      : PDT_Op<"create_inited_dense_tensor.cpu.f32", [NoSideEffect]> {
  let arguments = (ins Context:$context, I64ArrayAttr:$dims, 
    LayoutAttr:$layout, I64ArrayAttr:$lod, F32Attr:$value);
  let results = (outs DenseTensor:$output);
}

class FillDenseTensorOp<Attr attr_type, string dtype> :
      PDT_Op<"fill_dense_tensor." # dtype> {
  let arguments = (ins
      DenseTensor:$input,
      attr_type:$value
  );
  let results = (outs);
  let assemblyFormat = "`(` $input `:` type($input) `)` attr-dict";
}

class PrintDenseTensorOp:
      PDT_Op<"print_tensor"> {
  let arguments = (ins DenseTensor:$input);
  let results = (outs);
  let assemblyFormat = "`(` $input `:` type($input) `)` attr-dict";
}

class CreateContextOp<string target>
      : PDT_Op<"create_context." # target, [NoSideEffect]> {
  let arguments = (ins);
  let results = (outs Context:$output);
}

def PDT_LoadParamsOp : PDT_Op<"load_params", [NoSideEffect]> {
  // input path of model params.
  let arguments = (ins StrAttr:$path);
  let results = (outs PD_DenseTensorMap:$out);

  let assemblyFormat = "`(``)`attr-dict";
}

def PDT_LoadCombinedParamsOp : PDT_Op<"load_combined_params", [NoSideEffect]> {
  // input path of model params.
  let arguments = (ins StrAttr:$model_path, StrAttr:$params_path);
  let results = (outs PD_DenseTensorMap:$out);

  let assemblyFormat = "`(``)`attr-dict";
}

def PDT_LoadCombinedParamsGpuOp : PDT_Op<"load_combined_params_to_gpu", [NoSideEffect]> {
  // input path of model params.
  let arguments = (ins StrAttr:$model_path, StrAttr:$params_path);
  let results = (outs PD_DenseTensorMap:$out);

  let assemblyFormat = "`(``)`attr-dict";
}

def PDT_TensorMapGetSizeOp : PDT_Op<"tensor_map_get_size", [NoSideEffect]> {
  let arguments = (ins PD_DenseTensorMap:$map);
  let results = (outs I32:$size);
  let assemblyFormat = "`(` $map `)` attr-dict `->` type($size)";
}

class TensorMapGetTensorOp:
      PDT_Op<"tensor_map_get_tensor"> {
  let arguments = (ins
          PD_DenseTensorMap:$map,
          StrAttr:$name
          );
  let results = (outs DenseTensor:$output);
  let assemblyFormat = "`(` operands `)` attr-dict `->` type($output)";
  let verifier = ?;
}

def PDT_CreateCPUDenseTensorOp : CreateDenseTensorOp<"cpu">;
def PDT_CreateGPUDenseTensorOp : CreateDenseTensorOp<"gpu">;
def PDT_FillDenseTensorOp_f32 : FillDenseTensorOp<F32ArrayAttr, "f32">;
def PDT_CreateCPUContextOp : CreateContextOp<"cpu">;
def PDT_CreateGPUContextOp : CreateContextOp<"gpu">;
def PDT_PrintDenseTensor : PrintDenseTensorOp;
def PDT_TensorMapGetTensorOp: TensorMapGetTensorOp;

def FakeKernelOp : PDT_Op<"fake_phi_kernel"> {
  let arguments = (ins Context:$dev_ctx, DenseTensor:$x, DenseTensor:$y, BoolAttr:$transpose_x, BoolAttr:$transpose_y);
  let results = (outs DenseTensor:$output);
}

// TODO(wilber): Add a infrt_gpu dialect.
def PDT_GpuMemCopyOp : PDT_Op<"memcpy.gpu", [NoSideEffect]> {
  let summary = "phi_dt.gpu.memcpy";
  let description = [{gpu memcpy d2h or h2d}];
  // TODO(wilber): add context argument to support stream.
  let arguments = (ins
    DenseTensor:$input,
    Context:$context,
    BoolAttr:$d2h
  );
  let results = (outs DenseTensor:$output);
}

#endif
