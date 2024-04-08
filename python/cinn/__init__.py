# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from .runtime.cinn_jit import to_cinn_llir  # noqa: F401
from .version import full_version as __version__  # noqa: F401

cinndir = os.path.dirname(os.path.abspath(__file__))
runtime_include_dir = os.path.join(cinndir, "libs")
cuhfile = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")

if os.path.exists(cuhfile):
    os.environ.setdefault('runtime_include_dir', runtime_include_dir)

from .backends import (  # noqa: F401
    Compiler,
    ExecutionEngine,
    ExecutionOptions,
)
from .common import (  # noqa: F401
    BFloat16,
    Bool,
    CINNValue,
    CINNValuePack,
    DefaultHostTarget,
    DefaultNVGPUTarget,
    DefaultTarget,
    Float,
    Float16,
    Int,
    RefCount,
    Shared_CINNValuePack_,
    String,
    Target,
    Type,
    UInt,
    Void,
    _CINNValuePack_,
    get_target,
    is_compiled_with_bangc,
    is_compiled_with_cuda,
    is_compiled_with_cudnn,
    is_compiled_with_hip,
    is_compiled_with_sycl,
    make_const,
    reset_name_id,
    set_target,
    type_of,
)
from .ir import (  # noqa: F401
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Args,
    Argument,
    BinaryOpNodeAdd,
    BinaryOpNodeAnd,
    BinaryOpNodeDiv,
    BinaryOpNodeEQ,
    BinaryOpNodeFracOp,
    BinaryOpNodeGE,
    BinaryOpNodeGT,
    BinaryOpNodeLE,
    BinaryOpNodeLT,
    BinaryOpNodeMax,
    BinaryOpNodeMin,
    BinaryOpNodeMod,
    BinaryOpNodeMul,
    BinaryOpNodeNE,
    BinaryOpNodeOr,
    BinaryOpNodeSub,
    Block,
    Call,
    CallOp,
    CallType,
    Cast,
    ComputeOp,
    Div,
    Expr,
    ExprNode_Module_,
    ExprNode_Tensor_,
    ExprNode_Var_,
    ExprNodeAdd,
    ExprNodeAnd,
    ExprNodeBlock,
    ExprNodeCall,
    ExprNodeCast,
    ExprNodeDiv,
    ExprNodeEQ,
    ExprNodeFloatImm,
    ExprNodeFracOp,
    ExprNodeGE,
    ExprNodeGT,
    ExprNodeIntImm,
    ExprNodeLE,
    ExprNodeLet,
    ExprNodeLoad,
    ExprNodeLT,
    ExprNodeMax,
    ExprNodeMin,
    ExprNodeMinus,
    ExprNodeMod,
    ExprNodeMul,
    ExprNodeNE,
    ExprNodeNot,
    ExprNodeOr,
    ExprNodeProduct,
    ExprNodeReduce,
    ExprNodeSelect,
    ExprNodeStore,
    ExprNodeStringImm,
    ExprNodeSub,
    ExprNodeSum,
    ExprNodeUIntImm,
    FloatImm,
    FracOp,
    IntImm,
    IrNode,
    IrNodeRef,
    IrNodeTy,
    IRVisitor,
    Let,
    Load,
    LoadStoreAddrMnger,
    LoweredFunc,
    Max,
    Min,
    Minus,
    Mod,
    Mul,
    Not,
    Operation,
    Or,
    PackedFunc,
    PlaceholderOp,
    Product,
    Reduce,
    Registry,
    Select,
    SharedIrNode,
    Store,
    StringImm,
    Sub,
    Sum,
    Tensor,
    UIntImm,
    UnaryOpNodeMinus,
    UnaryOpNodeNot,
    Var,
    _Module_,
    _Tensor_,
    _Var_,
)
from .lang import (  # noqa: F401
    Buffer,
    Module,
    Placeholder,
    ReturnType,
    call_extern,
    call_lowered,
    compute,
    create_placeholder,
    lower,
    lower_vec,
    reduce_all,
    reduce_any,
    reduce_max,
    reduce_min,
    reduce_mul,
    reduce_sum,
)
from .poly import (  # noqa: F401
    Condition,
    Iterator,
    SharedStage,
    SharedStageMap,
    Stage,
    StageMap,
    create_stages,
)

is_compiled_with_device = (
    is_compiled_with_cuda() or is_compiled_with_sycl() or is_compiled_with_hip()
)
if is_compiled_with_device:
    cinndir = os.path.dirname(os.path.abspath(__file__))
    runtime_include_dir = os.path.join(cinndir, "libs")
    if is_compiled_with_cuda():
        hfile = os.path.join(
            runtime_include_dir, "cinn_cuda_runtime_source.cuh"
        )
    elif is_compiled_with_sycl():
        hfile = os.path.join(runtime_include_dir, "cinn_sycl_runtime_source.h")
    elif is_compiled_with_hip():
        hfile = os.path.join(runtime_include_dir, "cinn_hip_runtime_source.h")
    if os.path.exists(hfile):
        os.environ.setdefault('runtime_include_dir', runtime_include_dir)
