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

# build with gpu
ABSL_INCLUDE_FLAG=-Ithird_party/absl/include
ISL_INCLUDE_FLAG=-Ithird_party/isl/include
LLVM_INCLUDE_FLAG=-Ithird_party/llvm/include
GLOG_INCLUDE_FLAG=-Ithird_party/glog/include
GFLAGS_INCLUDE_FLAG=-Ithird_party/gflags/include
PROTOBUF_INCLUDE_FLAG=-Ithird_party/protobuf/include
MKLML_INCLUDE_FLAG=-Ithird_party/mklml/include

THIRD_PARTY_INCLUDES="${ABSL_INCLUDE_FLAG} ${ISL_INCLUDE_FLAG} ${LLVM_INCLUDE_FLAG} ${GLOG_INCLUDE_FLAG} ${GFLAGS_INCLUDE_FLAG} ${PROTOBUF_INCLUDE_FLAG} ${MKLML_INCLUDE_FLAG}"
g++ demo.cc -o demo -fPIC -mavx -Wno-write-strings -Wno-psabi   -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -std=c++17 ${THIRD_PARTY_INCLUDES} -I./cinn/include cinn/lib/libcinnapi.so -lpthread -ldl

# build without gpu
# g++ demo.cc -o demo -fPIC -mavx -Wno-write-strings -Wno-psabi   -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D_GNU_SOURCE -D_DEBUG -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -std=gnu++1z -I./cinn/include -I./third_party/llvm11/include -I./third_party/glog/include -I./third_party/gflags/include -I./third_party/protobuf/include -I./third_party/mklml/include ./cinn/lib/libcinncore.a ../cinn/frontend/paddle/libframework_proto.a ./third_party/glog/lib/libglog.a ./third_party/gflags/lib/libgflags.a ./third_party/protobuf/lib/libprotobuf.a ./third_party/llvm11/lib/libLLVMX86CodeGen.a ./third_party/llvm11/lib/libLLVMAsmPrinter.a ./third_party/llvm11/lib/libLLVMDebugInfoDWARF.a ./third_party/llvm11/lib/libLLVMCFGuard.a ./third_party/llvm11/lib/libLLVMGlobalISel.a ./third_party/llvm11/lib/libLLVMSelectionDAG.a ./third_party/llvm11/lib/libLLVMX86AsmParser.a ./third_party/llvm11/lib/libLLVMX86Desc.a ./third_party/llvm11/lib/libLLVMX86Disassembler.a ./third_party/llvm11/lib/libLLVMMCDisassembler.a ./third_party/llvm11/lib/libLLVMX86Info.a ./third_party/llvm11/lib/libLLVMOrcJIT.a ./third_party/llvm11/lib/libLLVMJITLink.a ./third_party/llvm11/lib/libLLVMOrcError.a ./third_party/llvm11/lib/libLLVMPasses.a ./third_party/llvm11/lib/libLLVMCoroutines.a ./third_party/llvm11/lib/libLLVMipo.a ./third_party/llvm11/lib/libLLVMIRReader.a ./third_party/llvm11/lib/libLLVMAsmParser.a ./third_party/llvm11/lib/libLLVMInstrumentation.a ./third_party/llvm11/lib/libLLVMVectorize.a ./third_party/llvm11/lib/libLLVMFrontendOpenMP.a ./third_party/llvm11/lib/libLLVMLinker.a ./third_party/llvm11/lib/libLLVMMCJIT.a ./third_party/llvm11/lib/libLLVMExecutionEngine.a ./third_party/llvm11/lib/libLLVMRuntimeDyld.a ./third_party/llvm11/lib/libLLVMCodeGen.a ./third_party/llvm11/lib/libLLVMTarget.a ./third_party/llvm11/lib/libLLVMBitWriter.a ./third_party/llvm11/lib/libLLVMScalarOpts.a ./third_party/llvm11/lib/libLLVMAggressiveInstCombine.a ./third_party/llvm11/lib/libLLVMInstCombine.a ./third_party/llvm11/lib/libLLVMTransformUtils.a ./third_party/llvm11/lib/libLLVMAnalysis.a ./third_party/llvm11/lib/libLLVMProfileData.a ./third_party/llvm11/lib/libLLVMObject.a ./third_party/llvm11/lib/libLLVMBitReader.a ./third_party/llvm11/lib/libLLVMCore.a ./third_party/llvm11/lib/libLLVMRemarks.a ./third_party/llvm11/lib/libLLVMBitstreamReader.a ./third_party/llvm11/lib/libLLVMMCParser.a ./third_party/llvm11/lib/libLLVMMC.a ./third_party/llvm11/lib/libLLVMDebugInfoCodeView.a ./third_party/llvm11/lib/libLLVMDebugInfoMSF.a ./third_party/llvm11/lib/libLLVMTextAPI.a ./third_party/llvm11/lib/libLLVMBinaryFormat.a ./third_party/llvm11/lib/libLLVMSupport.a ./third_party/llvm11/lib/libLLVMDemangle.a -lpthread -ldl -ltinfo /usr/local/lib/libisl.so -lginac
