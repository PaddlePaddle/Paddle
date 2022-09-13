#################################################################################################
#
# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

# System imports
import struct
import io
import ctypes

# CUDA Python import
from cuda import cuda
from cuda import nvrtc

# CUTLASS imports
from library import *
from gemm_operation import EmitGemmUniversalInstance

#################################################################################################
#
# CUTLASS Py Runtime Components
#
#################################################################################################

#
def MaxAlignment(fmt):
  align = 1
  for x in fmt:
    align = max(align, struct.calcsize(x))
  return align
  
#
def AlignedOffset(offset, align):
  remainder = (offset % align)
  if remainder:
    offset += (align - remainder)
  return offset

#
def PackInteger(host_workspace, offset, value):
  fmt = "i"
  padding = AlignedOffset(offset, 4)  
  struct.pack_into(fmt, host_workspace, offset, value)
  return padding + struct.calcsize(fmt)

#
def PackDevicePointer(host_workspace, offset, value):
  fmt = "P"
  offset = AlignedOffset(offset, 8)
  struct.pack_into(fmt, host_workspace, offset, value)
  return offset + struct.calcsize(fmt)
  
#
def ceil_div(a, b):
  return -(a // -b)

#################################################################################################

#
class PitchLinearCoord:
  def __init__(self, contiguous, strided):
    self.contiguous = contiguous
    self.strided = strided

#
class GemmCoord:
  def __init__(self, m = 1, n = 1, k = 1):
    self.m = m
    self.n = n
    self.k = k
    self.fmt = "iii"

  #
  def ceil_div(self, rhs):
    return GemmCoord(ceil_div(self.m, rhs.m), ceil_div(self.n, rhs.n), ceil_div(self.k, rhs.k))

  #
  def size(self):
    return struct.calcsize(self.fmt)

  #
  def alignment(self):
    return MaxAlignment(self.fmt)

  #
  def pack_into(self, host_workspace, offset):
    
    offset = AlignedOffset(offset, 4)
  
    struct.pack_into(
      self.fmt, 
      host_workspace, 
      offset, 
      self.m, self.n, self.k)

    return offset + self.size()

#
class TensorRef:
  def __init__(self, pointer = None, layout = 0):
    self.pointer = pointer
    self.layout = layout

  def __str__(self):
    return "(%x, %d)" % (self.pointer._ptr, self.layout)

#################################################################################################

#
class PredicatedTileAccessIteratorDesc:
  '''
  '''
  
  def __init__(
      self, 
      element_size_bits, 
      advance_rank, 
      threadblock_shape, 
      threadmap_iterations, 
      threadmap_delta):

    self.element_size_bits = element_size_bits
    self.advance_rank = advance_rank
    self.threadblock_shape = threadblock_shape
    self.threadmap_iterations = threadmap_iterations
    self.threadmap_delta = threadmap_delta

#
class PredicatedTileAccessIteratorParams:
  '''
  '''
  #
  def __init__(self, desc, label):
    self.desc = desc
    self.label = label
    self.fmt = "qqqq"
  #
  def size(self):
    return struct.calcsize(self.fmt)

  #
  def alignment(self):
    return MaxAlignment(self.fmt)

  #
  def initialize(self, host_workspace, offset, stride):

    offset = AlignedOffset(offset, self.alignment())

    inc_strided = stride *                            \
                  self.desc.threadmap_delta.strided * \
                  self.desc.element_size_bits // 8

    if self.desc.advance_rank:
      inc_advance = self.desc.threadblock_shape.strided * \
                          stride *                        \
                          self.desc.element_size_bits // 8
    else:
      inc_advance = self.desc.threadblock_shape.contiguous * \
                          self.desc.element_size_bits // 8

    inc_next = inc_advance - (self.desc.threadmap_iterations.strided - 1) * \
                      self.desc.threadmap_delta.strided *                   \
                      stride *                                              \
                      self.desc.element_size_bits // 8

    struct.pack_into(
      self.fmt, 
      host_workspace, 
      offset, 
      stride, inc_strided, inc_next, inc_advance)

    return offset + self.size()
  #

#################################################################################################

#
class EpilogueTileDesc:
  '''
  '''
  def __init__(self, column, row, group, cluster, tile):
    self.column = column
    self.row = row
    self.group = group
    self.cluster = cluster
    self.tile = tile

#
class EpilogueThreadMap:
  '''
  '''
  def __init__(self, threads, elements_per_access, element_size_bits, shape, iterations, delta, count):
    self.threads = threads
    self.elements_per_access = elements_per_access
    self.element_size_bits = element_size_bits
    self.shape = shape
    self.iterations = iterations
    self.delta = delta
    self.count = count
    pass

#
class EpilogueTileIteratorParams:
  '''
  '''
  #
  def __init__(self, desc, label):
    self.desc = desc
    self.label = label
    self.fmt = "qqqqqqqq"

  #
  def size(self):
    return struct.calcsize(self.fmt)

  #
  def alignment(self):
    return MaxAlignment(self.fmt)

  #
  def initialize(self, host_workspace, offset, stride):

    stride = stride * self.desc.element_size_bits // 8

    offset = AlignedOffset(offset, self.alignment())

    increment_row = stride * self.desc.delta.row

    increment_group = stride * self.desc.delta.group \
      - stride * self.desc.delta.row * (self.desc.iterations.row - 1)

    increment_cluster = stride * self.desc.delta.cluster \
      - stride * self.desc.delta.group * (self.desc.iterations.group - 1) \
      - stride * self.desc.delta.row * (self.desc.iterations.row - 1)
      
    advance_row = stride * self.desc.shape.row

    advance_group = stride *                   \
      (self.desc.shape.group - 1) * \
      self.desc.shape.row *         \
      self.desc.count.row

    advance_cluster = stride * \
      self.desc.count.group * \
      self.desc.shape.group * \
      self.desc.count.row   * \
      self.desc.shape.row

    advance_tile = stride * \
      self.desc.shape.group * \
      self.desc.shape.row   * \
      self.desc.shape.cluster * \
      self.desc.shape.tile

    struct.pack_into(
      self.fmt,                                           \
      host_workspace,                                     \
      offset,                                             \
      stride,                                             \
      increment_row, increment_group, increment_cluster,  \
      advance_row, advance_group, advance_cluster, advance_tile)

    return offset + self.size()
#

#################################################################################################
#
# Launch configuration
#
#################################################################################################

class LaunchConfiguration:
  def __init__(self, grid = [1,1,1], block = [1,1,1], smem = 0):
    self.grid = grid
    self.block = block
    self.shared_memory_capacity = smem

#################################################################################################
#
# Functors
#
#################################################################################################

#
class Functor:
  def __init__(self):
    self.decl = ''
    self.definition = ''
    self.fmt = ''
    self.identifier = ''

  #
  def emit_declaration(self):
    return self.decl

  #
  def emit_definition(self):
    return self.definition

  # 
  def size(self):
    '''
    Size of the packed Params structure
    '''
    return struct.calcsize(self.fmt)

  #
  def alignment(self):
    return MaxAlignment(self.fmt)

  # 
  def initialize(self, host_workspace, offset, arguments):
    return offset + self.size()

#################################################################################################

#
class LinearCombinationFunctorArguments:
  def __init__(self, alpha = 1.0, beta = 0.0):
    self.alpha = alpha
    self.beta = beta
    self.alpha_ptr = 0
    self.beta_ptr = 0

#
class LinearCombinationFunctor(Functor):
  def __init__(self):
    super().__init__()

    self.decl = """
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >"""
    self.identifier = 'linear_combination'
    self.fmt = "ffPP"

  # 
  def size(self):
    '''
    Size of the packed Params structure
    '''
    return struct.calcsize(self.fmt)

  #
  def alignment(self):
    return MaxAlignment(self.fmt)

  # 
  def initialize(self, host_workspace, offset, arguments):

    offset = AlignedOffset(offset, self.alignment())

    struct.pack_into(
      self.fmt, 
      host_workspace, offset, 
      arguments.alpha, arguments.beta, arguments.alpha_ptr, arguments.beta_ptr)

    return offset + self.size()

#################################################################################################
#
# Base class for an executable operation
#
#################################################################################################

#
class ExecutableOperation:
  '''
  '''
  def __init__(self, operation):
    self.operation = operation
    self.module = None
    self.kernel = None

  #
  def name(self):
    return self.operation.procedural_name()

  #
  def emit(self):
    return ''

  #
  def can_implement(self, configuration, arguments):
    return False

  #
  def get_host_workspace_size(self, arguments):
    return 0

  #
  def get_device_workspace_size(self, arguments):
    return 0

  #
  def plan(self, arguments):
    return LaunchConfiguration()

  #
  def initialize(self, host_workspace, device_workspace, launch_config, arguments, stream = cuda.CUstream(0)):
    raise NotImplementedError()

  #
  def run(self, host_workspace, device_workspace, launch_config, stream = cuda.CUstream(0)):

    cArg = (ctypes.c_char * len(host_workspace)).from_buffer(host_workspace)
    packed = (ctypes.c_void_p * 1)()
    packed[0] = ctypes.addressof(cArg)

    err, = cuda.cuLaunchKernel(
      self.kernel, 
      launch_config.grid[0], launch_config.grid[1], launch_config.grid[2], 
      launch_config.block[0], launch_config.block[1], launch_config.block[2], 
      launch_config.shared_memory_capacity, 
      stream, 
      packed, 
      0)

    return err

#################################################################################################


#
class GemmArguments:
  '''
  '''
  def __init__(self):
    self.problem_size = GemmCoord(0, 0, 0)
    self.A = TensorRef()
    self.B = TensorRef()
    self.C = TensorRef()
    self.D = TensorRef()
    self.output_op = LinearCombinationFunctorArguments()

#
class ThreadblockSwizzle:
  def __init__(self, threadblock_shape, log_threadblock_cohort = 0):
    self.threadblock_shape = threadblock_shape
    self.log_threadblock_cohort = log_threadblock_cohort

  def grid_tiled_shape(self, problem_size):
    return GemmCoord(
      ceil_div(problem_size.m, self.threadblock_shape.m), 
      ceil_div(problem_size.n, self.threadblock_shape.n), 
      1)

#
class Gemm(ExecutableOperation):
  '''
  GEMM manages the CUTLASS runtime components
  '''
  #
  def __init__(self, operation):
    super().__init__(operation)

    self.emitter = EmitGemmUniversalInstance('_type')
    self.threadblock_swizzle = ThreadblockSwizzle(GemmCoord(128, 128, 8))

    self.threads = 256
    self.shared_memory_capacity = (32 << 10)

    self.params_A = PredicatedTileAccessIteratorParams(
        PredicatedTileAccessIteratorDesc(
          32, 
          1, 
          PitchLinearCoord(128, 8), 
          PitchLinearCoord(1, 4), 
          PitchLinearCoord(1, 2)), 'A')

    self.params_B = PredicatedTileAccessIteratorParams(
        PredicatedTileAccessIteratorDesc(
          32, 
          1, 
          PitchLinearCoord(128, 8), 
          PitchLinearCoord(1, 4), 
          PitchLinearCoord(1, 2)), 'B')

    self.params_C = EpilogueTileIteratorParams(
      EpilogueThreadMap(
        256, 
        1, 
        32,
        EpilogueTileDesc(128, 1, 4, 4, 1), 
        EpilogueTileDesc(4, 1, 2, 1, 1), 
        EpilogueTileDesc(32, 1, 8, 1, 1), 
        EpilogueTileDesc(1, 4, 2, 1, 8)), 'C')

    self.params_D = EpilogueTileIteratorParams(
      EpilogueThreadMap(
        256, 
        1,
        32,
        EpilogueTileDesc(128, 1, 4, 4, 1), 
        EpilogueTileDesc(4, 1, 2, 1, 1), 
        EpilogueTileDesc(32, 1, 8, 1, 1), 
        EpilogueTileDesc(1, 4, 2, 1, 8)), 'D')

    self.output_op = LinearCombinationFunctor()

  #
  def emit(self):
    return self.emitter.emit(self.operation)

  #
  def can_implement(self, configuration, arguments):
    pass

  #
  def get_host_workspace_size(self, arguments):
    return 336

  #
  def get_device_workspace_size(self, arguments):
    return 0

  #
  def plan(self, arguments):
    grid = self.threadblock_swizzle.grid_tiled_shape(arguments.problem_size)
    return LaunchConfiguration([grid.m, grid.n, grid.k], [self.threads, 1, 1], self.shared_memory_capacity)

  #
  def initialize(self, host_workspace, device_workspace, launch_config, arguments, stream = cuda.CUstream(0)):
    
    offset = 0

    # Compute intermediate results
    swizzle_log_tile = 0
    gemm_mode = 0
    batch_count = 1
    gemm_k_size = arguments.problem_size.k

    # Pack into the host workspace buffer
    offset = arguments.problem_size.pack_into(host_workspace, offset)

    grid_tiled_shape = self.threadblock_swizzle.grid_tiled_shape(arguments.problem_size)
    offset = grid_tiled_shape.pack_into(host_workspace, offset)

    offset = PackInteger(host_workspace, offset, swizzle_log_tile)

    offset = self.params_A.initialize(host_workspace, offset, arguments.A.layout)
    offset = self.params_B.initialize(host_workspace, offset, arguments.B.layout)
    offset = self.params_C.initialize(host_workspace, offset, arguments.C.layout)
    offset = self.params_D.initialize(host_workspace, offset, arguments.D.layout)

    offset = self.output_op.initialize(host_workspace, offset, arguments.output_op)

    offset = PackInteger(host_workspace, offset, gemm_mode)
    offset = PackInteger(host_workspace, offset, batch_count)
    offset = PackInteger(host_workspace, offset, gemm_k_size)
    offset = PackDevicePointer(host_workspace, offset, int(arguments.A.pointer))
    offset = PackDevicePointer(host_workspace, offset, int(arguments.B.pointer))
    offset = PackDevicePointer(host_workspace, offset, int(arguments.C.pointer))
    offset = PackDevicePointer(host_workspace, offset, int(arguments.D.pointer))   

    return offset


#################################################################################################
#
# Module represents a compilation unit 
#
#################################################################################################

#
class CompilationOptions:
  '''
  Compilation options.
  '''

  #
  def __init__(self, architectures = [80], include_paths = []):
    self.includes = []
    self.include_paths = include_paths
    self.flags = ['-std=c++11', '-default-device']
    self.architectures = architectures

  #
  def get(self):
    options = []

    for flag in self.flags:
      options.append(bytes(str.encode(flag)))

    for incl in self.include_paths:
      options.append(bytes(str.encode('--include-path=%s' % incl)))

    arch_list = "-arch="
    for idx, arch in enumerate(self.architectures):
      if idx:
        arch_list += ","
      arch_list += "sm_%d" % arch

    options.append(bytes(str.encode(arch_list)))

    return options

IncludeTemplate = r'''#include "${include}"
'''

KernelTemplate = r'''
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  ${operation_name}${operation_suffix}::SharedStorage *shared_storage =
      reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);

  ${operation_name}${operation_suffix} op;

  op(params, *shared_storage);
}

'''

#
class Module:
  def __init__(self, name, operations, compilation_options):
    self.name = name
    self.operations = operations
    self.module = None
    self.log = None
    self.cubin_image = None
    self.source_buffer = ''

    #
    # Emit source
    #
    self.emit_()

    #
    # Compile
    #
    self.compile_(compilation_options)

    #
    # Load module
    #
    self.load_()
    
    # Done
    return

  # Emit a source buffer
  def emit_(self):

    # 1. Includes
    includes = []
    for operation in self.operations:
      for incl in operation.emitter.includes:
        if incl not in includes:
          includes.append(incl)

    for incl in includes:
      self.source_buffer += SubstituteTemplate(IncludeTemplate, { 'include': incl} )

    # 2. Operations
    for operation in self.operations:
      self.source_buffer += operation.emit()
      values = {
        'operation_name': operation.name(),
        'operation_suffix': operation.emitter.operation_suffix
      }
      self.source_buffer += SubstituteTemplate(KernelTemplate, values)

    # Done
    return

  # Compile with NVRTC
  def compile_(self, compilation_options):

    err, program = nvrtc.nvrtcCreateProgram(
      str.encode(self.source_buffer), 
      bytes(str.encode(self.name)), 
      0, [], [])

    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError('NVRTC Error: {}'.format(err))

    # Compile program
    options = compilation_options.get()

    err, = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:

      error_string = 'NVRTC Error: {}\n'.format(err)

      # Get log from compilation
      err, logSize = nvrtc.nvrtcGetProgramLogSize(program)
      if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
          raise RuntimeError('NVRTC Error: {}'.format(err))
      
      self.log = b' ' * logSize
      err, = nvrtc.nvrtcGetProgramLog(program, self.log)
      if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
          raise RuntimeError('NVRTC Error: {}'.format(err))
      
      raise RuntimeError(error_string + self.log.decode() + self.source_buffer)

    # Get data from compilation
    err, dataSize = nvrtc.nvrtcGetCUBINSize(program)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError('NVRTC Error: {}'.format(err))
    
    self.cubin_image = b' ' * dataSize
    err, = nvrtc.nvrtcGetCUBIN(program, self.cubin_image)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError('NVRTC Error: {}'.format(err))

    return
    
  #
  def load_(self):

    # Load data as module data
    err, self.module = cuda.cuModuleLoadData(self.cubin_image)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Cuda Error: {}'.format(err))
    
    # Get functions
    for operation in self.operations:
      err, operation.kernel = cuda.cuModuleGetFunction(
        self.module, 
        bytes(str.encode(operation.name())))

      if err != cuda.CUresult.CUDA_SUCCESS:
          raise RuntimeError('Cuda Error: {}'.format(err))

      operation.module = self

    return


#################################################################################################
#
# Manifest represents an 'owner' for modules and operations
#
#################################################################################################

#
class Manifest:

  #
  def __init__(self):
    self.operations = {}
    self.modules = []
    pass

  #
  def append_module(self, module):
    '''
    Appends a module and takes ownership of operations used to construct it.
    '''
    
    self.modules.append(module)

    for operation in module.operations:
      self.operations[operation.name()] = operation


#################################################################################################
