
# System modules
import numpy as np
import os.path
import sys
import ctypes

# CUDA Python modules
from cuda import cuda
from cuda import nvrtc

# CUTLASS modules
import library
import manifest as cutlass_manifest
import generator
import rt


#
# Construct an SGEMM
#

manifest = cutlass_manifest.Manifest()

generator.GenerateSM50_Simt(manifest, "11.5.0")

#
# Construct a GEMM operation
#

operation = manifest.operations_by_name['cutlass_simt_sgemm_128x128_8x2_nt_align1']

#
# Construct a runtime GEMM operation
#
gemm = rt.Gemm(operation)

#
# Initialize context
#
err, = cuda.cuInit(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, device = cuda.cuDeviceGet(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, context = cuda.cuCtxCreate(0, device)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

#
# Construct a module
#

architectures = [80,]
include_paths = [
  '../../include',
  '../../tools/util/include',
]

compilation_options = rt.CompilationOptions(architectures, include_paths)

module = rt.Module('module.cu', [gemm], compilation_options)

#
# Setup a workspace
#

M, N, K = (128, 128, 128)

tensor_A = np.ndarray(M * K, dtype=np.float32)
tensor_B = np.ndarray(N * K, dtype=np.float32)
tensor_C = np.ndarray(M * N, dtype=np.float32)
tensor_D = np.ndarray(M * N, dtype=np.float32)

err, tensor_A_d = cuda.cuMemAlloc(tensor_A.size * tensor_A.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_B_d = cuda.cuMemAlloc(tensor_B.size * tensor_B.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_C_d = cuda.cuMemAlloc(tensor_C.size * tensor_C.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_D_d = cuda.cuMemAlloc(tensor_D.size * tensor_D.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, stream = cuda.cuStreamCreate(0)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

tensors = [
  (tensor_A_d, tensor_A),
  (tensor_B_d, tensor_B),
  (tensor_C_d, tensor_C),
  (tensor_D_d, tensor_D)
]

for tensor_device, tensor_host in tensors:
  bytes = tensor_host.size * tensor_host.itemsize
  print("Tensor has dimensions: %s (%d bytes)" % (str(tensor_host.size), tensor_host.itemsize))
  err, = cuda.cuMemcpyHtoDAsync(tensor_device, tensor_host, bytes, stream)
  print("updating tensor in device memory ", hex(int(tensor_device)))
  if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError('CUDA Error %s' % str(err))

#
# Initialize a host buffer
#

arguments = rt.GemmArguments()

arguments.problem_size = rt.GemmCoord(M, N, K)

arguments.A = rt.TensorRef(tensor_A_d, M)
arguments.B = rt.TensorRef(tensor_B_d, N)
arguments.C = rt.TensorRef(tensor_C_d, M)
arguments.D = rt.TensorRef(tensor_D_d, M)

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

#
# Launch the kernel
#

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))

#
# Verify results
#
err, = cuda.cuStreamSynchronize(stream)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))


#
# Debug reporting of byte array contents
#

def PrintBytearray(host_workspace):
  uint_str = None
  prefix = None
  print("uint32_t host_workspace[] = {")
  for idx, byte in enumerate(host_workspace):
    if not (idx % 4):
      if uint_str is not None:
        print(prefix, uint_str, ",")
      prefix = "/* offset: %d B */    0x" % idx
      uint_str = ""
    uint_str = "{:02x}".format(byte) + uint_str
  print("};")
