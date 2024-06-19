export MACA_PATH=${1}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib
export PATH=$CUCC_PATH/tools:${CUCC_PATH}/bin:${CUDA_PATH}/bin:$PATH
