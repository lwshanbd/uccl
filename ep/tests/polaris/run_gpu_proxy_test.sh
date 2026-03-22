#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -A diomp
#PBS -l filesystems=home:eagle:grand
#PBS -N test_gpu_proxy
#PBS -j oe
#PBS -o /lus/eagle/projects/diomp/baodi/uccl/ep/tests/polaris/

set -e
cd /lus/eagle/projects/diomp/baodi/uccl/ep
module use /soft/modulefiles
module load cudatoolkit-standalone/12.8.1

CUDA_PATH=${CUDA_HOME}
NVCC=${CUDA_PATH}/bin/nvcc
FABRIC_HOME=/opt/cray/libfabric/2.2.0rc1
CXX=/usr/bin/g++
MPI_DIR=$(dirname $(dirname $(which mpicc)))
FLAGS="-O0 -g -std=c++17 -Wno-interference-size -Wno-unused-function"
INC="-I${FABRIC_HOME}/include -Iinclude -I${CUDA_PATH}/include -I../include -I${MPI_DIR}/include"
LIB="-L${CUDA_PATH}/lib64 -lcudart -lcuda -L${MPI_DIR}/lib -lmpi -lpthread -lrt -Wl,-rpath,${CUDA_PATH}/lib64"

# Detect GPU arch
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

echo "=== Environment ==="
echo "CUDA_PATH=${CUDA_PATH}"
echo "FABRIC_HOME=${FABRIC_HOME}"
echo "SM=${SM}"
echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo ""

# --- Step 1: Compile CUDA kernel with nvcc ---
echo "=== Compiling gpu_write_kernel.cu ==="
${NVCC} -arch=sm_${SM} -O0 -g -std=c++17 \
  -DUSE_LIBFABRIC \
  -Xcompiler "-Wall -fPIC -Wno-interference-size" \
  -Iinclude -I${CUDA_PATH}/include -I../include \
  -c src/gpu_write_kernel.cu -o src/gpu_write_kernel.o 2>&1
echo "OK"

# --- Step 2: Compile and link the test ---
# Ring buffer path (no USE_MSCCLPP_FIFO_BACKEND) so GPU can push to
# DeviceToHostCmdBuffer via atomic_set_and_commit.
echo "=== Compiling test_gpu_proxy_fabric ==="
${CXX} ${FLAGS} -DUSE_LIBFABRIC \
  ${INC} \
  tests/test_gpu_proxy_fabric.cpp \
  src/uccl_comm.cpp src/fabric.cpp src/common.cpp \
  src/uccl_proxy.cpp src/proxy.cpp src/fifo.cpp \
  src/gpu_write_kernel.o \
  -L${FABRIC_HOME}/lib64 -lfabric -Wl,-rpath,${FABRIC_HOME}/lib64 \
  ${LIB} -o tests/test_gpu_proxy_fabric 2>&1
echo "Build OK"
echo ""

# --- Step 3: Run tests ---

echo "========== GPU-Proxy P2P + A2A: 2 ranks (1/node) 1 MiB =========="
mpiexec -n 2 --ppn 1 \
  --env FI_CXI_DEFAULT_CQ_SIZE=131072 \
  --env FI_CXI_RX_MATCH_MODE=hybrid \
  --env CUDA_LAUNCH_BLOCKING=1 \
  ./tests/test_gpu_proxy_fabric 1048576 2>&1
echo ""

# Multi-rank tests disabled during debugging
# echo "========== GPU-Proxy P2P + A2A: 8 ranks (4/node) 1 MiB =========="
# mpiexec -n 8 --ppn 4 ...


echo "=== All tests complete ==="
