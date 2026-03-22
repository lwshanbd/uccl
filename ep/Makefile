CUDA_PATH ?= /usr/local/cuda
override CXX := /usr/bin/g++
NVCC      := $(CUDA_PATH)/bin/nvcc

# --- Transport backend selection ---
# Set USE_LIBFABRIC=1 to build with libfabric/CXI instead of ibverbs.
USE_LIBFABRIC ?= 0

ifeq ($(USE_LIBFABRIC),1)
  FABRIC_HOME ?= /opt/cray/libfabric/2.2.0rc1
  FABRIC_CFLAGS := -DUSE_LIBFABRIC -I$(FABRIC_HOME)/include
  FABRIC_LDFLAGS := -L$(FABRIC_HOME)/lib64 -lfabric -Wl,-rpath,$(FABRIC_HOME)/lib64
  $(info Building with libfabric (CXI) from $(FABRIC_HOME))
  # No EFA/ibverbs when using libfabric
  EFA_CFLAGS :=
  EFA_LDFLAGS :=
else
  FABRIC_CFLAGS :=
  FABRIC_LDFLAGS :=

EFA_HOME?=/opt/amazon/efa
ifeq ($(wildcard $(EFA_HOME)),)
  $(warning EFA not detected, building without EFA)
  EFA_CFLAGS :=
  EFA_LDFLAGS :=
else
  $(info EFA detected, building with EFA)
  EFA_CFLAGS := -DEFA -I$(EFA_HOME)/include
  EFA_LDFLAGS := -L$(EFA_HOME)/lib -lefa
endif
endif

ARCH := $(shell uname -m)
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
CPU_IS_ARM64 := 0
GPU_IS_HOPPER := 0
ifeq ($(ARCH),aarch64)
  CPU_IS_ARM64 := 1
endif
ifneq (,$(findstring GH200,$(GPU_NAME)))
  GPU_IS_HOPPER := 1
endif
ifeq ($(and $(CPU_IS_ARM64),$(GPU_IS_HOPPER)),0)
  $(warning GH200 not detected, building without GH200)
  GH_CFLAGS :=
else
  $(info GH200 detected, building with GH200)
  GH_CFLAGS := -DUSE_GRACE_HOPPER
endif

# Nanobind configuration
PYTHON             ?= python3
NB_DIR             := $(shell $(PYTHON) -c "import nanobind as _nb, os; print(os.path.dirname(_nb.__file__))")
NB_INCLUDES        := -I$(NB_DIR)/include -I$(NB_DIR)/ext/robin_map/include \
                      -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
# Stable ABI requires Python >= 3.12
PY_STABLE_ABI_OK   := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
ifeq ($(PY_STABLE_ABI_OK),1)
NB_ABI_FLAGS       := -DPy_LIMITED_API=0x030C0000 -DNB_STABLE_ABI=1
EXT_SUFFIX         := .abi3.so
else
NB_ABI_FLAGS       :=
EXT_SUFFIX         := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
endif
NB_SRC_DIR         := $(NB_DIR)/src
NB_SRCS            := nb_internals.cpp nb_func.cpp nb_type.cpp nb_enum.cpp \
                      nb_ndarray.cpp nb_static_property.cpp common.cpp error.cpp \
                      trampoline.cpp implicit.cpp
NB_OBJECTS         := $(addprefix nb_,$(NB_SRCS:.cpp=.o))

TORCH_ROOT        := $(shell $(PYTHON) -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
TORCH_INCLUDE     := $(TORCH_ROOT)/include
TORCH_INCLUDE_API := $(TORCH_ROOT)/include/torch/csrc/api/include
TORCH_LIBDIR      := $(TORCH_ROOT)/lib
TORCH_ABI          := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INCS := -I$(TORCH_INCLUDE) -I$(TORCH_INCLUDE_API)

# Python installation path
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c "import site; print(site.getsitepackages()[0])")
INSTALL_DIR          := $(PYTHON_SITE_PACKAGES)/uccl

# GPU arch (auto-detected, override with: make SM=90)
DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM ?= $(DETECTED_SM)

CXXFLAGS  := -O3 -std=c++17 -Wall -pthread -fPIC -fvisibility=hidden -Wno-interference-size -Wno-unused-function -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
ifeq ($(USE_LIBFABRIC),1)
LDFLAGS := -lpthread $(FABRIC_LDFLAGS) -lnuma -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
else
LDFLAGS := -lpthread  -libverbs -lnl-3 -lnl-route-3 -lnuma -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64
endif
NVCCFLAGS := -O3 -std=c++17 -Xcompiler "-Wall -pthread -fPIC -fvisibility=hidden" -ccbin /usr/bin/g++ --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES := -Iinclude -I$(CUDA_PATH)/include -I/usr/include -I../include 

# Add Intel RDMA NIC support if USE_INTEL_RDMA_NIC=1
ifeq ($(USE_INTEL_RDMA_NIC),1)
  override CXXFLAGS += -DINTEL_RDMA_NIC
  override NVCCFLAGS += -DINTEL_RDMA_NIC
endif

# Enable per-expert batching path in dispatch-LL (default: disabled)
PER_EXPERT_BATCHING ?= 0
ifeq ($(PER_EXPERT_BATCHING),1)
  override CXXFLAGS += -DPER_EXPERT_BATCHING
  override NVCCFLAGS += -DPER_EXPERT_BATCHING
endif

CXXFLAGS  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS) $(FABRIC_CFLAGS)
NVCCFLAGS += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS) $(FABRIC_CFLAGS)
LDFLAGS   += $(EFA_LDFLAGS)
INCLUDES  += $(EFA_CFLAGS) $(GH_CFLAGS) $(NORMAL_CFLAGS) $(FABRIC_CFLAGS)

ifeq ($(USE_LIBFABRIC),1)
SRC_CPP := src/proxy.cpp src/fabric.cpp src/common.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/fifo.cpp
else
SRC_CPP := src/proxy.cpp src/rdma.cpp src/common.cpp src/uccl_proxy.cpp src/uccl_bench.cpp src/fifo.cpp
endif
SRC_CU  := src/bench_kernel.cu src/gpu_write_kernel.cu src/internode_ll.cu src/internode.cu src/layout.cu src/intranode.cu src/ep_runtime.cu

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(SRC_CU:.cu=.o)

SRC_BIND := src/uccl_ep.cc
OBJ_BIND := $(SRC_BIND:.cc=.o)
EP_EXT  := ep$(EXT_SUFFIX)
PYTARGET      := ep$(EXT_SUFFIX)

.PHONY: all py clean tests

all: $(EP_EXT)

# C++ compilation rule with dependency generation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(NB_ABI_FLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) -MMD -MP -c $< -o $@
# CUDA compilation rule with dependency generation
%.o: %.cu
	$(NVCC) -arch=sm_$(SM) $(NVCCFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

# Binding source compilation
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(NB_ABI_FLAGS) $(TORCH_INCS) $(INCLUDES) $(NB_INCLUDES) -MMD -MP -c $< -o $@

# Nanobind core sources (compiled with minimal flags)
nb_%.o: $(NB_SRC_DIR)/%.cpp
	$(CXX) -O3 -std=c++17 -fPIC -fvisibility=hidden $(NB_ABI_FLAGS) $(NB_INCLUDES) -c $< -o $@

$(EP_EXT): $(OBJ_CPP) $(OBJ_CU) $(OBJ_BIND) $(NB_OBJECTS)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
	    $(INCLUDES) $(NB_INCLUDES) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
	    -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    -lcuda -lcudart $(LDFLAGS) -o $@

py: $(PYTARGET) $(EP_EXT)

# Repo uccl package dir (so in-repo runs use the same build)
UCCL_UCCL := $(abspath $(CURDIR)/../uccl/uccl)

install: $(EP_EXT)
	@mkdir -p $(INSTALL_DIR)
	@rm -f $(INSTALL_DIR)/ep.cpython-*.so $(INSTALL_DIR)/ep.abi3.so
	@cp $(EP_EXT) $(INSTALL_DIR)/
	@echo "Installed $(EP_EXT) -> $(INSTALL_DIR)/"
	@if [ -d "$(UCCL_UCCL)" ]; then \
		rm -f "$(UCCL_UCCL)"/ep.cpython-*.so "$(UCCL_UCCL)"/ep.abi3.so; \
		cp $(EP_EXT) "$(UCCL_UCCL)/"; \
		echo "Installed $(EP_EXT) -> $(UCCL_UCCL)/ (for in-repo runs)"; \
	fi

# ---------------------------------------------------------------------------
# Test targets (standalone executables, linked with MPI)
# ---------------------------------------------------------------------------

# Object files needed by tests (subset of the main library objects).
TEST_OBJ_CPP := $(OBJ_CPP)
TEST_OBJ_CU  := $(OBJ_CU)
TEST_LDFLAGS := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 \
                -lcuda -lcudart $(LDFLAGS)

# uccl_comm objects (always compiled with fabric flags etc.)
src/uccl_comm.o: src/uccl_comm.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

tests/test_gpu_proxy_fabric.o: tests/test_gpu_proxy_fabric.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

test_gpu_proxy_fabric: tests/test_gpu_proxy_fabric.o src/uccl_comm.o $(TEST_OBJ_CPP) $(TEST_OBJ_CU)
	$(CXX) $(CXXFLAGS) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    $(TEST_LDFLAGS) -o $@

tests/test_uccl_comm.o: tests/test_uccl_comm.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

test_uccl_comm: tests/test_uccl_comm.o src/uccl_comm.o $(TEST_OBJ_CPP) $(TEST_OBJ_CU)
	$(CXX) $(CXXFLAGS) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    $(TEST_LDFLAGS) -o $@

tests/test_fabric_alltoall_v2.o: tests/test_fabric_alltoall_v2.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_INCS) $(INCLUDES) -MMD -MP -c $< -o $@

test_fabric_alltoall_v2: tests/test_fabric_alltoall_v2.o src/uccl_comm.o $(TEST_OBJ_CPP) $(TEST_OBJ_CU)
	$(CXX) $(CXXFLAGS) $^ \
	    -L$(TORCH_LIBDIR) -Wl,-rpath,$(TORCH_LIBDIR) \
	    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
	    $(TEST_LDFLAGS) -o $@

tests: test_gpu_proxy_fabric test_uccl_comm test_fabric_alltoall_v2

# Clean all generated files
clean:
	rm -f $(OBJ_CPP) $(OBJ_CC) $(OBJ_CU) $(OBJ_BIND) $(NB_OBJECTS) \
	      $(EP_EXT) src/uccl_comm.o \
	      tests/*.o test_gpu_proxy_fabric test_uccl_comm test_fabric_alltoall_v2 \
	      *.d src/*.d bench/*.d src/*.o **/*.hip **/*_hip.*

# Automatically include dependency files if they exist
DEPS := $(OBJ_CPP:.o=.d) $(OBJ_CC:.o=.d) $(OBJ_CU:.o=.d) $(OBJ_BIND:.o=.d)
-include $(DEPS)
