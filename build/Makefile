
ifneq ($(cuda_build),yes)
  cuda_build := "no"
endif

ifneq ($(raja_cuda_build),yes)
	cuda_build := "no"
endif

ifneq ($(raja_omp_build),yes)
	omp_build := "no"
endif

DFLAGS = -DSOA

ifeq ($(cuda_build),yes)
	export NVCC:=$(shell which nvcc)
	CXX = $(NVCC)
	OMPOPT =
	LINKER = CC

  	CUDA_HOME:=$(patsubst %/compilers/bin/nvcc,%/cuda,$(NVCC))
	CUDA_LIBS = -lcudart $(addprefix -L, $(wildcard $(CUDA_HOME)/lib64))
	
	DFLAGS   += -DUSE_GPU
	INCLUDE  += -I$(CUDA_DIR)/include/ -I$(MPICH_DIR)/include
	OPT = -O3 -arch=sm_80 --extended-lambda -I$(CUDA_HOME)/include -I$(MPICH_DIR)/include --ptxas-options=-v -lineinfo --x cu -DUSE_CUDA
	LINKFLAGS = -O3 -lmpi_gtl_cuda -lcudart -lcudadevrt -L$(CUDA_LIBS)
    CXXFLAGS = $(DFLAGS) $(OPT) $(INCLUDE) -g -std=c++17

	CPPSTDFLAG=-std=c++17
	NVCCFLAGS=-arch=sm_80 $(CPPSTDFLAG) $(INCLUDE) --ptxas-options=-v -lineinfo $(DFLAGS)
	
else ifeq ($(catalyst_build),yes)
	export NVCC:=$(shell which nvcc)
	CXX = $(NVCC)
	OMPOPT =
	LINKER = CC

  	CUDA_HOME:=$(patsubst %/compilers/bin/nvcc,%/cuda,$(NVCC))
	CUDA_LIBS = -lcudart $(addprefix -L, $(wildcard $(CUDA_HOME)/lib64))
	DFLAGS   += -DUSE_GPU
	INCLUDE  += -I$(CUDA_DIR)/include/ -I$(MPICH_DIR)/include
	OPT = -O3 -arch=sm_80 --extended-lambda -I$(CUDA_HOME)/include -I$(MPICH_DIR)/include --ptxas-options=-v -lineinfo --x cu -DUSE_CUDA
	
	CATALYST_PREFIX=/lus/eagle/projects/multiphysics_aesp/azy4/Catalyst_W_Python/catalyst-install
    CATALYST_INCLUDE=$(CATALYST_PREFIX)/include/catalyst-2.0/
    CATALYST_LIBS=$(CATALYST_PREFIX)/lib64/
	LD_LIBRARY_PATH+=/lus/eagle/projects/multiphysics_aesp/azy4/Catalyst_W_Python/catalyst-install/lib64/
	LINKFLAGS = -O3 -L$(CUDA_LIBS) -L$(CATALYST_LIBS) -lmpi_gtl_cuda -lcudart -lcudadevrt -lcatalyst
	CXXFLAGS = $(DFLAGS) $(OPT) $(INCLUDE) -I$(CATALYST_INCLUDE) -g -std=c++17

	CPPSTDFLAG=-std=c++17
	NVCCFLAGS=-arch=sm_80 $(CPPSTDFLAG) $(INCLUDE) --ptxas-options=-v -lineinfo $(DFLAGS)
	

else ifeq ($(raja_cuda_build),yes)
	CXX = NVCC
	OMPOPT =
	RAJA_LOCATION=/ccs/proj/cfd136/gounley1/RAJA/build-200319cuda
	LINKER = mpicxx

	CUDA_HOME=/sw/summit/cuda/10.1.243
	CUDA_LIBS = $(CUDA_HOME)/lib64
	OPT = -O3 -dc -ccbin mpicxx -Xcompiler="" --expt-extended-lambda -restrict -arch=sm_70 -I$(RAJA_LOCATION)/include --x cu -DUSERAJA_CUDA
	LINKFLAGS = -O3 -L $(RAJA_LOCATION)/lib -lRAJA -lmpi_ibm -lcudart -lcudadevrt -L$(CUDA_LIBS)
	CXXFLAGS = $(DFLAGS) $(OPT) $(INCLUDE) -g -std=c++11
else ifeq ($(omp_build),yes)
	CXX = mpicxx
	OMPOPT = -fopenmp
	RAJA_LOCATION=/ccs/proj/cfd136/gounley1/RAJA/build-200319omp
	LINKER = mpicxx

	CXXFLAGS = $(INCLUDE) -g -std=c++11 -O3 -Wall $(OMPOPT) -I$(RAJA_LOCATION)/include
	CXXFLAGS += $(DFLAGS) -DUSERAJA_OMP
	LINKFLAGS = $(CXXFLAGS) -L $(RAJA_LOCATION)/lib -lRAJA
else
	CXX = mpicxx
	CXXFLAGS = $(INCLUDE) -g -std=c++11 -O3 -Wall
	# CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address
	CXXFLAGS += $(DFLAGS)
	LINKFLAGS = $(CXXFLAGS)
endif

OBJS =	Main.o \
			Communication.o \
			Parameters.o \
			Geometry.o GeometryAAAB.o \
			LatticeD3Q19.o LatticeD3Q27.o LatticeD3Q39.o \
			Propagation.o \
			XletsD3Q19.o XletsRegularized.o
ifeq ($(cuda_build),yes)
	OBJS += 	CommUtilsCUDA.o \
				KernelAA_SOA_CUDA.o KernelAA_SOA_CUDA_39.o \
				KernelAB_SOA_CUDA.o KernelAB_SOA_CUDA_39.o \
				PropAA_CUDA.o PropAB_CUDA.o \
				XletsD3Q19_CUDA.o
else ifeq ($(catalyst_build),yes)
	OBJS += 	CommUtilsCUDA.o \
				KernelAA_SOA_CUDA.o KernelAA_SOA_CUDA_39.o \
				KernelAB_SOA_CUDA.o KernelAB_SOA_CUDA_39.o \
				PropAA_CUDA.o PropAB_CUDA.o CatalystAdaptor.o \
				XletsD3Q19_CUDA.o
else
	OBJS += KernelAA_SOA_Base.o KernelAA_AOS_Base.o \
      KernelAA_SOA_Unroll.o KernelAA_SOA_Unroll_39.o \
      KernelAB_SOA_Base.o KernelAB_AOS_Base.o \
      KernelAB_SOA_Pull.o \
      KernelAB_SOA_Unroll.o KernelAB_SOA_Unroll_Pull.o \
      KernelAB_SOA_Unroll_39.o KernelAB_SOA_Unroll_Pull_39.o \
		PropagationAA.o PropagationAB.o
endif

INCLUDE = -I ../src -I ../src/geometries -I ../src/kernels -I ../src/lattices -I ../src/propagation -I ../src/xlets

builddir = test_build
FOBJS = $(addprefix $(builddir)/,$(OBJS))

lbm-proxy-app: $(FOBJS)
ifeq ($(cuda_build),yes)
	cd $(builddir); $(CXX) -dlink -arch=sm_80 -o file_link.o $(OBJS) $(LINKFLAGS)
	cd $(builddir); $(LINKER) -o $@ file_link.o $(OBJS) $(LINKFLAGS)
else ifeq ($(catalyst_build),yes)
	cd $(builddir); $(CXX) -dlink -arch=sm_80 -o file_link.o $(OBJS) $(LINKFLAGS)
	cd $(builddir); $(LINKER) -o $@ file_link.o $(OBJS) $(LINKFLAGS)
else
	cd $(builddir); $(CXX) -o $@ $(OBJS) $(LINKFLAGS)
endif

clean:
	/bin/mkdir -p $(builddir)
	cd $(builddir); /bin/rm -f lbm-proxy-app *.o

$(builddir)/%.o:src/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/%.cu
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(NVCC) -lineinfo $(NVCCFLAGS) $(INCLUDE) -c ../$<

$(builddir)/%.o:src/geometries/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/kernels/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/kernels/%.cu
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(NVCC) -lineinfo $(NVCCFLAGS) $(INCLUDE) -c ../$<

$(builddir)/%.o:src/lattices/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/propagation/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/xlets/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<

$(builddir)/%.o:src/xlets/%.cu
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(NVCC) -lineinfo $(NVCCFLAGS) $(INCLUDE) -c ../$<


## $(NVCC) -lineinfo $(NVCCFLAGS) $(INCLUDE) -c -o $@ $<
