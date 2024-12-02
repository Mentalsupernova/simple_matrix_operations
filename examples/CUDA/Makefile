# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS =  -lm  -ldl -O3
CUDAFLAGS = -I/usr/local/cuda/include  -O3
LDFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

# Files
TARGET = main
SRCS = main.c
CUDA_SRCS = ../../cuda_kernels/nvidia_gpu_dot.cu ../../cuda_kernels/nvidia_gpu_sub.cu ../../cuda_kernels/nvidia_gpu_add.cu
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(CUDA_OBJS)

.PHONY: all clean
