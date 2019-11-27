#ifndef CL_DEVICE_ASSIST_
#define CL_DEVICE_ASSIST_

#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

#define KERNEL_LAUNCHER extern "C" __declspec(dllexport)

struct CommonKernelData {
	size_t totalX, totalY, totalZ;

	CommonKernelData(size_t totalX, size_t totalY, size_t totalZ)
		: totalX(totalX), totalY(totalY), totalZ(totalZ)
	{}
};

typedef uint32_t uint;
typedef uint64_t ulong;
typedef uint64_t cl_mem_fence_flags;

static inline __device__ float clcuda_builtin_log(float x, CommonKernelData data) {
	return logf(x);
}

static inline __device__ double clcuda_builtin_log(double x, CommonKernelData data) {
	return log(x);
}

static inline __device__ float clcuda_builtin_exp(float x, CommonKernelData data) {
	return expf(x);
}

static inline __device__ double clcuda_builtin_exp(double x, CommonKernelData data) {
	return exp(x);
}

static inline __device__ float clcuda_builtin_sqrt(float x, CommonKernelData data) {
	return sqrtf(x);
}

static inline __device__ double clcuda_builtin_sqrt(double x, CommonKernelData data) {
	return sqrt(x);
}

static inline __device__ float clcuda_builtin_sin(float x, CommonKernelData data) {
	return sinf(x);
}

static inline __device__ double clcuda_builtin_sin(double x, CommonKernelData data) {
	return sin(x);
}

static inline __device__ float clcuda_builtin_cos(float x, CommonKernelData data) {
	return cosf(x);
}

static inline __device__ double clcuda_builtin_cos(double x, CommonKernelData data) {
	return cos(x);
}

static inline __device__ size_t clcuda_builtin_get_local_id(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return threadIdx.x;
	case 1:
		return threadIdx.y;
	case 2:
		return threadIdx.z;
	default:
		return 0;
	}
}

static inline __device__ size_t clcuda_builtin_get_local_size(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return blockDim.x;
	case 1:
		return blockDim.y;
	case 2:
		return blockDim.z;
	default:
		return 1;
	}
}

static inline __device__ size_t clcuda_builtin_get_global_id(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return blockIdx.x * blockDim.x + threadIdx.x;
	case 1:
		return blockIdx.y * blockDim.y + threadIdx.y;
	case 2:
		return blockIdx.z * blockDim.z + threadIdx.z;
	default:
		return 0;
	}
}

static inline __device__ size_t clcuda_builtin_get_global_size(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return data.totalX;
	case 1:
		return data.totalY;
	case 2:
		return data.totalZ;
	default:
		return 1;
	}
}

static inline __device__ size_t clcuda_builtin_get_group_id(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return blockIdx.x;
	case 1:
		return blockIdx.y;
	case 2:
		return blockIdx.z;
	default:
		return 0;
	}
}

static inline __device__ size_t clcuda_builtin_get_num_groups(uint dim, CommonKernelData data) {
	switch (dim) {
	case 0:
		return gridDim.x;
	case 1:
		return gridDim.y;
	case 2:
		return gridDim.z;
	default:
		return 1;
	}
}

static inline __device__ void clcuda_builtin_barrier(cl_mem_fence_flags flags, CommonKernelData data) {
	__syncthreads();
}

#endif
