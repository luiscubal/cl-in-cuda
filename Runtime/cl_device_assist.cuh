#ifndef CL_DEVICE_ASSIST_
#define CL_DEVICE_ASSIST_

#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#define KERNEL_LAUNCHER extern "C" __declspec(dllexport)
#else
#define KERNEL_LAUNCHER extern "C"
#endif
#define FULL_MASK 0xffffffff

struct CommonKernelData {
	size_t totalX, totalY, totalZ;

	CommonKernelData(size_t totalX, size_t totalY, size_t totalZ)
		: totalX(totalX), totalY(totalY), totalZ(totalZ)
	{}
};

typedef unsigned char uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef uint64_t ulong;
typedef uint64_t cl_mem_fence_flags;

static __device__ __forceinline__ float clcuda_builtin_log(float x, CommonKernelData data) {
	return logf(x);
}

static __device__ __forceinline__ double clcuda_builtin_log(double x, CommonKernelData data) {
	return log(x);
}

static __device__ __forceinline__ float clcuda_builtin_exp(float x, CommonKernelData data) {
	return expf(x);
}

static __device__ __forceinline__ double clcuda_builtin_exp(double x, CommonKernelData data) {
	return exp(x);
}

static __device__ __forceinline__ float clcuda_builtin_sqrt(float x, CommonKernelData data) {
	return sqrtf(x);
}

static __device__ __forceinline__ double clcuda_builtin_sqrt(double x, CommonKernelData data) {
	return sqrt(x);
}

static __device__ __forceinline__ float clcuda_builtin_sin(float x, CommonKernelData data) {
	return sinf(x);
}

static __device__ __forceinline__ double clcuda_builtin_sin(double x, CommonKernelData data) {
	return sin(x);
}

static __device__ __forceinline__ float clcuda_builtin_cos(float x, CommonKernelData data) {
	return cosf(x);
}

static __device__ __forceinline__ double clcuda_builtin_cos(double x, CommonKernelData data) {
	return cos(x);
}

static __device__ __forceinline__ size_t clcuda_builtin_get_local_id(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ size_t clcuda_builtin_get_local_size(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ size_t clcuda_builtin_get_global_id(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ size_t clcuda_builtin_get_global_size(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ size_t clcuda_builtin_get_group_id(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ size_t clcuda_builtin_get_num_groups(uint dim, CommonKernelData data) {
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

static __device__ __forceinline__ void clcuda_builtin_barrier(cl_mem_fence_flags flags, CommonKernelData data) {
	__syncthreads();
}

static __device__ float clcuda_builtin_sub_group_reduce_add(float x, CommonKernelData data) {
	// Assuming warps of 32 lanes
	float acc = x;
	acc += __shfl_down_sync(FULL_MASK, acc, 1);
    acc += __shfl_down_sync(FULL_MASK, acc, 2);
    acc += __shfl_down_sync(FULL_MASK, acc, 4);
    acc += __shfl_down_sync(FULL_MASK, acc, 8);
    acc += __shfl_down_sync(FULL_MASK, acc, 16);
    return acc;
}

static __device__ float clcuda_builtin_work_group_reduce_add(float x, CommonKernelData data) {
	// Assuming 1024 is the maximum block size, and each warp has 32 lanes
	__shared__ float acc[1024 / 32];
	float warp_sum = clcuda_builtin_sub_group_reduce_add(x, data);

	size_t flatThreadIdx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	size_t lane = flatThreadIdx & 0x1F;
	size_t warp = flatThreadIdx >> 5;
	if (lane == 0) {
		acc[warp] = warp_sum;
	}

	__syncthreads();

	size_t numThreads = blockDim.x * blockDim.y * blockDim.z;
	size_t numWarps = numThreads / 32;
	float totalAcc;
	if (warp == 0) {
		float per_thread_acc = 0;
		for (int i = lane; i < numWarps; i += 32) {
			per_thread_acc += acc[i];
		}
		totalAcc = clcuda_builtin_sub_group_reduce_add(per_thread_acc, data);

		if (lane == 0) {
			acc[0] = totalAcc;
		}
	}

	__syncthreads();

	return acc[0];
}

#endif
