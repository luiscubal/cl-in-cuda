#ifndef CL_DEVICE_ASSIST_
#define CL_DEVICE_ASSIST_

#include <device_launch_parameters.h>

#define global
#define __attribute__(x)

typedef unsigned int uint;

static inline __device__ size_t get_local_id(size_t dim) {
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

static inline __device__ size_t get_local_size(size_t dim) {
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

static inline __device__ size_t get_global_id(size_t dim) {
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

static inline __device__ size_t get_group_id(size_t dim) {
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

static inline __device__ size_t get_num_groups(size_t dim) {
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

#endif
