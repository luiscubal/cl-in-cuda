#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__device__ int32_t clcuda_func_apply(
	int32_t var_x,
	CommonKernelData data
) {
	return var_x * 2;
}

__global__ void clcuda_func_auxfunc(
	int32_t *var_A,
	int32_t *var_B,
	CommonKernelData data
) {
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;
	
	size_t var_i = clcuda_builtin_get_global_id(0, data);
	var_A[var_i] = clcuda_func_apply(var_B[var_i], data);
}

KERNEL_LAUNCHER void clcuda_launcher_auxfunc(
	struct _cl_kernel *desc,
	float *elapsedMs
) {
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaEventRecord(start);
	clcuda_func_auxfunc<<<num_grids, local_size>>>(
		(int32_t*) desc->arg_data[0],
		(int32_t*) desc->arg_data[1],
		CommonKernelData(desc->totalX, desc->totalY, desc->totalZ)
	);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(elapsedMs, start, end);
}

