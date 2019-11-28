#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_modifiers(int32_t const * __restrict__ var_A, int32_t const * __restrict__ var_B, int32_t * __restrict__ var_C, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;
	
	size_t var_i = clcuda_builtin_get_global_id(0, data);
	var_C[var_i] = (var_A[var_i] + var_B[var_i]);
}

KERNEL_LAUNCHER void clcuda_launcher_modifiers(struct _cl_kernel *desc, float *elapsedMs)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaEventRecord(start);
	clcuda_func_modifiers<<<num_grids, local_size>>>(
		(int32_t const* __restrict__) desc->arg_data[0],
		(int32_t const* __restrict__) desc->arg_data[1],
		(int32_t* __restrict__) desc->arg_data[2],
		CommonKernelData(desc->totalX, desc->totalY, desc->totalZ)
	);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(elapsedMs, start, end);
}

