#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_multidecl(
	CommonKernelData data
) {
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;
	
	int32_t var_x = 0;
	int32_t *var_y = &var_x;
	int32_t var_z;
	{
		int32_t var_k = 1;
		int32_t var_j = 2;
		int32_t *var_w = &var_k;
		int32_t var_m;
		for (; ; var_k++)
		{
		}
	}
}

KERNEL_LAUNCHER void clcuda_launcher_multidecl(
	struct _cl_kernel *desc,
	float *elapsedMs
) {
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaEventRecord(start);
	clcuda_func_multidecl<<<num_grids, local_size>>>(
		CommonKernelData(desc->totalX, desc->totalY, desc->totalZ)
	);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(elapsedMs, start, end);
}

