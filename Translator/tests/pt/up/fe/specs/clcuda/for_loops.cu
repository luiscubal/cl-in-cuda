#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_for_loops(int *var_A, int var_N, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;

	for (int var_i = 0; var_i < var_N; var_i++)
	{
		var_A[(var_N * clcuda_builtin_get_global_size(0, data)) + clcuda_builtin_get_global_id(0, data)] = 1;
	}
}

KERNEL_LAUNCHER void clcuda_launcher_for_loops(struct _cl_kernel *desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);

	clcuda_func_for_loops<<<num_grids, local_size>>>(
		(int*) desc->arg_data[0],
		*(int*) desc->arg_data[1],
		CommonThreadData(desc->totalX, desc->totalY, desc->totalZ)
	);
}

