#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_manualbound(float *var_A, int var_N, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;
	
	size_t var_global_id;
	var_global_id = clcuda_builtin_get_global_id(0U, data);
	if (var_global_id < var_N)
	{
		var_A[var_global_id] = 1.0;
	}
}

KERNEL_LAUNCHER void clcuda_launcher_manualbound(struct _cl_kernel *desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);
	
	clcuda_func_manualbound<<<num_grids, local_size>>>(
		(float*) desc->arg_data[0],
		*(int*) desc->arg_data[1],
		CommonThreadData(desc->totalX, desc->totalY, desc->totalZ)
	);
}

