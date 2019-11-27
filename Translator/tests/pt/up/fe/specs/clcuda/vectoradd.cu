#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_vectoradd(int *var_A, int *var_B, int *var_C, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;

	size_t var_i = get_global_id(0, data);
	(var_C[var_i]) = ((var_A[var_i]) + (var_B[var_i]));
}

KERNEL_LAUNCHER void clcuda_launcher_vectoradd(struct _cl_kernel *desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);

	clcuda_func_vectoradd<<<num_grids, local_size>>>(
		(int*) desc->arg_data[0],
		(int*) desc->arg_data[1],
		(int*) desc->arg_data[2],
		CommonThreadData(desc->totalX, desc->totalY, desc->totalZ)
	);
}
