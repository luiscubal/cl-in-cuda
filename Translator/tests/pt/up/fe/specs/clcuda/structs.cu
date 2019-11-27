#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

struct clcuda_type_array
{
	float *field_data;
	size_t field_dim1;
};

__global__ void clcuda_func_structs(float *var_data, int var_dim1, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;

	struct clcuda_type_array var_arr;
	var_arr.field_data = var_data;
	var_arr.field_dim1 = var_dim1;
	(var_arr.field_data[var_arr.field_dim1 - get_global_id(0, data) - 1]) = get_global_id(0, data);
}

KERNEL_LAUNCHER void clcuda_launcher_structs(struct _cl_kernel *desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);

	clcuda_func_structs<<<num_grids, local_size>>>(
		(float*) desc->arg_data[0],
		*(int*) desc->arg_data[1],
		CommonThreadData(desc->totalX, desc->totalY, desc->totalZ)
	);
}

