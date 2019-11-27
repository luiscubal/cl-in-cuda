#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void clcuda_func_dynamic_local_mem(size_t clcuda_offset_A, size_t clcuda_offset_B, CommonKernelData data)
{
	if (blockIdx.x * blockDim.x + threadIdx.x >= data.totalX) return;
	if (blockIdx.y * blockDim.y + threadIdx.y >= data.totalY) return;
	if (blockIdx.z * blockDim.z + threadIdx.z >= data.totalZ) return;
	
	extern __shared__ char local_mem[];
	int *var_A = (int*) (local_mem + clcuda_offset_A);
	double *var_B = (double*) (local_mem + clcuda_offset_B);
	
	var_A[clcuda_builtin_get_local_id(0, data)] = var_B[clcuda_builtin_get_local_id(0, data)];
}

KERNEL_LAUNCHER void clcuda_launcher_dynamic_local_mem(struct _cl_kernel *desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);
	
	size_t local_mem_size = 0;
	size_t clcuda_offset_A = local_mem_size;
	local_mem_size += desc->arg_data[0];
	local_mem_size = ((local_mem_size + 7) / 8) * 8;
	size_t clcuda_offset_B = local_mem_size;
	local_mem_size += desc->arg_data[1];
	
	clcuda_func_dynamic_local_mem<<<num_grids, local_size, local_mem_size>>>(
		clcuda_offset_A,
		clcuda_offset_B,
		CommonThreadData(desc->totalX, desc->totalY, desc->totalZ)
	);
}

