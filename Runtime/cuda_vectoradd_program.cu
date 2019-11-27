#include "cl_device_assist.cuh"
#include "cl_interface_shared.h"

__global__ void vecadd(int* A, int* B, int* C, int nX, int nY, int nZ)
{
	int idx = get_global_id(0);

	C[idx] = A[idx] + B[idx];
}

KERNEL_LAUNCHER void vecadd_launcher(struct _cl_kernel* desc)
{
	dim3 num_grids = dim3(desc->gridX, desc->gridY, desc->gridZ);
	dim3 local_size = dim3(desc->localX, desc->localY, desc->localZ);

	vecadd<<<num_grids, local_size>>>(
		(int*) desc->arg_data[0],
		(int*) desc->arg_data[1],
		(int*) desc->arg_data[2],
		desc->totalX,
		desc->totalY,
		desc->totalZ
	);
}
