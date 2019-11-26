#include <iostream>
#include "cl_device_assist.cuh"
#include "cl_interface_shared.cuh"

extern "C" __global__ void vecadd(int* A, int* B, int* C, int nX, int nY, int nZ)
{
	int idx = get_global_id(0);

	C[idx] = A[idx] + B[idx];
}

void vecadd_launcher(kernel_descriptor* desc)
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

arg_descriptor make_mem_obj_arg() {
	arg_descriptor arg;

	arg.arg_type = ARG_TYPE_MEM_OBJ;

	return arg;
}

arg_descriptor vecadd_args[] = {
	make_mem_obj_arg(),
	make_mem_obj_arg(),
	make_mem_obj_arg()
};

kernel_descriptor kernels[] = {
	{
		"vecadd",
		3,
		vecadd_args,
		vecadd_launcher,
		0, 0, 0,
		0, 0, 0,
		0
	}
};

program_descriptor vectoradd_program = {
	1,
	kernels,
	"Hello World",
	"<opts>"
};
