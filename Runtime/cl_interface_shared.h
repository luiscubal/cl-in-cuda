#ifndef CL_INTERFACE_SHARED_CUH_
#define CL_INTERFACE_SHARED_CUH_

#include <stddef.h>

#define ARG_TYPE_SCALAR 1
#define ARG_TYPE_LOCAL_MEM 2
#define ARG_TYPE_GLOBAL_MEM 3

struct kernel_descriptor;

typedef void (*kernel_launcher)(struct _cl_kernel* desc, float* elapsedMs);

struct arg_descriptor {
	int arg_type;
	union {
		size_t scalar_size;
	} data;
};

typedef struct _cl_kernel {
	char* name;
	char* symbol_name;
	size_t num_args;
	arg_descriptor* arg_descriptors;
	kernel_launcher launcher;

	size_t gridX, gridY, gridZ;
	size_t localX, localY, localZ;
	size_t totalX, totalY, totalZ;
	void** arg_data;
} _cl_kernel;

typedef struct _cl_program {
	size_t num_kernels;
	_cl_kernel* kernels;
	char* build_log;
	char* build_options;
} _cl_program;

#endif
