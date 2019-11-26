#ifndef CL_INTERFACE_SHARED_CUH_
#define CL_INTERFACE_SHARED_CUH_

#include <stddef.h>

#define ARG_TYPE_SCALAR 1
#define ARG_TYPE_LOCAL_MEM 2
#define ARG_TYPE_MEM_OBJ 3

struct kernel_descriptor;

typedef void (*kernel_launcher)(kernel_descriptor* desc);

struct arg_descriptor {
	int arg_type;
	union {
		size_t scalar_size;
	} data;
};

struct kernel_descriptor {
	const char* name;
	size_t num_args;
	arg_descriptor* arg_descriptors;
	kernel_launcher launcher;

	int gridX, gridY, gridZ;
	int localX, localY, localZ;
	int totalX, totalY, totalZ;
	void** arg_data;
};

struct program_descriptor {
	size_t num_kernels;
	kernel_descriptor* kernels;
	const char* build_log;
	const char* build_options;
};

#endif
