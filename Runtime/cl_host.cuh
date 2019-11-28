#ifndef CLHOST_H_
#define CLHOST_H_

#include <stdint.h>
#include <stddef.h>
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

typedef struct _cl_event {
	cl_ulong startNanos;
	cl_ulong endNanos;
	int refs;
} _cl_event;

typedef struct _cl_mem {
	void* ptr;
	size_t size;
	int refs;
} _cl_mem;

#endif
