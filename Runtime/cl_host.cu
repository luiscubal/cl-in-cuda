#include "cl_host.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include "cl_interface_shared.cuh"

#define HAS_FLAG(bitfield, flag) (bitfield & flag)
#define HARDCODED_PLATFORM_ID 1
#define HARDCODED_DEVICE_ID 2
#define HARDCODED_CONTEXT_ID 3
#define HARDCODED_COMMAND_QUEUE_ID 4
#define HARDCODED_PROGRAM_ID 5

cl_int clGetPlatformIDs(
	cl_uint num_entries,
	cl_platform_id *platforms,
	cl_uint *num_platforms
) {
	if (num_entries == 0 && platforms != nullptr) {
		return CL_INVALID_VALUE;
	}
	if (num_platforms == nullptr && platforms == nullptr) {
		return CL_INVALID_VALUE;
	}
	if (num_platforms != nullptr) {
		*num_platforms = 1;
	}
	if (platforms != nullptr) {
		*platforms = HARDCODED_PLATFORM_ID;
	}
	return CL_SUCCESS;
}

cl_int clGetDeviceIDs(
	cl_platform_id platform,
	cl_device_type device_type,
	cl_uint num_entries,
	cl_device_id *devices,
	cl_uint *num_devices
) {
	if (platform != HARDCODED_PLATFORM_ID) {
		return CL_INVALID_PLATFORM;
	}
	if (num_entries == 0 && devices != nullptr) {
		return CL_INVALID_VALUE;
	}
	if (device_type == CL_DEVICE_TYPE_GPU || device_type == CL_DEVICE_TYPE_ALL) {
		if (num_devices != nullptr) {
			*num_devices = 1;
		}
		if (devices != nullptr) {
			*devices = HARDCODED_DEVICE_ID;
		}
		return CL_SUCCESS;
	}

	return CL_DEVICE_NOT_FOUND;
}

static cl_int getInfoFromString(
	const char *value,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret
) {
	if (param_value_size_ret != nullptr) {
		*param_value_size_ret = strlen(value) + 1;
	}
	if (param_value != nullptr) {
		if (param_value_size < strlen(value) + 1) {
			return CL_INVALID_VALUE;
		}
		strcpy((char*) param_value, value);
	}
	return CL_SUCCESS;
}

cl_int clGetPlatformInfo(
	cl_platform_id platform,
	cl_platform_info param_name,
	size_t param_value_size,
	void* param_value,
	size_t* param_value_size_ret
) {
	if (platform != HARDCODED_PLATFORM_ID) {
		return CL_INVALID_PLATFORM;
	}
	
	switch (param_name) {
	case CL_PLATFORM_NAME:
		return getInfoFromString("OpenCL emulator (via CUDA)", param_value_size, param_value, param_value_size_ret);
	default:
		return CL_INVALID_VALUE;
	}
}

cl_int clGetDeviceInfo(
	cl_device_id device,
	cl_device_info param_name,
	size_t param_value_size,
	void* param_value,
	size_t* param_value_size_ret
) {
	if (device != HARDCODED_DEVICE_ID) {
		return CL_INVALID_DEVICE;
	}
	
	switch (param_name) {
	case CL_DEVICE_NAME:
		return getInfoFromString("NVIDIA GPU", param_value_size, param_value, param_value_size_ret);
	case CL_DEVICE_TYPE:
		if (param_value_size_ret != nullptr) {
			*param_value_size_ret = sizeof(cl_device_type);
		}
		else if (param_value != nullptr && param_value_size < sizeof(cl_device_type)) {
			return CL_INVALID_VALUE;
		}
		if (param_value != nullptr) {
			*(cl_device_type*)param_value = CL_DEVICE_TYPE_GPU;
		}
		return CL_SUCCESS;
	default:
		return CL_INVALID_VALUE;
	}
}

cl_context clCreateContext(
	cl_context_properties *properties,
	cl_uint num_devices,
	const cl_device_id *devices,
	void *callback(const char *errinfo, const void *private_info, size_t cb, void *user_data),
	void *user_data,
	cl_int *errcode_ret
) {
	if (num_devices != 1) {
		*errcode_ret = CL_INVALID_DEVICE;
		return 0;
	}

	cl_device_id device = *devices;
	if (device != HARDCODED_DEVICE_ID) {
		*errcode_ret = CL_DEVICE_NOT_AVAILABLE;
		return 0;
	}

	*errcode_ret = CL_SUCCESS;
	return HARDCODED_CONTEXT_ID;
}

cl_int clReleaseContext(
	cl_context context
) {
	if (context != HARDCODED_CONTEXT_ID) {
		return CL_INVALID_CONTEXT;
	}

	return CL_SUCCESS;
}

cl_command_queue clCreateCommandQueue(
	cl_context context,
	cl_device_id device,
	cl_command_queue_properties properties,
	cl_int *errcode_ret
) {
	if (context != HARDCODED_CONTEXT_ID) {
		*errcode_ret = CL_INVALID_CONTEXT;
		return 0;
	}
	if (device != HARDCODED_DEVICE_ID) {
		*errcode_ret = CL_INVALID_DEVICE;
		return 0;
	}
	return HARDCODED_COMMAND_QUEUE_ID;
}

cl_int clReleaseCommandQueue(
	cl_command_queue command_queue
) {
	if (command_queue != HARDCODED_COMMAND_QUEUE_ID) {
		return CL_INVALID_COMMAND_QUEUE;
	}
	return CL_SUCCESS;
}

void* clSVMAlloc(
	cl_context context,
	cl_svm_mem_flags flags,
	size_t size,
	unsigned int alignment
) {
	void* buf;
	if (cudaMallocManaged(&buf, size) == 0) {
		return buf;
	}
	return nullptr;
}

void clSVMFree(
	cl_context context,
	void* svm_pointer
) {
	cudaFree(svm_pointer);
}

cl_mem clCreateBuffer(
	cl_context context,
	cl_mem_flags flags,
	size_t size,
	void *host_ptr,
	cl_int *errcode_ret
) {
	if (size == 0) {
		*errcode_ret = CL_INVALID_BUFFER_SIZE;
		return nullptr;
	}

	if (HAS_FLAG(flags, CL_MEM_USE_HOST_PTR)) {
		std::cerr << "Unsupported flag CL_MEM_USE_HOST_PTR." << std::endl;
		*errcode_ret = CL_INVALID_VALUE;
		return nullptr;
	}

	bool shouldCopy = HAS_FLAG(flags, CL_MEM_COPY_HOST_PTR);
	if (shouldCopy && host_ptr == nullptr) {
		*errcode_ret = CL_INVALID_HOST_PTR;
		return nullptr;
	}

	void* ptr;
	cudaError err = cudaMalloc(&ptr, size);
	if (err == cudaErrorMemoryAllocation) {
		*errcode_ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
		return nullptr;
	}

	cl_mem mem = (cl_mem) malloc(sizeof(*mem));
	if (mem == nullptr) {
		cudaFree(ptr);
		*errcode_ret = CL_OUT_OF_HOST_MEMORY;
		return nullptr;
	}
	mem->refs = 1;
	mem->ptr = ptr;
	mem->size = size;

	if (shouldCopy) {
		cudaMemcpy(ptr, host_ptr, size, cudaMemcpyHostToDevice);
	}

	return mem;
}

cl_int clReleaseMemObject(
	cl_mem memobj
) {
	if (memobj == nullptr) {
		return CL_INVALID_MEM_OBJECT;
	}

	if (--memobj->refs == 0) {
		cudaFree(memobj->ptr);
		free(memobj);
	}
	return CL_SUCCESS;
}

cl_int clEnqueueReadBuffer(
	cl_command_queue command_queue,
	cl_mem buffer,
	cl_bool blocking_read,
	size_t offset,
	size_t cb,
	void *ptr,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event
) {
	if (command_queue != HARDCODED_COMMAND_QUEUE_ID) {
		return CL_INVALID_COMMAND_QUEUE;
	}

	if (buffer == nullptr) {
		return CL_INVALID_MEM_OBJECT;
	}

	if (ptr == nullptr) {
		return CL_INVALID_VALUE;
	}

	if (offset >= buffer->size || cb > buffer->size || buffer->size - cb < offset) {
		return CL_INVALID_VALUE;
	}

	cudaError err = cudaMemcpy((char*)ptr + offset, buffer->ptr, cb, cudaMemcpyDeviceToHost);
	if (err == cudaErrorInvalidDevicePointer) {
		// We've been reading from garbage
		return CL_INVALID_MEM_OBJECT;
	}
	if (err == cudaErrorInvalidValue) {
		return CL_INVALID_VALUE;
	}
	return CL_SUCCESS;
}

cl_int clEnqueueWriteBuffer(
	cl_command_queue command_queue,
	cl_mem buffer,
	cl_bool blocking_write,
	size_t offset,
	size_t cb,
	const void *ptr,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event
) {
	if (command_queue != HARDCODED_COMMAND_QUEUE_ID) {
		return CL_INVALID_COMMAND_QUEUE;
	}

	if (buffer == nullptr) {
		return CL_INVALID_MEM_OBJECT;
	}

	if (ptr == nullptr) {
		return CL_INVALID_VALUE;
	}

	if (offset >= buffer->size || cb > buffer->size || buffer->size - cb < offset) {
		return CL_INVALID_VALUE;
	}

	cudaError err = cudaMemcpy(buffer->ptr, (char*)ptr + offset, cb, cudaMemcpyHostToDevice);
	if (err == cudaErrorInvalidDevicePointer) {
		// We've been reading from garbage
		return CL_INVALID_MEM_OBJECT;
	}
	if (err == cudaErrorInvalidValue) {
		return CL_INVALID_VALUE;
	}
	return CL_SUCCESS;
}

cl_program clCreateProgramWithBinary(
	cl_context context,
	cl_uint num_devices,
	const cl_device_id *device_list,
	const size_t *lengths,
	const unsigned char **binaries,
	cl_int *binary_status,
	cl_int *errcode_ret
) {
	if (binary_status != nullptr) {
		std::cerr << "Unsupported binary_status" << std::endl;
		*errcode_ret = CL_INVALID_VALUE;
		return 0;
	}

	if (context != HARDCODED_CONTEXT_ID) {
		*errcode_ret = CL_INVALID_CONTEXT;
		return 0;
	}
	if (num_devices == 0 || device_list == nullptr) {
		*errcode_ret = CL_INVALID_VALUE;
		return 0;
	}
	if (num_devices != 1) {
		std::cerr << "Invalid number of devices: " << num_devices << std::endl;
		*errcode_ret = CL_INVALID_VALUE;
		return 0;
	}
	if (lengths[0] == 0 || binaries[0] == nullptr) {
		*errcode_ret = CL_INVALID_VALUE;
		return 0;
	}
	if (device_list[0] != HARDCODED_DEVICE_ID) {
		*errcode_ret = CL_INVALID_DEVICE;
		return 0;
	}
	if (lengths[0] != sizeof(int)) {
		*errcode_ret = CL_INVALID_BINARY;
		return 0;
	}
	int value = *(int*)binaries[0];
	if (value != 1) {
		*errcode_ret = CL_INVALID_BINARY;
		return 0;
	}

	*errcode_ret = CL_SUCCESS;
	return HARDCODED_PROGRAM_ID;
}

cl_int clBuildProgram(
	cl_program program,
	cl_uint num_devices,
	const cl_device_id *device_list,
	const char *options,
	void(*pfn_notify)(cl_program, void *user_data),
	void *user_data
) {
	if (program != HARDCODED_PROGRAM_ID) {
		return CL_INVALID_PROGRAM;
	}
	if (num_devices == 0 && device_list != nullptr) {
		return CL_INVALID_VALUE;
	}
	if (num_devices != 0 && device_list == nullptr) {
		return CL_INVALID_VALUE;
	}
	if(num_devices != 0 && device_list[0] != HARDCODED_DEVICE_ID) {
		return CL_INVALID_DEVICE;
	}

	return CL_SUCCESS;
}

extern program_descriptor vectoradd_program;

static cl_int getBuildInfoFromString(
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret,
	const char* value
) {
	size_t size = strlen(value) + 1;
	if (param_value_size < size && param_value != nullptr) {
		return CL_INVALID_VALUE;
	}
	if (param_value_size_ret != nullptr) {
		*param_value_size_ret = size;
	}
	if (param_value != nullptr) {
		strcpy((char*)param_value, value);
	}
	return CL_SUCCESS;
}

cl_int clGetProgramBuildInfo(
	cl_program program,
	cl_device_id device,
	cl_program_build_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret
) {
	if (program != HARDCODED_PROGRAM_ID) {
		return CL_INVALID_PROGRAM;
	}
	if (device != HARDCODED_DEVICE_ID) {
		return CL_INVALID_DEVICE;
	}
	switch (param_name) {
	case CL_PROGRAM_BUILD_STATUS:
		if (param_value_size < sizeof(cl_build_status) && param_value != nullptr) {
			return CL_INVALID_VALUE;
		}
		if (param_value_size_ret != nullptr) {
			*param_value_size_ret = sizeof(cl_build_status);
		}
		if (param_value != nullptr) {
			*(cl_build_status*)param_value = CL_BUILD_SUCCESS;
		}
		return CL_SUCCESS;
	case CL_PROGRAM_BUILD_OPTIONS:
		return getBuildInfoFromString(param_value_size, param_value, param_value_size_ret, vectoradd_program.build_options);
	case CL_PROGRAM_BUILD_LOG:
		return getBuildInfoFromString(param_value_size, param_value, param_value_size_ret, vectoradd_program.build_log);
	default:
		return CL_INVALID_VALUE;
	}
}

cl_kernel clCreateKernel(
	cl_program program,
	const char *kernel_name,
	cl_int *errcode_ret
) {
	if (program != HARDCODED_PROGRAM_ID) {
		*errcode_ret = CL_INVALID_PROGRAM;
		return 0;
	}

	if (kernel_name == nullptr) {
		*errcode_ret = CL_INVALID_VALUE;
		return 0;
	}

	for (int kernel_id = 0; kernel_id < vectoradd_program.num_kernels; kernel_id++) {
		kernel_descriptor* kdesc = &(vectoradd_program.kernels[kernel_id]);
		if (strcmp(kdesc->name, kernel_name) == 0) {
			if (kdesc->arg_data == 0) {
				kdesc->arg_data = (void**)malloc(sizeof(void*) * kdesc->num_args);
				if (kdesc->arg_data == 0) {
					*errcode_ret = CL_OUT_OF_HOST_MEMORY;
					return 0;
				}
				for (int i = 0; i < kdesc->num_args; i++) {
					arg_descriptor* adesc = &(kdesc->arg_descriptors[i]);
					switch (adesc->arg_type) {
					case ARG_TYPE_SCALAR:
					case ARG_TYPE_LOCAL_MEM:
						kdesc->arg_data[i] = malloc(adesc->data.scalar_size);
						if (kdesc->arg_data[i] == nullptr) {
							*errcode_ret = CL_OUT_OF_HOST_MEMORY;
							return 0;
						}
						break;
					case ARG_TYPE_MEM_OBJ:
						kdesc->arg_data[i] = nullptr;
						break;
					}
				}
			}
			*errcode_ret = CL_SUCCESS;
			return (cl_kernel)(kernel_id + 1);
		}
	}

	*errcode_ret = CL_INVALID_KERNEL_NAME;
	return 0;
}

cl_int clReleaseKernel(
	cl_kernel kernel
) {
	// FIXME Leaking the arg_data
	if (kernel == 0 || kernel > vectoradd_program.num_kernels) {
		return CL_INVALID_KERNEL;
	}
	return CL_SUCCESS;
}

cl_int clReleaseProgram(
	cl_program program
) {
	if (program != HARDCODED_PROGRAM_ID) {
		return CL_INVALID_PROGRAM;
	}
	return CL_SUCCESS;
}

cl_int clSetKernelArg(
	cl_kernel kernel,
	cl_uint arg_index,
	size_t arg_size,
	const void *arg_value
) {
	if (kernel == 0 || kernel > vectoradd_program.num_kernels) {
		return CL_INVALID_KERNEL;
	}
	
	kernel_descriptor* kdesc = &(vectoradd_program.kernels[kernel - 1]);
	if (arg_index >= kdesc->num_args) {
		return CL_INVALID_ARG_INDEX;
	}

	arg_descriptor* adesc = &(kdesc->arg_descriptors[arg_index]);
	switch (adesc->arg_type) {
	case ARG_TYPE_SCALAR:
		if (arg_size != adesc->data.scalar_size) {
			return CL_INVALID_ARG_SIZE;
		}
		memcpy(((void**)kdesc->arg_data)[arg_index], arg_value, arg_size);
		return CL_SUCCESS;
	case ARG_TYPE_LOCAL_MEM:
		if (arg_value != nullptr) {
			return CL_INVALID_ARG_VALUE;
		}
		if (arg_size == 0) {
			return CL_INVALID_ARG_SIZE;
		}
		memcpy(((void**)kdesc->arg_data)[arg_index], &arg_size, sizeof(size_t));
		return CL_SUCCESS;
	case ARG_TYPE_MEM_OBJ:
		if (arg_value == nullptr) {
			return CL_INVALID_ARG_VALUE;
		}
		if (arg_size != sizeof(cl_mem)) {
			return CL_INVALID_ARG_SIZE;
		}
		kdesc->arg_data[arg_index] = (*(cl_mem*)arg_value)->ptr;
		return CL_SUCCESS;
	}
	abort();
}

cl_int clEnqueueNDRangeKernel(
	cl_command_queue command_queue,
	cl_kernel kernel,
	cl_uint work_dim,
	const size_t *global_work_offset,
	const size_t *global_work_size,
	const size_t *local_work_size,
	cl_uint num_events_in_wait_list,
	const cl_event *event_wait_list,
	cl_event *event
) {
	if (command_queue != HARDCODED_COMMAND_QUEUE_ID) {
		return CL_INVALID_COMMAND_QUEUE;
	}
	if (kernel == 0 || kernel > vectoradd_program.num_kernels) {
		return CL_INVALID_KERNEL;
	}
	if (work_dim == 0 || work_dim > 3) {
		return CL_INVALID_WORK_DIMENSION;
	}
	if (global_work_offset != nullptr) {
		return CL_INVALID_GLOBAL_OFFSET;
	}
	if (local_work_size == nullptr) {
		std::cerr << "Explicit local size required" << std::endl;
		return CL_INVALID_WORK_GROUP_SIZE;
	}
	if (global_work_size == nullptr) {
		std::cerr << "Global size required" << std::endl;
		return CL_INVALID_WORK_GROUP_SIZE;
	}

	size_t num_groups[3] = { 1, 1, 1 };
	size_t local_sizes[3] = { 1, 1, 1 };
	size_t total_sizes[3] = { 1, 1, 1 };
	for (int i = 0; i < work_dim; i++) {
		size_t size = global_work_size[i];
		size_t local_size = local_work_size[i];

		if (size % local_size != 0) {
			return CL_INVALID_WORK_GROUP_SIZE;
		}

		total_sizes[i] = size;
		local_sizes[i] = local_size;
		num_groups[i] = size / local_size;
	}

	kernel_descriptor* kdesc = &(vectoradd_program.kernels[kernel - 1]);
	kdesc->gridX = num_groups[0];
	kdesc->gridY = num_groups[1];
	kdesc->gridZ = num_groups[2];
	kdesc->localX = local_sizes[0];
	kdesc->localY = local_sizes[1];
	kdesc->localZ = local_sizes[2];
	kdesc->totalX = total_sizes[0];
	kdesc->totalY = total_sizes[1];
	kdesc->totalZ = total_sizes[2];
	kdesc->launcher(kdesc);
	cudaDeviceSynchronize();

	return CL_SUCCESS;
}
