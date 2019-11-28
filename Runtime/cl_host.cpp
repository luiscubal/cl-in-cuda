#include "cl_host.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <sstream>
#include "cpptoml.hpp"
#include "cl_interface_shared.h"
#include "os_interop.hpp"

#define HAS_FLAG(bitfield, flag) (bitfield & flag)
#define HARDCODED_PLATFORM_ID 1
#define HARDCODED_DEVICE_ID 2
#define HARDCODED_CONTEXT_ID 3
#define HARDCODED_COMMAND_QUEUE_ID 4

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
	*errcode_ret = CL_SUCCESS;

	// Force device initialization
	cudaFree(nullptr);

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

	*errcode_ret = CL_SUCCESS;
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

static cl_int createEvent(cl_event& evt)
{
	evt = (cl_event)malloc(sizeof(*evt));
	if (evt == nullptr) {
		return CL_OUT_OF_HOST_MEMORY;
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

	cl_event evt = nullptr;
	if (event != nullptr) {
		cl_int err = createEvent(evt);
		if (err != CL_SUCCESS) {
			return err;
		}
		*event = evt;
	}

	if (evt != nullptr) {
		evt->startNanos = measure_time_nanos();
	}
	cudaError err = cudaMemcpy((char*)ptr + offset, buffer->ptr, cb, cudaMemcpyDeviceToHost);
	if (evt != nullptr) {
		evt->endNanos = measure_time_nanos();
	}
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

	cl_event evt = nullptr;
	if (event != nullptr) {
		cl_int err = createEvent(evt);
		if (err != CL_SUCCESS) {
			return err;
		}
		*event = evt;
	}

	if (evt != nullptr) {
		evt->startNanos = measure_time_nanos();
	}
	cudaError err = cudaMemcpy(buffer->ptr, (char*)ptr + offset, cb, cudaMemcpyHostToDevice);
	if (evt != nullptr) {
		evt->endNanos = measure_time_nanos();
	}
	if (err == cudaErrorInvalidDevicePointer) {
		// We've been reading from garbage
		return CL_INVALID_MEM_OBJECT;
	}
	if (err == cudaErrorInvalidValue) {
		return CL_INVALID_VALUE;
	}
	return CL_SUCCESS;
}

static cl_int buildKernel(
	cpptoml::table& kernel_toml,
	cl_kernel storage
) {
	cl_int errcode = CL_SUCCESS;

	auto name = kernel_toml.get_as<std::string>("name");
	auto symbol_name = kernel_toml.get_as<std::string>("symbol_name");

	if (!name || !symbol_name) {
		return CL_INVALID_BINARY;
	}

	size_t name_length = name->length();
	storage->name = (char*)malloc(name_length + 1);
	size_t symbol_name_length = symbol_name->length();
	storage->symbol_name = (char*)malloc(symbol_name_length + 1);

	storage->launcher = get_launcher_by_name(symbol_name->c_str());
	if (storage->launcher == nullptr) {
		return CL_INVALID_BINARY;
	}

	storage->num_args = 0;

	if (!storage->name || !storage->symbol_name) {
		errcode = CL_OUT_OF_HOST_MEMORY;
	}
	else {
		strcpy(storage->name, name->c_str());
		strcpy(storage->symbol_name, symbol_name->c_str());

		auto args_toml = kernel_toml.get_table_array("args");

		if (args_toml) {
			auto args_length = args_toml->end() - args_toml->begin();
			storage->arg_data = (void**)calloc(args_length, sizeof(void*));
			storage->arg_descriptors = (arg_descriptor*)calloc(args_length, sizeof(struct arg_descriptor));
			if (storage->arg_data == nullptr || storage->arg_descriptors == nullptr) {
				errcode = CL_OUT_OF_HOST_MEMORY;
			}
			for (const auto& arg : *args_toml) {
				auto type = arg->get_as<std::string>("type");
				if (!type) {
					errcode = CL_INVALID_BINARY;
					break;
				}
				if (*type == "global_ptr") {
					storage->arg_descriptors[storage->num_args].arg_type = ARG_TYPE_GLOBAL_MEM;
				}
				else if (*type == "local_ptr") {
					storage->arg_descriptors[storage->num_args].arg_type = ARG_TYPE_LOCAL_MEM;
					storage->arg_descriptors[storage->num_args].data.scalar_size = sizeof(size_t);
					storage->arg_data[storage->num_args] = malloc(sizeof(size_t));
					if (storage->arg_data == nullptr) {
						errcode = CL_OUT_OF_HOST_MEMORY;
						break;
					}
				}
				else if (*type == "scalar") {
					storage->arg_descriptors[storage->num_args].arg_type = ARG_TYPE_SCALAR;

					auto size = arg->get_as<int>("size");
					if (!size) {
						errcode = CL_INVALID_BINARY;
						break;
					}
					storage->arg_descriptors[storage->num_args].data.scalar_size = *size;
					storage->arg_data[storage->num_args] = malloc(*size);
					if (storage->arg_data == nullptr) {
						errcode = CL_OUT_OF_HOST_MEMORY;
						break;
					}
				}
				else {
					std::cerr << "Not implemented " << *type << std::endl;
					errcode = CL_INVALID_BINARY;
					break;
				}
				++storage->num_args;
			}
		}
	}

	if (errcode != CL_SUCCESS) {
		free(storage->name);
		free(storage->symbol_name);
		free(storage->arg_data);
		free(storage->arg_descriptors);
		for (int i = 0; i < storage->num_args; i++) {
			free(storage->arg_data[i]);
		}
	}
	return errcode;
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

	std::string string((const char*)*binaries, *lengths);
	std::istringstream stream(string);
	cpptoml::parser parser(stream);
	std::shared_ptr<cpptoml::table> obj;
	try {
		obj = parser.parse();
	}
	catch (cpptoml::parse_exception exc) {
		*errcode_ret = CL_INVALID_BINARY;
		return 0;
	}
	auto program_toml = obj->get_table("program");
	if (!program_toml) {
		*errcode_ret = CL_INVALID_BINARY;
		return 0;
	}

	auto build_log = program_toml->get_as<std::string>("build_log");
	auto build_options = program_toml->get_as<std::string>("build_options");
	auto kernels_toml = obj->get_table_array("kernels");
	if (!build_log || !build_options) {
		*errcode_ret = CL_INVALID_BINARY;
		return 0;
	}

	cl_program program = nullptr;

	program = (cl_program)malloc(sizeof(*program));
	if (program == nullptr) {
		*errcode_ret = CL_OUT_OF_HOST_MEMORY;
	}
	else {
		size_t build_log_length = build_log->length();
		program->build_log = (char*)malloc(build_log_length + 1);
		size_t build_options_length = build_options->length();
		program->build_options = (char*)malloc(build_options_length + 1);
		program->kernels = nullptr;
		program->num_kernels = 0;

		if (program->build_log == nullptr || program->build_options == nullptr) {
			*errcode_ret = CL_OUT_OF_HOST_MEMORY;
		}
		else {
			strcpy(program->build_log, build_log->c_str());
			strcpy(program->build_options, build_options->c_str());

			if (kernels_toml != nullptr) {
				auto kernels_length = kernels_toml->end() - kernels_toml->begin();
				program->kernels = (cl_kernel) calloc(kernels_length, sizeof(struct _cl_kernel));
				if (program->kernels == nullptr) {
					*errcode_ret = CL_OUT_OF_HOST_MEMORY;
				}
				else {
					*errcode_ret = CL_SUCCESS;
					for (const auto& kernel : *kernels_toml) {
						cl_int err = buildKernel(*kernel, program->kernels + program->num_kernels);

						if (err != CL_SUCCESS) {
							*errcode_ret = err;
							break;
						}
						++program->num_kernels;
					}
				}
			}

			if (*errcode_ret == CL_SUCCESS) {
				// Do not free fields
				return program;
			}
		}
	}

	// free(nullptr) is a no-op
	if (program != nullptr) {
		free(program->build_log);
		free(program->build_options);
		for (int i = 0; i < program->num_kernels; i++) {
			free(program->kernels[i].name);
			free(program->kernels[i].symbol_name);
			free(program->kernels[i].arg_data);
		}
		free(program->kernels);

	}
	free(program);
	return nullptr;
}

cl_int clBuildProgram(
	cl_program program,
	cl_uint num_devices,
	const cl_device_id *device_list,
	const char *options,
	void(*pfn_notify)(cl_program, void *user_data),
	void *user_data
) {
	if (program == nullptr) {
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
	if (program == nullptr) {
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
		return getBuildInfoFromString(param_value_size, param_value, param_value_size_ret, program->build_options);
	case CL_PROGRAM_BUILD_LOG:
		return getBuildInfoFromString(param_value_size, param_value, param_value_size_ret, program->build_log);
	default:
		return CL_INVALID_VALUE;
	}
}

cl_kernel clCreateKernel(
	cl_program program,
	const char *kernel_name,
	cl_int *errcode_ret
) {
	if (program == nullptr) {
		*errcode_ret = CL_INVALID_PROGRAM;
		return nullptr;
	}

	if (kernel_name == nullptr) {
		*errcode_ret = CL_INVALID_VALUE;
		return nullptr;
	}

	for (int kernel_id = 0; kernel_id < program->num_kernels; kernel_id++) {
		cl_kernel kdesc = &(program->kernels[kernel_id]);
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
					case ARG_TYPE_GLOBAL_MEM:
						kdesc->arg_data[i] = nullptr;
						break;
					}
				}
			}
			*errcode_ret = CL_SUCCESS;
			return kdesc;
		}
	}

	*errcode_ret = CL_INVALID_KERNEL_NAME;
	return nullptr;
}

cl_int clReleaseKernel(
	cl_kernel kernel
) {
	// FIXME Leaking the arg_data
	if (kernel == nullptr) {
		return CL_INVALID_KERNEL;
	}
	return CL_SUCCESS;
}

cl_int clReleaseProgram(
	cl_program program
) {
	if (program == 0) {
		return CL_INVALID_PROGRAM;
	}
	free((void*) program->build_log);
	free((void*) program->build_options);
	free(program);
	return CL_SUCCESS;
}

cl_int clSetKernelArg(
	cl_kernel kernel,
	cl_uint arg_index,
	size_t arg_size,
	const void *arg_value
) {
	if (kernel == nullptr) {
		return CL_INVALID_KERNEL;
	}
	
	if (arg_index >= kernel->num_args) {
		return CL_INVALID_ARG_INDEX;
	}

	arg_descriptor* adesc = &(kernel->arg_descriptors[arg_index]);
	switch (adesc->arg_type) {
	case ARG_TYPE_SCALAR:
		if (arg_size != adesc->data.scalar_size) {
			return CL_INVALID_ARG_SIZE;
		}
		memcpy(((void**)kernel->arg_data)[arg_index], arg_value, arg_size);
		return CL_SUCCESS;
	case ARG_TYPE_LOCAL_MEM:
		if (arg_value != nullptr) {
			return CL_INVALID_ARG_VALUE;
		}
		if (arg_size == 0) {
			return CL_INVALID_ARG_SIZE;
		}
		memcpy(((void**)kernel->arg_data)[arg_index], &arg_size, sizeof(size_t));
		return CL_SUCCESS;
	case ARG_TYPE_GLOBAL_MEM:
		if (arg_value == nullptr) {
			return CL_INVALID_ARG_VALUE;
		}
		if (arg_size != sizeof(cl_mem)) {
			return CL_INVALID_ARG_SIZE;
		}
		kernel->arg_data[arg_index] = (*(cl_mem*)arg_value)->ptr;
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
	if (kernel == nullptr) {
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
	for (int i = 0; i < (int)work_dim; i++) {
		size_t size = global_work_size[i];
		size_t local_size = local_work_size[i];

		if (size % local_size != 0) {
			return CL_INVALID_WORK_GROUP_SIZE;
		}

		total_sizes[i] = size;
		local_sizes[i] = local_size;
		num_groups[i] = size / local_size;
	}

	cl_event evt = nullptr;
	if (event != nullptr) {
		cl_int err = createEvent(evt);
		if (err != CL_SUCCESS) {
			return err;
		}
		*event = evt;
	}

	kernel->gridX = num_groups[0];
	kernel->gridY = num_groups[1];
	kernel->gridZ = num_groups[2];
	kernel->localX = local_sizes[0];
	kernel->localY = local_sizes[1];
	kernel->localZ = local_sizes[2];
	kernel->totalX = total_sizes[0];
	kernel->totalY = total_sizes[1];
	kernel->totalZ = total_sizes[2];
	if (evt != nullptr) {
		evt->startNanos = measure_time_nanos();
	}

	float elapsedMs;
	kernel->launcher(kernel, &elapsedMs);

	if (evt != nullptr) {
		evt->endNanos = evt->startNanos + (uint64_t)(elapsedMs * 1e6);
	}

	return CL_SUCCESS;
}

cl_int clEnqueueBarrier(
	cl_command_queue command_queue
) {
	// No action needed. Async compute is not supported anyway
	return CL_SUCCESS;
}

cl_int clFinish(
	cl_command_queue command_queue
) {
	// No action needed. Async compute is not supported anyway
	return CL_SUCCESS;
}

cl_int clRetainEvent(
	cl_event event
) {
	if (event == nullptr) {
		return CL_INVALID_EVENT;
	}
	++event->refs;
	return CL_SUCCESS;
}

cl_int clReleaseEvent(
	cl_event event
) {
	if (event == nullptr) {
		return CL_INVALID_EVENT;
	}
	
	if (--event->refs == 0) {
		free(event);
	}
	return CL_SUCCESS;
}

cl_int clSetEventCallback(
	cl_event event,
	cl_int command_exec_callback_type,
	void (CL_CALLBACK  *pfn_event_notify) (cl_event event, cl_int event_command_exec_status, void *user_data),
	void *user_data
) {
	if (event == nullptr) {
		return CL_INVALID_EVENT;
	}
	if (pfn_event_notify == nullptr) {
		return CL_INVALID_VALUE;
	}
	if (command_exec_callback_type != CL_COMPLETE) {
		std::cerr << "TODO command_exec_callback_type " << command_exec_callback_type << std::endl;
		return CL_INVALID_VALUE;
	}

	pfn_event_notify(event, command_exec_callback_type, user_data);
	return CL_SUCCESS;
}

cl_int clGetEventProfilingInfo(
	cl_event event,
	cl_profiling_info param_name,
	size_t param_value_size,
	void *param_value,
	size_t *param_value_size_ret
) {
	if (param_name != CL_PROFILING_COMMAND_QUEUED &&
		param_name != CL_PROFILING_COMMAND_SUBMIT &&
		param_name != CL_PROFILING_COMMAND_START &&
		param_name != CL_PROFILING_COMMAND_END
	) {
		return CL_INVALID_VALUE;
	}
	if (param_value != nullptr && param_value_size < sizeof(cl_ulong)) {
		return CL_INVALID_VALUE;
	}
	if (event == nullptr) {
		return CL_INVALID_VALUE;
	}
	if (param_value_size_ret != nullptr) {
		*param_value_size_ret = sizeof(cl_ulong);
	}
	if (param_value != nullptr) {
		switch (param_name) {
		case CL_PROFILING_COMMAND_START:
		case CL_PROFILING_COMMAND_SUBMIT:
		case CL_PROFILING_COMMAND_QUEUED:
			*(cl_ulong*) param_value = event->startNanos;
			break;
		case CL_PROFILING_COMMAND_END:
			*(cl_ulong*) param_value = event->endNanos;
		}
	}

	return CL_SUCCESS;
}
