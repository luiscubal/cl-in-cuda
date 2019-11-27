#ifndef CLHOST_H_
#define CLHOST_H_

#include <stdint.h>
#include <stddef.h>
#include "cl_host_constants.h"

typedef int8_t cl_bool;
typedef int8_t cl_char;
typedef uint8_t cl_uchar;
typedef int16_t cl_short;
typedef uint16_t cl_ushort;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef int64_t cl_long;
typedef uint64_t cl_ulong;
typedef int cl_event;
typedef struct _cl_kernel *cl_kernel;

typedef int cl_platform_id;
typedef int cl_device_id;
typedef struct _cl_program *cl_program;
typedef int cl_context;
typedef int cl_command_queue;
typedef struct {
	void* ptr;
	size_t size;
	int refs;
} *cl_mem;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_svm_mem_flags;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_build_status;
typedef int cl_context_properties;

#define CL_CALLBACK

#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1 << 10)

#ifdef __cplusplus
extern "C" {
#endif

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);

cl_context clCreateContext(cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void *callback(
	const char *errinfo,
	const void *private_info,
	size_t cb,
	void *user_data
), void *user_data, cl_int *errcode_ret);
cl_int clReleaseContext(cl_context context);

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);
cl_int clReleaseCommandQueue(cl_command_queue command_queue);

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
cl_int clReleaseMemObject(cl_mem memobj);

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t cb, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t cb, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret);
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void(*pfn_notify)(
	cl_program,
	void *user_data
), void *user_data);
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);
cl_int clReleaseKernel(cl_kernel kernel);
cl_int clReleaseProgram(cl_program program);
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int clEnqueueBarrier(cl_command_queue command_queue);
cl_int clFinish(cl_command_queue command_queue);

void* clSVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, unsigned int alignment);
void clSVMFree(cl_context context, void* svm_pointer);

#ifdef __cplusplus
}
#endif

#endif
