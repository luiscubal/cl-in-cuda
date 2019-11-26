// This program implements a vector addition using OpenCL
// Error checking added by Kate Cowles

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cl_host.h"

// Simple OpenCL error checking function
void chk(cl_int status, const char* cmd) {

	if (status != CL_SUCCESS) {
		printf("%s failed (%d)\n", cmd, status);
		exit(-1);
	}
}

void read_file(const char* path, char** values, size_t* size) {
	FILE* f = fopen(path, "rb");
	if (!f) {
		std::cerr << "Error reading file" << std::endl;
		abort();
	}
	fseek(f, 0, SEEK_END);
	*size = ftell(f);
	fseek(f, 0, SEEK_SET);

	char* buf = (char*)malloc(*size);
	*values = buf;
	size_t remaining_size = *size;
	while (remaining_size > 0) {
		size_t read = fread(buf, 1, remaining_size, f);

		remaining_size -= read;
		buf += read;
	}
	fclose(f);
}

void execute(int* A, int* B, int* C, const int* elements) {
	// This code executes on the OpenCL host    
	// Compute the size of the data 
	size_t datasize = sizeof(int)*(*elements);

	// Use this to check the output of each API call
	cl_int status;

	// Retrieve the number of platforms
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	chk(status, "clGetPlatformIDs");


	// Allocate enough space for each platform
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(
		numPlatforms * sizeof(cl_platform_id));

	// Fill in the platforms
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	// Retrieve the number of devices
	cl_uint numDevices = 0;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0,
		NULL, &numDevices);
	chk(status, "clGetDeviceIDs");

	// Allocate enough space for each device
	cl_device_id *devices;
	devices = (cl_device_id*)malloc(
		numDevices * sizeof(cl_device_id));

	// Fill in the devices 
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
		numDevices, devices, NULL);
	chk(status, "clGetDeviceIDs");

	// Create a context and associate it with the devices
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL,
		NULL, &status);
	chk(status, "clCreateContext");

	// Create a command queue and associate it with the device 
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0,
		&status);
	chk(status, "clCreateCommandQueue");


	// Create a buffer object that will contain the data 
	// from the host array A
	cl_mem bufA;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	chk(status, "clCreateBuffer");

	// Create a buffer object that will contain the data 
	// from the host array B
	cl_mem bufB;
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	chk(status, "clCreateBuffer");

	// Create a buffer object that will hold the output data
	cl_mem bufC;
	bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize,
		NULL, &status);
	chk(status, "clCreateBuffer");

	// Write input array A to the device buffer bufferA
	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE,
		0, datasize, A, 0, NULL, NULL);
	chk(status, "clEnqueueWriteBuffer");
	// Write input array B to the device buffer bufferB
	status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE,
		0, datasize, B, 0, NULL, NULL);

	// Create a program with source code
	char *binary_values;
	size_t binary_size;
	read_file("vecadd.txt", &binary_values, &binary_size);
	cl_program program = clCreateProgramWithBinary(context,
		1,
		devices,
		&binary_size,
		(const unsigned char**) &binary_values,
		NULL,
		&status);
	chk(status, "clCreateProgramWithBinary");

	// Build (compile) the program for the device
	// Show log if errors occur

	if (clBuildProgram(program, numDevices, devices, NULL, NULL, NULL)
		!= CL_SUCCESS)
	{
		// Shows the log
		size_t log_size;
		// First call to get the proper size
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
			NULL, &log_size);
		char* build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
			log_size, build_log, NULL);
		build_log[log_size] = '\0';
		printf("Compile error: %s \n", build_log);
		delete[] build_log;
		exit(-1);
	}


	// Create the vector addition kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vecadd", &status);
	chk(status, "clCreateKernel");


	// Associate the input and output buffers with the kernel 
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	chk(status, "clSetKernelArg");

	// Define an index space (global work size) of work 
	// items for execution. A workgroup size (local work size) 
	// is not required, but can be used.
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	// There are 'elements' work-items 
	globalWorkSize[0] = *elements;
	localWorkSize[0] = 64;

	// Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
		globalWorkSize, localWorkSize, 0, NULL, NULL);
	chk(status, "clEnqueueNDRangeKernel");


	// Read the device output buffer to the host output array
	clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0,
		datasize, C, 0, NULL, NULL);
	chk(status, "clEnqueueReadBuffer");


	// Verify the output
	//int result = 1;
	//int i;
	//for(i = 0; i < *elements; i++) {
	//    if(C[i] != i+i) {
	//        result = 0;
	//        break;
	//    }
	//}

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseContext(context);

	// Free host resources
	//free(A);
	//free(B);
	//free(C);
	free(platforms);
	free(devices);
}

int main() {
	int elements = 256;
	int *A = new int[elements];
	int *B = new int[elements];
	int *C = new int[elements];
	for (int i = 0; i < 256; i++) {
		A[i] = i;
		B[i] = i * 10;
	}
	execute(A, B, C, &elements);

	std::cout << "Done" << std::endl;
	for (int i = 0; i < elements; i++) {
		if (C[i] != i + (i * 10)) {
			std::cout << "Wrong result at " << i << std::endl;
			std::cout << " Expected: " << (i + i * 10) << std::endl;
			std::cout << " Actual: " << C[i] << std::endl;
			break;
		}
	}

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
