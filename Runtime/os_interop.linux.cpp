#ifdef __linux__
#include "os_interop.hpp"
#include <dlfcn.h>
#include <time.h>
#include <iostream>

kernel_launcher get_launcher_by_name(
	const char* name
) {
	void* current_module = dlopen(nullptr, RTLD_NOW);
	if (current_module == nullptr) {
		std::cerr << "could not get self module" << std::endl;
		exit(1);
	}
	kernel_launcher launcher = (kernel_launcher) dlsym(current_module, name);
	if (launcher == nullptr) {
		std::cerr << "could not get symbol " << name << std::endl;
	}
	return launcher;
}

uint64_t measure_time_nanos()
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	uint64_t nanos = (uint64_t)start.tv_sec;
	nanos *= 1000000000;
	nanos += start.tv_nsec;

	return nanos;
}

#endif
