#include "cl_interface_shared.h"
#include <stdint.h>

kernel_launcher get_launcher_by_name(const char* name);
uint64_t measure_time_nanos();
