#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdlib.h>

// Sleep for a given number of milliseconds
void sleep_ms(size_t ms);

// Get the time in milliseconds since UNIX epoch
size_t time_ms();

#endif
