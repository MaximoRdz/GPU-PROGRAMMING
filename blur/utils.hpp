#ifndef UTILS_H
#define UTILS_H


#include <chrono>

std::chrono::high_resolution_clock::time_point TickNow();

double DurationMicroseconds(
        std::chrono::high_resolution_clock::time_point start,
        std::chrono::high_resolution_clock::time_point end);

#endif
