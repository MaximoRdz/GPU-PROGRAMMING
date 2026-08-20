#include <chrono>

std::chrono::high_resolution_clock::time_point TickNow(void)
{
    return std::chrono::high_resolution_clock::now();
}

double DurationMicroseconds(
        std::chrono::high_resolution_clock::time_point start,
        std::chrono::high_resolution_clock::time_point end)
{
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}
