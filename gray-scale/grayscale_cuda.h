#pragma once

void launchGrayScaleConversion(const unsigned char* h_input, 
        unsigned char* h_output, int width, int height, int channels, 
        size_t inputStep, size_t outputStep,
        float* kernelTimeMs, float* totalTimeMs);
