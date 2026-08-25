#include <iostream>
#include <string>

#include "../include/benchmarks.hpp"


void PrintUsage(const char* program)
{
    std::cout
        << "Usage:\n"
        << "  " << program << "              Run default test\n"
        << "  " << program << " basic       Run basic matmul benchmark\n"
        << "  " << program << " datatypes   Benchmark data types\n"
        << "  " << program << " 16   Benchmark integer and float '16' data types on GPU only\n"
        << "  " << program << " help        Show this message\n";
}


int main(int argc, char* argv[])
{
    if (argc == 1) {
        RunBasicMatmul();
        return 0;
    }

    const std::string command = argv[1];

    if (command == "basic") {
        RunBasicMatmul();
    }
    else if (command == "datatypes") {
        RunDatatypeBenchmark();
    }
    else if (command == "16") {
        Runi16fp16bf16Benchmark();
    }
    else if (command == "help" || command == "--help" || command == "-h") {
        PrintUsage(argv[0]);
    }
    else {
        std::cerr << "Unknown command: " << command << "\n\n";
        PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
