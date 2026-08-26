#ifndef BENCHMARKS_H 
#define BENCHMARKS_H 

#include <fstream>
#include <chrono>


void RunBasicMatmul();
void RunDatatypeBenchmark();
void Runi16fp16bf16Benchmark();
void RunTilingBenchmark();


template <typename Func>
double TimeMatmulLatencyMicroseconds(
        Func&& function, 
        size_t iterations = 100
)
{
    // warm up
    function();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        function();
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);

    return duration.count() / static_cast<double>(iterations);
}

// Source - https://stackoverflow.com/a/40760972
// Posted by Valdemar_Rudolfovich, modified by community. See post 'Timeline' for change history
// Retrieved 2026-08-25, License - CC BY-SA 4.0
class csvfile; // foward declaration

// inline: compiler inserts the function directly instead of calling it (today
// is more common the multiple definitions across translation units explana-
// tion)
inline static csvfile& endrow(csvfile& file);
inline static csvfile& flush(csvfile& file);

class csvfile
{
    // private by default (encapsulated)
    std::ofstream fs_;
    const std::string separator_;
public:
    // constructor() : ... everything after ":" is the member initializer list
    csvfile(const std::string filename, const std::string separator = ";")
        : fs_()
        , separator_(separator)
    {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        fs_.open(filename);
    }

    // destructor: runs automatically when the object is destroyed (or goes out
    // of scope)
    // RAII: resource acquisition is initialization
    ~csvfile()
    {
        flush();
        fs_.close();
    }

    void flush()
    {
        fs_.flush();
    }

    void endrow()
    {
        fs_ << std::endl;
    }

    // operator << : overloading the "<<" operator
    // csvfile& (* val)(csvfile&)) -> function pointer: val is a pointer to a 
    // function that takes csvfile& as an input and returns csvfile&
    // if you send a function to csvObject using << operator: file << endrow
    // then call that function on myself:
    //     file << endrow -> endrow(file) -> file << std::endl; returning after
    //     the same object
    //
    csvfile& operator << ( csvfile& (* val)(csvfile&))
    {
        return val(*this);
    }

    // string overload: define the behavior of the operator when passed C-like 
    // strings
    csvfile& operator << (const char * val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    // string overload for C++ std::string
    csvfile& operator << (const std::string & val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    // overload other types: for example, numbers do not need to be enclosed by
    // "..."
    template<typename T>
    csvfile& operator << (const T& val)
    {
        fs_ << val << separator_;
        return *this;
    }
};


inline static csvfile& endrow(csvfile& file)
{
    file.endrow();
    return file;
}

inline static csvfile& flush(csvfile& file)
{
    file.flush();
    return file;
}
#endif
