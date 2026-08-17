#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

void myGrayImage(const cv::Mat& image, cv::Mat& output_image)
{
    // RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
    for (int r=0; r < image.rows; ++r){
        for (int c=0; c < image.cols; ++c){
            cv::Vec3b pixel_value = image.at<cv::Vec3b>(r, c);
            uchar B = pixel_value[0];
            uchar G = pixel_value[1];
            uchar R = pixel_value[2];

            output_image.at<uchar>(r, c) = 0.299f * R + 0.587 * G + 0.114 * B;
        }
    }
}

void myGrayImageOpt(const cv::Mat& image, cv::Mat& output_image)
{
    // RGB[A] to Gray:Y←0.299⋅R+0.587⋅G+0.114⋅B
    for (int r=0; r < image.rows; ++r){
        const uchar* src_row = image.ptr<uchar>(r);
        uchar* dst_row = output_image.ptr<uchar>(r);
        for (int c=0; c < image.cols; ++c){
            uchar B = src_row[c * 3 + 0];
            uchar G = src_row[c * 3 + 1];
            uchar R = src_row[c * 3 + 2];

            dst_row[c] = (uchar)(0.299f * R + 0.587 * G + 0.114 * B);
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    cv::Mat image = cv::imread(filename);

    if (image.empty()){
        std::cerr << "Error: Could not load image: "
                  << filename << std::endl;
        return 1;
    }

    cv::Mat resized;
    double scale = 1024.0f / std::max(image.cols, image.rows);
    cv::resize(image, resized, cv::Size(), scale, scale);

    cv::Mat gray_image(resized.rows, resized.cols, CV_8UC1);

    const size_t iterations = 1000;
    cv::cvtColor(resized, gray_image, cv::COLOR_BGR2GRAY);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        cv::cvtColor(resized, gray_image, cv::COLOR_BGR2GRAY);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_execution_time = duration.count() / (double)iterations;

    std::cout << "OpenCV Average time per call: " << avg_execution_time << " microseconds\n";

    
    myGrayImage(resized, gray_image);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        myGrayImage(resized, gray_image);
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_execution_time = duration.count() / (double)iterations;

    std::cout << "myGrayImage Average time per call: " << avg_execution_time << " microseconds\n";

    myGrayImageOpt(resized, gray_image);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i < iterations; ++i){
        myGrayImageOpt(resized, gray_image);
    }
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_execution_time = duration.count() / (double)iterations;

    std::cout << "myGrayImageOpt Average time per call: " << avg_execution_time << " microseconds\n";

    // Finally some viz
    myGrayImage(resized, gray_image);

    cv::imshow("Image Viewer", gray_image);

    cv::waitKey(0);

    return 0;
}
