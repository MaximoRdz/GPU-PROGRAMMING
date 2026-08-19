/* Implement image processing function for (initially) 3x3 blurring */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "blur_cuda.h"


int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return 1;
    }
    
    const char* filepath = argv[1];

    cv::Mat raw = cv::imread(filepath);

    if (raw.empty()){
        std::cerr << "ERROR: Could not open image: " << filepath << std::endl;
        return 1;
    }

    cv::Mat image;
    double scale = 1024.0f / std::max(raw.cols, raw.rows);
    cv::resize(raw, image, cv::Size(), scale, scale);


    cv::imshow("Image Viewer", image);

    cv::waitKey(0);

    return 0;
}

