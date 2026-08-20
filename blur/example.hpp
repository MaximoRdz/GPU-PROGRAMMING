#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <opencv2/opencv.hpp>

cv::Mat MakeLabeledPanel(const cv::Mat&, const std::string&);

void DisplayExample(cv::Mat);

#endif
