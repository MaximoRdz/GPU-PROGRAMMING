#include <algorithm>
#include <cstddef>
#include <vector>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "blur_cpu.hpp"


cv::Mat MakeLabeledPanel(const cv::Mat& gray, const std::string& label) {
    cv::Mat panel;
    cv::cvtColor(gray, panel, cv::COLOR_GRAY2BGR);
    cv::putText(panel, label, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, 
            cv::Scalar(0, 255, 0), 2);
    return panel;
}

void DisplayExample(cv::Mat raw)
{
    cv::Mat image;
    double scale = 800.0 / std::max(raw.cols, raw.rows);
    cv::resize(raw, image, cv::Size(), scale, scale);

    const std::vector<double> sigmas = {1.0, 2.5, 5.0};

    std::vector<cv::Mat> panels;
    panels.push_back(MakeLabeledPanel(image, "original"));

    for (const double sigma : sigmas) {
        const size_t kernel_size = KernelSizeForSigma(sigma);
        std::vector<double> kernel(kernel_size);

        ComputeGaussianKernel1D(kernel.data(), kernel_size, sigma);

        cv::Mat blurred(image.rows, image.cols, CV_8UC1);
        GaussianFilterCPU(image.data, blurred.data, image.rows, image.cols,
                kernel.data(), kernel_size);

        std::ostringstream label;
        label << "sigma = " << std::fixed << std::setprecision(1) << sigma;
        panels.push_back(MakeLabeledPanel(blurred, label.str()));
    }

    cv::Mat mosaic = cv::Mat::zeros(
        image.rows * 2,
        image.cols * 2,
        CV_8UC3
    );

    for (size_t i = 0; i < panels.size(); ++i) {
        int row = static_cast<int>(i) / 2;
        int col = static_cast<int>(i) % 2;

        cv::Mat roi = mosaic(
            cv::Rect(
                col * image.cols,
                row * image.rows,
                image.cols,
                image.rows
            )
        );

        panels[i].copyTo(roi);
    }
    cv::imshow("Gaussian Blur - increasing sigma", mosaic);
    cv::waitKey(0);
}
