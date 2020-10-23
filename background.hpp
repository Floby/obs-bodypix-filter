#include <opencv2/opencv.hpp>

const int DEFAULT_SMOOTHING = 5;
cv::Mat removeBackground (cv::Mat frame, int smoothing = DEFAULT_SMOOTHING);
