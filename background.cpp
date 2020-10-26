#include "background.hpp"
#include "bodypix.hpp"

using namespace cv;

const int NORMAL_HEIGHT = 420;

Mat hflip (Mat& original) {
  Mat flipped;
  flip(original, flipped, 1);
  return flipped;
}

Mat postProcessMask (Mat &mask, int smoothing) {
  Mat processed;
  dilate(mask, processed, UMat::ones(smoothing, smoothing, CV_8UC1));
  blur(processed, processed, Size(smoothing, smoothing), Point(-1,-1), BorderTypes::BORDER_REPLICATE);
  return processed;
}

Mat addAlpha (Mat image, Mat alpha) {
  std::vector<Mat> channels;
  std::vector<Mat> alphaChannels;
  split(image, channels);
  split(alpha, alphaChannels);
  channels[3] = alphaChannels[0];
  Mat transparent;
  merge(channels, transparent);
  waitKey();
  return transparent;
}

cv::Mat removeBackground (cv::Mat frame, int smoothing) {
  Mat result, processingFrame;
  Size originalSize = frame.size();
  if (originalSize.height > NORMAL_HEIGHT) {
    float resizeFactor = 1.0 * NORMAL_HEIGHT / originalSize.height;
    int processingHeight = static_cast<int>(originalSize.height * resizeFactor);
    int processingWidth = static_cast<int>(originalSize.width * resizeFactor);
    resize(frame, processingFrame, Size(processingWidth, processingHeight), 0, 0, INTER_AREA);
  } else {
    processingFrame = frame;
  }
  Mat flipped = hflip(processingFrame);
  Mat leftMask = getMask(processingFrame);
  Mat rightMask = getMask(flipped);
  rightMask = hflip(rightMask);
  Mat mask = leftMask*0.5 + rightMask*0.5;
  threshold(mask, mask, 250, 255, THRESH_BINARY);

  mask = postProcessMask(mask, smoothing);
  Mat resizedMask;
  resize(mask, resizedMask, frame.size(), 0, 0, INTER_CUBIC);

  multiply(frame, resizedMask, result, 1.0/255);
  return result;
}

