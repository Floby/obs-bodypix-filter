#include "bodypix.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

Mat sigmoid (Mat x);

auto net = readNetFromTensorflow("../model.pb");

const float THRESHOLD = 0.6;

Mat getMask (Mat& input) {
  Mat blob = blobFromImage(input, 1.0/255, input.size(), Scalar(1.0, 1.0, 1.0), true, false);
  net.setInput(blob);
  auto results = net.forward("float_segments/conv");
  int sz[] = {results.size[2], results.size[3]};
  Mat segments(2, sz, results.type(), results.ptr<float>(0));
  resize(segments, segments, input.size());
  segments = sigmoid(segments);

  Mat bodypixmask;
  cv::threshold(segments, bodypixmask, THRESHOLD, 1, THRESH_BINARY);
  cvtColor(bodypixmask, bodypixmask, COLOR_GRAY2BGR);
  bodypixmask.convertTo(bodypixmask, CV_8UC1, 255);
  return bodypixmask;
}

Mat sigmoid (Mat x) {
  int height = x.rows;
  int width = x.cols;
  auto zeroes = UMat::zeros(height, width, CV_32F);
  auto ones = UMat::ones(height, width, CV_32F);

  Mat subbed;
  subtract(zeroes, x, subbed);
  Mat added;
  Mat exped;
  exp(subbed, exped);
  add(ones, exped, added);

  Mat divided;
  divide(ones, added, divided);
  return divided;
}
