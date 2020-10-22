#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

cv::Mat removeBackground (cv::Mat frame, cv::Mat mask) {
  cv::Mat invertedMask = mask.clone();
  cv::subtract(cv::Scalar::all(255), mask, invertedMask);

  cv::Mat result = frame.clone();
  cv::multiply(frame, mask, result, 1.0/255);
  return result;
}

void runStuff (string basename, Net net) {
  std::string frameName = basename + ".jpg";
  std::string maskName = basename + ".mask.jpg";
  cv::Mat frame = cv::imread( frameName, 1 );
  cv::Mat mask = cv::imread( maskName, 1 );

  //Mat blob = blobFromImage(frame, 1.0, frame.size(), Scalar(1.0, 1.0, 1.0), true, false);
  Mat blob;
  /*
  imshow("example", frame);
  cv::waitKey(0);
  */
  blobFromImage(frame, blob, 1/255);
  cout << frame.size() << ' ' << blob.size() << '\n';
  net.setInput(blob);
  auto results = net.forward("float_segments/conv");
  //results.resize(480);
  cout << "shape " << results.size() << '\n';
  Mat bodypixmask = frame.clone();
  cv::threshold(results, bodypixmask, 0.70, 1, THRESH_BINARY);
  cvtColor(bodypixmask, bodypixmask, COLOR_GRAY2BGR);

  cv::Mat backgroundRemoved = removeBackground(frame, mask);

  if ( !frame.data || !mask.data ) {
    printf("No image data \n");
  }

  //cv::imshow("Background removed", backgroundRemoved);
  //cv::waitKey(0);
}

int main(int argc, char** argv ) {
  auto net = readNetFromTensorflow("../model.pb");

  while (true) {
    runStuff("/home/floby/Images/TestFloby/A", net);
    cout << "waiting...\n";
    sleep(1);
  }

  return 0;
}

