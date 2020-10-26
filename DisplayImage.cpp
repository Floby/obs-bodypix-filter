#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "background.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

unsigned int BODYPIX_OUTPUT_STRIDE = 16;

Mat postProcessMask (Mat &mask, int smoothing);
void show (Mat toShow);

int main(int argc, char** argv ) {
  if (argc < 2) {
    cout << "You MUST provide an image as argument" << endl;
    exit(1);
  }
  string imageName = argv[1];
  Mat frame = cv::imread(imageName, 1);
  show (extractMask(frame, 5));
  show (removeBackground(frame, 5));
}


void show (Mat toShow) {
  imshow("SHOW", toShow);
  waitKey();
}
