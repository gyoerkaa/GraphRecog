#ifndef HOUGHTRANS_H
#define HOUGHTRANS_H

#include <opencv2/opencv.hpp>

namespace HoughTrans
{        
    void myHoughCircles(cv::InputArray _image, cv::OutputArray _circles,
                        double dp, double minDist,
                        double param1=100, double param2=100,
                        int minRadius=0, int maxRadius=0);
};

#endif HOUGHTRANS_H