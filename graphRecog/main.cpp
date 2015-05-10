#include <iostream>

#include <opencv2/opencv.hpp>

#include "graphrecog.h"

// For Notebooks:
// Force Nvidia Optimus enabled devices to use
// dedicated graphics by default. (might crash when using 
// integrated intel graphics depending on driver)
extern "C" {
    _declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

int main(int argc, char** argv)
{
    // Read the image
    cv::Mat srcImg;
    if (argc < 2)
    {
        srcImg = cv::imread("test.bmp", 1);
    }
    else
    {        
        srcImg = cv::imread(argv[1], 1);
    }       
    if(!srcImg.data)
    {
        std::cerr<<"Invalid input image\n";
        return -1;
    }

    GraphRecog recog = GraphRecog(srcImg);
    
    cv::Mat baseImg = recog.getBaseImg();
    
    // Display Results of Hough transformation
    cv::Mat houghImg;
    cv::cvtColor(baseImg, houghImg, CV_GRAY2RGB);
    std::vector<cv::Vec3f> houghCircles = recog.houghCircles;
    for(size_t i = 0; i < houghCircles.size(); i++)
    {
        cv::Point center(cvRound(houghCircles[i][0]), cvRound(houghCircles[i][1]));

        std::cout << "(" << houghCircles[i][0] << "," << houghCircles[i][1] << "), " << houghCircles[i][2] << std::endl;
        // circle center
        circle(houghImg, center, 3, cv::Scalar(0,255,0), -1, 8, 0);
        // circle outline
        circle(houghImg, center, cvRound(houghCircles[i][2]), cv::Scalar(0,0,255), 3, 8, 0);
    }
    cv::imshow("Circles", houghImg);
    /*
    // Display Results of contour detection
    std::vector<std::vector<cv::Point>> contours = recog.contours;
    std::vector<cv::Vec4i> contourHierarchy = recog.contourHierarchy;
    cv::RNG rng(12345);
    cv::Mat contourImg = cv::Mat::zeros(baseImg.size(), CV_8UC3);
    for(size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
        drawContours(contourImg, contours, (int)i, color, 2, 8, contourHierarchy, 0, cv::Point());
        
        //cv::Mat contourImg2 = cv::Mat::zeros(baseImg.size(), CV_8UC3);
        //drawContours(contourImg2, contours, (int)i, color, 2, 8, contourHierarchy, 0, cv::Point());
        //cv::imshow("Contours" + i, contourImg2);
        
    }
    cv::imshow("All Contours", contourImg);
    */
    cv::waitKey();
    return 0;
}