#include "GraphRecog.h"
//#include "HoughTrans.h" $(OPENCV_DIR)

#include <algorithm>
#include <iostream>

#include <typeinfo>


GraphRecog::GraphRecog(const cv::Mat& srcImg)
{
    this->toBinary(srcImg, this->baseImg);
    this->thinning(this->baseImg, this->baseImg);
    this->detectGNodes(this->baseImg);
    //this->detectGEdges(this->baseImg);
}


GraphRecog::~GraphRecog(void)
{
}


cv::Mat GraphRecog::getBaseImg() const
{
    return this->baseImg;
}


void GraphRecog::thinningIter(cv::Mat& img, int subIter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat deletions = cv::Mat::zeros(img.size(), CV_8UC1);  
        
    // 3x3 Window
    uchar *p9, *p2, *p3; //       North
    uchar *p8, *p1, *p4; // West    +    East
    uchar *p7, *p6, *p5; //       South

    uchar *northRow = NULL;               // Northern row
    uchar *currRow  = img.ptr<uchar>(0);  // Current row
    uchar *southRow = img.ptr<uchar>(1);  // Southern row
    
    int x, y;
    for (y = 1; y < img.rows-1; ++y) 
    {
        // Shift rows up (moves 3x3 matrix down)
        northRow = currRow;
        currRow  = southRow;
        southRow = img.ptr<uchar>(y+1);

        // Initialize column pointers
        p2 = &(northRow[0]);  p3 = &(northRow[1]);      
        p1 = &(currRow[0]);   p4 = &(currRow[1]);      
        p6 = &(southRow[0]);  p5 = &(southRow[1]);
        
        for (x = 1; x < img.cols-1; ++x) 
        {
            // Shift columns left (moves 3x3 matrix right)
            p9 = p2;  p2 = p3;  p3 = &(northRow[x+1]);
            p8 = p1;  p1 = p4;  p4 = &(currRow[x+1]);
            p7 = p6;  p6 = p5;  p5 = &(southRow[x+1]);

            // Condition A: Number of 0->1 patterns in 
            //              the ordered set {p2,...p9}
            int A = (*p2 == 0 && *p3 == 1) + 
                    (*p3 == 0 && *p4 == 1) +
                    (*p4 == 0 && *p5 == 1) + 
                    (*p5 == 0 && *p6 == 1) +
                    (*p6 == 0 && *p7 == 1) + 
                    (*p7 == 0 && *p8 == 1) +
                    (*p8 == 0 && *p9 == 1) + 
                    (*p9 == 0 && *p2 == 1);
            // Condition B: Number of non-zero neighbours of p1
            int B = *p2 + *p3 + *p4 + *p5 + *p6 + *p7 + *p8 + *p9;
            // Condition C (dependent on subiteration)
            int C = subIter == 1 ? (*p2 * *p4 * *p6) : (*p2 * *p4 * *p8);
            // Condition D (dependent on subiteration)
            int D = subIter == 1 ? (*p4 * *p6 * *p8) : (*p2 * *p6 * *p8);

            // Check conditions. If all are met, delete point (x,y)
            if (A == 1 && (B >= 2 && B <= 6) && C == 0 && D == 0)
            {
                deletions.ptr<uchar>(y)[x] = 1;
            }
        }
    }
    img &= ~deletions;
}


void GraphRecog::thinning(const cv::Mat& srcImg, cv::Mat& dstImg)
{
    dstImg = srcImg.clone();
    dstImg /= 255; // convert to {0,1} binary image

    cv::Mat prevImg = cv::Mat::zeros(dstImg.size(), CV_8UC1);
    cv::Mat diffImg;

    do
    {
        this->thinningIter(dstImg, 1);
        this->thinningIter(dstImg, 2);
        cv::absdiff(dstImg, prevImg, diffImg);
        dstImg.copyTo(prevImg);
    }
    while (cv::countNonZero(diffImg) > 0);

    dstImg *= 255; // convert back to {0,255} binary image
}


void GraphRecog::toBinary(const cv::Mat& srcImg, cv::Mat& dstImg)
{
    cv::cvtColor(srcImg, dstImg, CV_BGR2GRAY);
    cv::threshold(dstImg, dstImg, 10, 255, CV_THRESH_BINARY);
}


void GraphRecog::detectGNodes(const cv::Mat& srcImg)
{
    int accumulatorThreshold = 5;
    int cannyThreshold = 50;

    cv::Mat tmpImg = srcImg.clone(); // for testing only, this is bad
    
    // Reduce the noise to avoid false circle detection
    //GaussianBlur(tmpImg, tmpImg, cv::Size(9, 9), 2, 2);

    HoughCircles(tmpImg, houghCircles, CV_HOUGH_GRADIENT, 1, srcImg.rows/8, cannyThreshold, accumulatorThreshold, 0, 0);
    //HoughTrans::myHoughCircles(tmpImg, houghCircles, 1, srcImg.rows/8, cannyThreshold, accumulatorThreshold, 0, 0);

    // Calculate radius Median. Discard large/small circles.
    // We can assume that the nodes are of similar size. 
    if (!houghCircles.empty()) 
    {
        size_t n = houghCircles.size() / 2;
        // partial sort for linear avg complexity
        std::nth_element(houghCircles.begin(), 
                         houghCircles.begin()+n, 
                         houghCircles.end(),
                         [](cv::Vec3f c1,cv::Vec3f c2)->bool{return c1[2]<c2[2];} );
        float radMedian = floor(houghCircles[n][2]);
    
        std::vector<cv::Vec3f>::iterator it;
        for(it = houghCircles.begin(); it != houghCircles.end(); ++it) 
        {       
            if ( fabs((*it)[2]-radMedian) > 0.25f *radMedian) 
            {
                (*it)[0] = 0.0f;
                (*it)[1] = 0.0f;
                (*it)[2] = 0.0f;            
            }
            else
            {
                (*it)[0] = floor((*it)[0]);
                (*it)[1] = floor((*it)[1]);
                (*it)[2] = floor((*it)[2]);            
            }
        }

        // TODO: Detect circle intersections with Bentley-Ottoman-alogorithm
        std::sort(houghCircles.begin(),
                  houghCircles.end(),
                  [](cv::Vec3f c1, cv::Vec3f c2)->bool{return c1[0]<c2[0];} );
    }  
}


void GraphRecog::detectGEdges(const cv::Mat& srcImg)
{
    int thresh = 100;
    int max_thresh = 255;

    cv::Mat tmpImg = srcImg.clone(); // for testing only, this is bad

    // TODO: Remove detected circles nodes from the image. 
    //       Only edges will will countours (and maybe letters/labels)
        
    findContours(tmpImg, contours, contourHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}


void GraphRecog::detectNodeLabel()
{
}
