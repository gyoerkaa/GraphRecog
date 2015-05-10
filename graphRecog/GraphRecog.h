#ifndef GRAPHRECOG_H
#define GRAPHRECOG_H

#include <opencv2/opencv.hpp>

//! @brief Graph recognition
//!
//! Testing algorithms for graph recognition in images
//! Will be ported to Java (Android)
//!
//! @author Attila Györkös
//!
class GraphRecog
{

public:
    GraphRecog(const cv::Mat& srcImg);
    ~GraphRecog(void);

    cv::Mat getBaseImg() const;

    // Private members, moved here for testing
        
    //! @brief Binarised and thinned image
    //!
    //! Source image after preprocessing (binarized and thinned)
    cv::Mat baseImg;

    //! @brief Holds results of hough circle detection
    std::vector<cv::Vec3f> houghCircles;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> contourHierarchy;

private:
    //! @brief Thinning iteration
    //! Performs both subiterations, depending on subIter
    //! parameter.
    //!
    //! @param img Binary image {0,1}
    //! @param subIter Which subiteration to perform (1 or 2)
    //! @sa thinning(cv::Mat& img, int iter)     
    void thinningIter(cv::Mat& img, int subIter);
    
    //! @brief Performs thinning
    //! Patterns in the source image are thinned to 1 Pixel. End points and
    //! pixel connectivity are preserved.
    //! Proposed by T.Y. Zhang and C.Y. Suen in
    //! "A fast parallel algorithm for thinning digital patterns"
    //! Communications of the ACM, March 1984, Vol. 27, No. 2, Page 236
    //!
    //! @param srcImg Source image (size > 3x3 , 1 channel)
    //! @param dstImg Thinned image
    //! @sa thinningIter(cv::Mat& img, int iter)   
    void thinning(const cv::Mat& srcImg, cv::Mat& dstImg);

    //! @brief Convert image to binary {0, 255}
    //! 
    //! @param srcImg Source image RGB
    //! @param dstImg Binary Image {0, 255}
    void toBinary(const cv::Mat& srcImg, cv::Mat& dstImg);

    void detectGNodes(const cv::Mat& srcImg);
    
    void detectGEdges(const cv::Mat& srcImg);

    void detectNodeLabel();
};

#endif GRAPHRECOG_H
