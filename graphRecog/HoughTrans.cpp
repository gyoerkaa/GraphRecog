#include "HoughTrans.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/internal.hpp>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#define hough_cmp_gt(l1,l2) (aux[l1] > aux[l2])

namespace HoughTrans
{

static CV_IMPLEMENT_QSORT_EX( icvHoughSortDescent32s, int, hough_cmp_gt, const int* )

static void myIcvHoughCirclesGradient(CvMat* img, float dp, float min_dist,
                                      int min_radius, int max_radius,
                                      int canny_threshold, int acc_threshold,
                                      CvSeq* circles, int circles_max)
{
    const int SHIFT = 10, ONE = 1 << SHIFT;
    cv::Ptr<CvMat> dx, dy;
    cv::Ptr<CvMat> edges, accum, dist_buf;
    std::vector<int> sort_buf;
    cv::Ptr<CvMemStorage> storage;

    int x, y, i, j, k, center_count, nz_count;
    float min_radius2 = (float)min_radius*min_radius;
    float max_radius2 = (float)max_radius*max_radius;
    int rows, cols, arows, acols;
    int astep, *adata;
    float* ddata;
    CvSeq *nz, *centers;
    float idp, dr;
    CvSeqReader reader;
    
    edges = cvCloneMat(img);
    //dx = cvCloneMat(img);
    //dy = cvCloneMat(img);
    
    /*
    // We don't need canny, we use skeletonization

    edges = cvCreateMat( img->rows, img->cols, CV_8UC1);
    cvCanny( img, edges, MAX(canny_threshold/2,1), canny_threshold, 3 );

    */
    dx = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    dy = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    cvSobel( img, dx, 1, 0, 3 );
    cvSobel( img, dy, 0, 1, 3 );
    
    if( dp < 1.f )
        dp = 1.f;
    idp = 1.f/dp;
    accum = cvCreateMat( cvCeil(img->rows*idp)+2, cvCeil(img->cols*idp)+2, CV_32SC1 );
    cvZero(accum);

    storage = cvCreateMemStorage();
    nz = cvCreateSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage );
    centers = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), storage );

    rows = img->rows;
    cols = img->cols;
    arows = accum->rows - 2;
    acols = accum->cols - 2;
    adata = accum->data.i;
    astep = accum->step/sizeof(adata[0]);

    // Accumulate circle evidence for each edge pixel
    for( y = 0; y < rows; y++ )
    {
        const uchar* edges_row = edges->data.ptr + y*edges->step;
        const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
        const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);

        for( x = 0; x < cols; x++ )
        {
            float vx, vy;
            int sx, sy, x0, y0, x1, y1, r;
            CvPoint pt;

            vx = dx_row[x];
            vy = dy_row[x];

            if( !edges_row[x] || (vx == 0 && vy == 0) )
                continue;

            float mag = sqrt(vx*vx+vy*vy);
            assert( mag >= 1 );
            sx = cvRound((vx*idp)*ONE/mag);
            sy = cvRound((vy*idp)*ONE/mag);

            x0 = cvRound((x*idp)*ONE);
            y0 = cvRound((y*idp)*ONE);
            // Step from min_radius to max_radius in both directions of the gradient
            for(int k1 = 0; k1 < 2; k1++ )
            {
                x1 = x0 + min_radius * sx;
                y1 = y0 + min_radius * sy;

                for( r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++ )
                {
                    int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
                    if( (unsigned)x2 >= (unsigned)acols ||
                        (unsigned)y2 >= (unsigned)arows )
                        break;
                    adata[y2*astep + x2]++;
                }

                sx = -sx; sy = -sy;
            }

            pt.x = x; pt.y = y;
            cvSeqPush( nz, &pt );
        }
    }

    nz_count = nz->total;
    if( !nz_count )
        return;
    //Find possible circle centers
    for( y = 1; y < arows - 1; y++ )
    {
        for( x = 1; x < acols - 1; x++ )
        {
            int base = y*(acols+2) + x;
            if( adata[base] > acc_threshold &&
                adata[base] > adata[base-1] && adata[base] > adata[base+1] &&
                adata[base] > adata[base-acols-2] && adata[base] > adata[base+acols+2] )
                cvSeqPush(centers, &base);
        }
    }

    center_count = centers->total;
    if( !center_count )
        return;

    sort_buf.resize( MAX(center_count,nz_count) );
    cvCvtSeqToArray( centers, &sort_buf[0] );

    icvHoughSortDescent32s( &sort_buf[0], center_count, adata );
    cvClearSeq( centers );
    cvSeqPushMulti( centers, &sort_buf[0], center_count );

    dist_buf = cvCreateMat( 1, nz_count, CV_32FC1 );
    ddata = dist_buf->data.fl;

    dr = dp;
    min_dist = MAX( min_dist, dp );
    min_dist *= min_dist;
    // For each found possible center
    // Estimate radius and check support
    for( i = 0; i < centers->total; i++ )
    {
        int ofs = *(int*)cvGetSeqElem( centers, i );
        y = ofs/(acols+2);
        x = ofs - (y)*(acols+2);
        //Calculate circle's center in pixels
        float cx = (float)((x + 0.5f)*dp), cy = (float)(( y + 0.5f )*dp);
        float start_dist, dist_sum;
        float r_best = 0;
        int max_count = 0;
        // Check distance with previously detected circles
        for( j = 0; j < circles->total; j++ )
        {
            float* c = (float*)cvGetSeqElem( circles, j );
            if( (c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist )
                break;
        }

        if( j < circles->total )
            continue;
        // Estimate best radius
        cvStartReadSeq( nz, &reader );
        for( j = k = 0; j < nz_count; j++ )
        {
            CvPoint pt;
            float _dx, _dy, _r2;
            CV_READ_SEQ_ELEM( pt, reader );
            _dx = cx - pt.x; _dy = cy - pt.y;
            _r2 = _dx*_dx + _dy*_dy;
            if(min_radius2 <= _r2 && _r2 <= max_radius2 )
            {
                ddata[k] = _r2;
                sort_buf[k] = k;
                k++;
            }
        }

        int nz_count1 = k, start_idx = nz_count1 - 1;
        if( nz_count1 == 0 )
            continue;
        dist_buf->cols = nz_count1;
        cvPow( dist_buf, dist_buf, 0.5 );
        icvHoughSortDescent32s( &sort_buf[0], nz_count1, (int*)ddata );

        dist_sum = start_dist = ddata[sort_buf[nz_count1-1]];
        for( j = nz_count1 - 2; j >= 0; j-- )
        {
            float d = ddata[sort_buf[j]];

            if( d > max_radius )
                break;

            if( d - start_dist > dr )
            {
                float r_cur = ddata[sort_buf[(j + start_idx)/2]];
                if( (start_idx - j)*r_best >= max_count*r_cur ||
                    (r_best < FLT_EPSILON && start_idx - j >= max_count) )
                {
                    r_best = r_cur;
                    max_count = start_idx - j;
                }
                start_dist = d;
                start_idx = j;
                dist_sum = 0;
            }
            dist_sum += d;
        }
        // Check if the circle has enough support
        if( max_count > acc_threshold )
        {
            float c[3];
            c[0] = cx;
            c[1] = cy;
            c[2] = (float)r_best;
            cvSeqPush( circles, c );
            if( circles->total > circles_max )
                return;
        }
    }
}

CV_IMPL CvSeq* myHoughCircles2(CvArr* src_image, void* circle_storage,
                               double dp, double min_dist,
                               double param1, double param2,
                               int min_radius, int max_radius)
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* circles = 0;
    CvSeq circles_header;
    CvSeqBlock circles_block;
    int circles_max = INT_MAX;
    int canny_threshold = cvRound(param1);
    int acc_threshold = cvRound(param2);

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    min_radius = MAX( min_radius, 0 );
    if( max_radius <= 0 )
        max_radius = MAX( img->rows, img->cols );
    else if( max_radius <= min_radius )
        max_radius = min_radius + 2;

    if( CV_IS_STORAGE( circle_storage ))
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else if( CV_IS_MAT( circle_storage ))
    {
        mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    myIcvHoughCirclesGradient(img, (float)dp, (float)min_dist,
                              min_radius, max_radius, canny_threshold,
                              acc_threshold, circles, circles_max);

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = circles->total;
        else
            mat->rows = circles->total;
    }
    else
        result = circles;

    return result;
}


const int STORAGE_SIZE = 1 << 12;


static void seqToMat(const CvSeq* seq, cv::OutputArray _arr)
{
    if( seq && seq->total > 0 )
    {
        _arr.create(1, seq->total, seq->flags, -1, true);
        cv::Mat arr = _arr.getMat();
        cvCvtSeqToArray(seq, arr.data);
    }
    else
        _arr.release();
}

void myHoughCircles(cv::InputArray _image, cv::OutputArray _circles,
                    double dp, double minDist,
                    double param1, double param2,
                    int minRadius, int maxRadius)
{
    cv::Ptr<CvMemStorage> storage = cvCreateMemStorage(STORAGE_SIZE);
    cv::Mat image = _image.getMat();
    CvMat c_image = image;
    CvSeq* seq = myHoughCircles2(&c_image, storage, 
                                 dp, minDist, 
                                 param1, param2, 
                                 minRadius, maxRadius);
    seqToMat(seq, _circles);    
}

}