// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
    using namespace std;

    enum PeronaMalikFluxFunctions
    {
        PERONA_MALIK_EXPONENTIAL = 0,
        PERONA_MALIK_INVERSE_QUADRATIC = 1
    };

    void PeronaMalik(InputArray src, OutputArray dst, double timeStepSize, double k, int noOfTimeSteps, int fluxFunc )
    {
        CV_Assert(k != 0);
        CV_Assert(timeStepSize > 0);
        CV_Assert(noOfTimeSteps >= 0);
        CV_Assert((fluxFunc == PERONA_MALIK_EXPONENTIAL) || (fluxFunc == PERONA_MALIK_INVERSE_QUADRATIC));
        CV_Assert(src.type() == CV_64FC1 || src.type() == CV_64FC3);

        Mat _src = src.getMat();
        dst.createSameSize(src, src.type());
        Mat _dst = dst.getMat();

        if 
        (
            _src.data == _dst.data 
        )
        {
            _src = _src.clone();
        }
        
        int channels = _src.channels();
        int nRows = _src.rows;
        int nCols = _src.cols * channels;
        
        double oneOnkSquared = 1 / (k * k);
        
        int i,j;
        for (int t = 0; t < noOfTimeSteps; t++)
        {
            for( i = 0; i < nRows; ++i)
            {
                double* p_src_S = _src.ptr<double>(min(i+1, nRows-1));
                double* p_src   = _src.ptr<double>(i);
                double* p_src_N = _src.ptr<double>(max(i-1, 0));
                double* p_dst   = _dst.ptr<double>(i);
                
                for ( j = 0; j < nCols; ++j)
                {
                    int ch = j % channels;
                    double center  = p_src[j];
                    double nablaN  = p_src_N[j]                         - center; 
                    double nablaS  = p_src_S[j]                         - center;
                    double nablaE  = p_src[min(j+channels,   nCols-ch)] - center;
                    double nablaW  = p_src[max(j-channels,   ch      )] - center;
                    double nablaNE = p_src_N[min(j+channels, nCols-ch)] - center;
                    double nablaSE = p_src_S[min(j+channels, nCols-ch)] - center;
                    double nablaNW = p_src_N[max(j-channels, ch      )] - center;
                    double nablaSW = p_src_S[max(j-channels, ch      )] - center;
                    
                    double cN, cS, cW, cE, cNE, cSE, cSW, cNW;
                    
                    if (fluxFunc == PERONA_MALIK_INVERSE_QUADRATIC)
                    {
                        cN  = 1 / ( 1 + (nablaN * nablaN * oneOnkSquared));
                        cS  = 1 / ( 1 + (nablaS * nablaS * oneOnkSquared));
                        cW  = 1 / ( 1 + (nablaW * nablaW * oneOnkSquared));
                        cE  = 1 / ( 1 + (nablaE * nablaE * oneOnkSquared));
                        cNE = 1 / ( 1 + (nablaNE * nablaNE * oneOnkSquared));
                        cSE = 1 / ( 1 + (nablaSE * nablaSE * oneOnkSquared));
                        cSW = 1 / ( 1 + (nablaSW * nablaSW * oneOnkSquared));
                        cNW = 1 / ( 1 + (nablaNW * nablaNW * oneOnkSquared));
                    }
                    else
                    {
                        cN  = exp(-nablaN * nablaN * oneOnkSquared);
                        cS  = exp(-nablaS * nablaS * oneOnkSquared);
                        cW  = exp(-nablaW * nablaW * oneOnkSquared);
                        cE  = exp(-nablaE * nablaE * oneOnkSquared);
                        cNE = exp(-nablaNE * nablaNE * oneOnkSquared);
                        cSE = exp(-nablaSE * nablaSE * oneOnkSquared);
                        cSW = exp(-nablaSW * nablaSW * oneOnkSquared);
                        cNW = exp(-nablaNW * nablaNW * oneOnkSquared);
                    }
                    
                    double delta = (nablaN * cN) + (nablaS * cS) + (nablaW * cW) + (nablaE * cE) + 0.5 * ((nablaNE * cNE) + (nablaSE * cSE) + (nablaSW * cSW) + (nablaNW * cNW)); 
                    p_dst[j] = center + timeStepSize * delta;
                }
            }
            swap(_src, _dst);
        }

        if (noOfTimeSteps % 2 == 0)
        {
            _src.copyTo(_dst);
        };
    };
}
