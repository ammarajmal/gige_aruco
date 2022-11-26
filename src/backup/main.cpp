// #include "CameraApi.h" // API header file of camera SDK
#include "CameraApi.h"
#include "opencv2/core.hpp"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui.hpp"
#include <iomanip>
#include <stdio.h>
#include <iostream>

// using namespace cv;
using namespace std;

unsigned char           * g_pRgbBuffer;     //processed data cache

int main()
{

    int                     iCameraCounts = 1;
    int                     iStatus=-1;
    tSdkCameraDevInfo       tCameraEnumList;
    int                     hCamera;
    tSdkCameraCapbility     tCapability;      //Device description information
    tSdkFrameHead           sFrameInfo;
    BYTE*			        pbyBuffer;
    int                     iDisplayFrames = 10000;
    IplImage *iplImage = NULL;
    int                     channel=3;

    CameraSdkInit(1);

    
    //Enumerate devices and create a list of devices
    iStatus = CameraEnumerateDevice(&tCameraEnumList,&iCameraCounts);
	printf("state = %d\n", iStatus);

	printf("count = %d\n", iCameraCounts);
    
    // no device connected
    if(iCameraCounts==0){
        return -1;
    }

    //Camera initialization. After the initialization is successful, 
    // any other camera-related operation interface can be called
    iStatus = CameraInit(&tCameraEnumList,-1,-1,&hCamera);

    //initialization failed
	printf("state = %d\n", iStatus);
    if(iStatus!=CAMERA_STATUS_SUCCESS){
        return -1;
    }
    //Get the camera's characteristic description structure. 
    // This structure contains the range information of various parameters that can be set
    //  by the camera. Determines the parameters of the relevant function
    
    CameraGetCapability(hCamera,&tCapability);

    //
    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);
    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

     /* Let the SDK enter the working mode and start receiving images sent from the camera
     data. If the current camera is in trigger mode, it needs to receive
     The image is not updated until the frame is triggered. */
    CameraPlay(hCamera);

    /*Other camera parameter settings
     For example CameraSetExposureTime CameraGetExposureTime set/read exposure time
          CameraSetImageResolution CameraGetImageResolution Set/read resolution
          CameraSetGamma, CameraSetConrast, CameraSetGain, etc. set image gamma, contrast, RGB digital gain, etc.
          This routine is just to demonstrate how to convert the image obtained in the SDK into the OpenCV image format, so as to call the OpenCV image processing function for subsequent development
     */

    if(tCapability.sIspCapacity.bMonoSensor){
        channel=1;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel=3;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
    }


// Loop to display 1000 frames of images
    while(iDisplayFrames--)
    {
        auto total_start = chrono::steady_clock::now();

        if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
		{
		    CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);
		    
		    cv::Mat matImage(
					cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight), 
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
					);

            auto total_end = chrono::steady_clock::now();
            float total_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();
            
            std::ostringstream stats_ss;
            
            stats_ss << fixed << setprecision(2);
            stats_ss << "Total FPS: " << total_fps;
            auto stats = stats_ss.str();
            int baseline;
            auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            
            cv::rectangle(frame, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
            cv::putText(frame, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));
            namedWindow("Opencv Demo");
			imshow("Opencv Demo", matImage);

            int key = waitKey(1);
            if (key == 27) // ESP stop
                break;

            // After successfully calling CameraGetImageBuffer, 
            // you must call CameraReleaseImageBuffer to release the obtained buffer.
            //Otherwise, when calling CameraGetImageBuffer again, the program will 
            // be suspended and blocked until other threads call CameraReleaseImageBuffer to release the buffer
            
			CameraReleaseImageBuffer(hCamera,pbyBuffer);

		}
    }

    CameraUnInit(hCamera);
    //Note, free after deinitialization
    
    free(g_pRgbBuffer);


    return 0;
}

