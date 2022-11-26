#include "CameraApi.h"
#include <iomanip>
#include <stdio.h>
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"


using namespace std;
using namespace cv;
int markNum(1);
int waitTime(50);
const float calibrationSquareDimension = 0.0245f; // meters
const float arucoSquareDimension = 0.1016f; // meters
const Size chessboardDimensions = Size(6,9);
unsigned char           * g_pRgbBuffer;     //processed data cache

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners){
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j ++) {
            corners.push_back(Point3f( j * squareEdgeLength, i * squareEdgeLength, 0.0f ));
        }
    }
}
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false){
    for  (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++){
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CALIB_CB_ADAPTIVE_THRESH);
        // bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE); // for fast speed
        
        if (found){
            allFoundCorners.push_back(pointBuf);
        }
        if (showResults){
            drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
            imshow("Look for Corners", *iter);
            waitKey(0);
        }
    }
}
Mat markerGenerator(int markNum){
    Mat markerImage;
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_1000);
    aruco::drawMarker(dictionary, markNum, 200, markerImage, 1);
    imwrite("marker0.png", markerImage);
    return markerImage;
}
void createArucoMarkers(){
    Mat outputMarker;
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    for (int i = 0; i<50; i++){
        aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
        ostringstream convert;
        string imageName = "Marker_4x4_";
        convert << imageName << i << ".jpg";
        imwrite(convert.str(), outputMarker);
    }
}
void markerDetector(int camera){
    VideoCapture inputVideo;
    inputVideo.open(camera);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_100);
    while (inputVideo.grab()) {
    Mat image, imageCopy;
    inputVideo.retrieve(image);
    image.copyTo(imageCopy);
    vector<int> ids;
    vector<vector<Point2f> > corners;
    aruco::detectMarkers(image, dictionary, corners, ids);
    
    // if at least one marker detected
    if (ids.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
    imshow("out", imageCopy);
    char key = (char) cv::waitKey(50);
    if (key == 'q')
        break;
    }
}
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients){
    vector<vector<Point2f>> checkerboardImageSpacePoints;
    getChessboardCorners(calibrationImages, checkerboardImageSpacePoints, false);
    vector<vector<Point3f>> worldSpaceCornerPoints(1);
    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);
    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8, 1, CV_64F);
    calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients){
    ofstream outstream(name);
    if (outstream){
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;
        outstream << rows << endl;
        outstream << columns << endl;

        for (int r = 0; r < rows; r++){
            for(int c = 0; c< columns ; c++){
                double value = cameraMatrix.at<double>(r,c);
                outstream << value << endl;

            }
        }
        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;
        outstream << rows << endl;
        outstream << columns << endl;
        for (int r = 0; r < rows; r++){
            for(int c = 0; c< columns ; c++){
                double value = distanceCoefficients.at<double>(r,c);
                outstream << value << endl;

            }
        }
        outstream.close();
        return true;
    }
    return false;
}
void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
    Mat frame;
    Mat drawToFrame;
    // Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    // Mat distanceCoefficients;
    vector<Mat> savedImages;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    VideoCapture vid(2);
    if (!vid.isOpened())
        return;
    int framesPerSecond = 30;
    namedWindow("webcam", WINDOW_AUTOSIZE);
    while (true){
        if (!vid.read(frame))
            break;
        vector<Vec2f> foundPoints;
        bool found = false;
        found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
        if (found)
            imshow("webcam", drawToFrame);
        else
            imshow("webcam", frame);
        char character = waitKey(1000/framesPerSecond);
            // 1. Save an image(Space Key) -- if we see a valid checkerboard image
            // 2. Start Camera clibration (Enter Key) -- if we wanted to start camera calibration so long as we have enough images like 10 or 15
            // 3. Exit(Escape Key)
        switch (character) {
            case ' ' :{
                //saving image
                if (found){
                    Mat temp;
                    frame.copyTo(temp);
                    savedImages.push_back(temp);
                }
                break;
            }
            case 13:{
                // start calibration
                if (savedImages.size() > 15) {
                    cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
                    saveCameraCalibration("my_camera_calibration", cameraMatrix, distanceCoefficients);
                }
                break;
            }
            case 27:{
                //exit 
                return;
                break;
            }
        }
    }
    return;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients){
    ifstream instream(name);
    if (instream){
        uint16_t rows;
        uint16_t columns;
        instream >> rows;
        instream >> columns;
        cameraMatrix = Mat(Size(columns, rows), CV_64F);
        for (int r = 0 ; r< rows; r++){
            for (int c =0; c < columns; c++){
                double read = 0.0f;
                instream >> read;
                cameraMatrix.at<double>(r,c) = read;
                cout<< cameraMatrix.at<double>(r, c) << endl;
            }
        }
        // Distance Coefficients
        instream >> rows;
        instream >> columns;
        distanceCoefficients = Mat::zeros(rows, columns, CV_64F);
        for (int r = 0; r < rows; r ++ ){
            for (int c = 0; c < columns ; c++){
                double read = 0.0f;
                instream >> read;
                distanceCoefficients.at<double>(r,c) = read;
                cout<< distanceCoefficients.at<double>(r,c) << endl;
            }
        }
        instream.close();
        return true;
    }
    return false;
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension) {
    Mat frame;
    vector<int>  markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    aruco::DetectorParameters parameters;
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    
    VideoCapture vid(2);
    if (!vid.isOpened())
        return -1;
    namedWindow("webcam", WINDOW_AUTOSIZE);

    vector<Vec3d> rotationVectors, translationVectors;

    while (true){
        if (!vid.read(frame))
            break;
        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);

        for (int i =0 ; i < markerIds.size(); i++)
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
        imshow("webcam", frame);
        if (waitKey(30)>=0)
            break;
    }
    return 1;
}

int gige_cam()
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
    while(true)
    {

        if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
		{
		    CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);
		    
		    cv::Mat matImage(
					cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight), 
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
					);

            cv::namedWindow("Opencv Demo");
			imshow("Opencv Demo", matImage);

            int key = cv::waitKey(1);
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
int gige_fpsCal()
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
        For example CameraSetExposureTime CameraGetExposureTime set/read exposure 
        time CameraSetImageResolution CameraGetImageResolution Set/read resolution
        CameraSetGamma, CameraSetConrast, CameraSetGain, etc. set image gamma, 
        contrast, RGB digital gain, etc.
        
        This routine is just to demonstrate how to convert the image obtained in the 
        SDK into the OpenCV image format, so as to call the OpenCV image processing 
        function for subsequent development
     */

    if(tCapability.sIspCapacity.bMonoSensor){
        channel=1;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    }else{
        channel=3;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
    }


    while(1)
    {

        auto total_start = chrono::steady_clock::now();

// *********************************************************
    if(CameraGetImageBuffer(hCamera,&sFrameInfo,&pbyBuffer,1000) == CAMERA_STATUS_SUCCESS)
            {
                CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer,&sFrameInfo);
                
                cv::Mat matImage(
                        cvSize(sFrameInfo.iWidth,sFrameInfo.iHeight), 
                        sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                        g_pRgbBuffer
                        );
                
                auto total_end_gige = chrono::steady_clock::now();


                float total_fps_gige = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end_gige - total_start).count();


                std::ostringstream stats_ss;
                stats_ss << fixed << setprecision(2);
                stats_ss << "Total FPS: " << total_fps_gige;
                auto stats = stats_ss.str();
                

                int baseline;
                auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                rectangle(matImage, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
                putText(matImage, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));

                cout<< stats.c_str()<<endl;
                cv::namedWindow("Opencv Demo");
                imshow("Opencv Demo", matImage);

                int key = cv::waitKey(1);
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

int main() {
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;

    cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
}