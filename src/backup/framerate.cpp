#include <iostream>
#include "opencv2/opencv.hpp"
#include <time.h>
using namespace std;
using namespace cv;

int main (int argc, char** argv){

    VideoCapture video(0);
    if (!video.isOpened()){
        printf("Cannot open the camera");
        return -1;
    }

    double fps = video.get(CAP_PROP_FPS);
    cout << "Frames per second :" << fps << endl;
    int num_frames = 1;
    clock_t start;
    clock_t end;
    Mat frame;
    cout << "Capturing " << num_frames << "frames" << endl;
    double ms, fpslive;
    while (true){
        start = clock();
        video >> frame;
        imshow("Webcam", frame);
        
        end = clock();
        double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        cout << "Time Taken: " << seconds << " seconds" << endl;
        fpslive = double(1) / double(seconds);
        cout<< "Estimated frames per second: " << fpslive << endl;
        putText(frame, "FPS: " + to_string(fpslive), {50, 50}, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0));
        imshow("Webcam", frame);
        int keyboard = waitKey(30);
        if (keyboard == 'q' or keyboard == 27) {
            break;
        }
    }
    return 0;
}