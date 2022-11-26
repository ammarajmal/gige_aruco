#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{

    // cv::VideoCapture cap("filename.avi");
    VideoCapture cap(2);

    if (!cap.isOpened())
    {
        std::cout << "웹캠을 열 수 없습니다.\n";
        return -1;
    }


    Mat frame;

    bool ret = cap.read(frame);
    if (!ret)
    {
        cout << "캡쳐 실패" << endl;
        return -1;
    }


    VideoWriter writer;

    int fourcc = VideoWriter::fourcc('X','2','6','4');

    // 녹화된 파일이 빠르게 재생되는 현상은 웹캠의 fps를 가져와서 VideoWriter에 입력해주면 해결됨.
    // 웹캠에서 fps값을 가져오지 못하는 경우 cap.get(CAP_PROP_FPS)의 리턴값이 0이 됨. 수동으로 입력해줘야함.
    // https://answers.opencv.org/question/16522/video-recording-is-too-fast/
    double fps = cap.get(CAP_PROP_FPS);
    writer.open("output.avi", fourcc, fps, frame.size());
    if (!writer.isOpened()) {
        cout << "녹화를 준비할 수 없습니다.\n";
        return -1;
    }

    while(1)
    {
        auto total_start = chrono::steady_clock::now();

        bool ret = cap.read(frame);
        if (!ret)
        {
            cout << "캡쳐 실패" << endl;
            return -1;
        }

       
        writer.write(frame);
       
        auto total_end = chrono::steady_clock::now();


        float total_fps = 1000.0 / chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();


        std::ostringstream stats_ss;
        stats_ss << fixed << setprecision(2);
        stats_ss << "Total FPS: " << total_fps;
        auto stats = stats_ss.str();
         

        int baseline;
        auto stats_bg_sz = getTextSize(stats.c_str(), FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        rectangle(frame, Point(0, 0), Point(stats_bg_sz.width, stats_bg_sz.height + 10), Scalar(0, 0, 0), FILLED);
        putText(frame, stats.c_str(), Point(0, stats_bg_sz.height + 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));


        namedWindow("output");
        imshow("output", frame);
       

        int key = waitKey(1);
        if (key == 27) // ESC 키 누르면 중지
            break;
    }

    cap.release();
    writer.release();

    return 0;
}