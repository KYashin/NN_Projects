#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QTimer>
#include <QtGui>

#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <typeinfo>
#include <queue>
#include <iterator>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#define ANGLE_RESOLUTION_X 0.0507
#define ANGLE_RESOLUTION_Y 0.0507

using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QGraphicsPixmapItem pixmap;
    QTimer *ptimer_MW;
    cv::Mat frame;
    cv::Mat frame_1;
    cv::VideoCapture capture;
    cv::dnn::Net net;
    bool flag;
    int baseLine;

    // для отрисовки yolov5
    const float INPUT_WIDTH = 416.0;
    const float INPUT_HEIGHT = 416.0;
    const float SCORE_THRESHOLD = 0.4;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.35;
    cv::Rect send_bbox;

    // Text parameters.
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS_FOR_RECT = 1;
    const int THICKNESS_FOR_TEXT = 1;
    const int FONT_SIZE = 10;

    // Colors.
    cv::Scalar BLACK = cv::Scalar(0,0,0);
    cv::Scalar BLUE = cv::Scalar(255, 178, 50);
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
    cv::Scalar RED = cv::Scalar(0,0,255);

    std::string net_name = "yolo-brig";
    cv::dnn::DetectionModel model;
    std::vector<std::string> class_list;
    bool SingleDetect = false;
    cv::Rect SingleBbox;
    int SingleClassId, SingleDetectCounter;
    QPoint click_position;
    bool setSingleBbox;

    std::vector<int> lefts;
    std::vector<int> tops;
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> centres_y;
    std::vector<int> centres_x;
    std::vector<int> class_indeces;
    std::vector<cv::Size> label_sizes;
    std::vector<string> labels;
    std::vector<int> indices_blue;
    std::vector<int> indices_bearing;

    bool which = false;
    const std::vector<cv::Scalar> colors = {cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

    int left_top_rectangle[4] = {0, 0, 640, 360};
    int right_top_rectangle[4] = {640, 0, 1280, 360};
    int left_bottom_rectangle[4] = {0, 360, 640, 720};
    int right_bottom_rectangle[4] = {640, 360, 1280, 720};

public slots:
    void start_detecting();
    void initialization();
    std::vector<std::string> load_class_list();
    void load_net(cv::dnn::Net &net, bool is_cuda);
    std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);
    void post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name);
    void draw_neuro_frame_new(cv::Mat &input_image);
};

#endif // MAINWINDOW_H
