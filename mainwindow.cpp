#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QCoreApplication>
#include <fstream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ptimer_MW = ptimer_MW = new QTimer();
    ptimer_MW->setInterval(1);
    ptimer_MW->start();

    ui->graphicsView->setScene(new QGraphicsScene(this));
    ui->graphicsView->scene()->addItem(&pixmap);
    ui->graphicsView->installEventFilter(this);

    connect(ptimer_MW, SIGNAL(timeout()), this, SLOT(start_detecting()));
    connect(ui->initialization, SIGNAL(clicked()), this, SLOT(initialization()));

//    capture.open("udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, "
//                 "payload=96 ! rtph265depay ! avdec_h265 ! decodebin ! videoconvert ! appsink",
//                 cv::CAP_GSTREAMER);

    capture.open("udpsrc port=5000 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink",
                 cv::CAP_GSTREAMER);

//    capture.open(0);

//    capture.open("udpsrc port=5000 caps = 'application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! decodebin ! videoconvert ! appsink", cv::CAP_GSTREAMER);

    flag = false; // создаем булевскую переменную для того, чтобы реализовать закрытие/открытие картинки одной кнопкой
    which = false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

std::vector<std::string> MainWindow::load_class_list() // функция, отвечающая за детектирование и вывод картинки на экран
{
    std::string path_to_list_of_classes_yolov5 = "/home/shine/Projects_Qt/Neuro/Kirill.txt";
    std::string path_to_list_of_classes_yolov4 = "config_files/" + net_name + ".txt";
    std::string path_to_list_of_classes;
    std::vector<std::string> class_list;
    which == true ? path_to_list_of_classes = path_to_list_of_classes_yolov4 : path_to_list_of_classes = path_to_list_of_classes_yolov5; // используется условие тренарного выбора:
                                                                                                                                         // Если переменная which равна true, то переменной path_to_list_of_classes присваивается значение "path_to_list_of_classes_yolov4".
                                                                                                                                         // В противном случае, если which равно false, то переменной path_to_list_of_classes присваивается значение "path_to_list_of_classes_yolov5".
    std::ifstream ifs(path_to_list_of_classes);
    /*
    Cтрока кода выше создает объект std::ifstream, который представляет собой потоковый ввод для чтения файлов в стандартной библиотеке шаблонов C++ (STL).
    Переменная ifs является объектом класса ifstream, который используется для чтения данных из файла.
    Путь к файлу передается в конструкторе объекта ifs через аргумент path_to_list_of_classes.
    */

    std::string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    /*
    Цикл выше считывает каждую строку из файла, используя объект ifs, который был создан ранее.
    Строка считывается до тех пор, пока не достигнут конец файла или не возникла ошибка.
    Каждая считанная строка добавляется в конец списка class_list с помощью метода push_back.
    */

    return class_list; // функция возвращает переменную class_list, которая является объектом класса vector<std::string>, в которой хранятся лейблы классов,
                       // загруженных из файла "/home/shine/Projects_Qt/Neuro/Kirill.txt";
}

void MainWindow::load_net(cv::dnn::Net &net, bool is_cuda) {
    if(which == true) // Если which == true, то загружается yolov4, в обратном случае - yolov5
    {
        auto result = cv::dnn::readNetFromDarknet("config_files/" + net_name + ".cfg", "config_files/" + net_name + ".weights");
    //    auto result = cv::dnn::readNetFromONNX("config_files/v5s_200.onnx");
        if (is_cuda)
        {
            std::cout << "Attempty to use CUDA\n";
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }

        else
        {
            std::cout << "Running on CPU\n";
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        net = result;
        model = cv::dnn::DetectionModel(net);
        model.setInputParams(1./255, cv::Size(416, 416), cv::Scalar(), true);
        class_list = load_class_list();
    }

    else
    {
         std::string path = "/home/shine/pt_to_onnx/best_opset_12.onnx";
         net = cv::dnn::readNet(path);

         if (is_cuda)
         {
             std::cout << "Attempty to use CUDA\n";
             net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
             net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
         }

         else
         {
             std::cout << "Running on CPU\n";
             net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // указывает, что предпочтительной средой выполнения является OpenCV, которая обычно используется для работы на GPU.
             net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // указывает, что предпочтительным целевым устройством является CPU.
                                                               // Таким образом, эти две строки кода говорят OpenCV использовать CPU для выполнения нейронной сети.
         }
    }
}

std::vector<cv::Mat> MainWindow::pre_process(cv::Mat &input_image, cv::dnn::Net &net) // на вход поступает frame из функции draw_neuro_frame_new()
{
    // Convert to blob.
    cv::Mat blob; // создается объект класса cv::Mat blob
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    /*
     Вызывает функцию cv::dnn::blobFromImage, которая преобразует изо2;бражение input_image в blob (двумерный массив).
     Параметр 1./255. используется для нормализации значений пикселей изображения (каждый пиксель делится на 255).
     Параметр cv::Size(INPUT_WIDTH, INPUT_HEIGHT) определяет размеры выходного изображения.
     Параметр cv::Scalar() используется для обрезки изображения (если пиксели выходят за пределы указанных размеров, они обрезаются).
     Флаг true указывает, что изображение должно быть нормализовано (среднее значение пикселей равно 0, стандартное отклонение равно 1).
     Флаг false указывает, что ориентация изображения не должна быть изменена.
    */

    net.setInput(blob); // Устанавливает входными данными для нейронной сети net созданный blob.

    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    /*
    Вызывается метод forward объекта net, который запускает процесс инференса.
    Аргументы outputs и net.getUnconnectedOutLayersNames() указывают, что результаты должны быть сохранены в векторе outputs,
    и что инференс должен быть проведен для всех несвязанных выходных слоев сети.
    */

    return outputs;
}

void MainWindow::post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    float *data = (float *)outputs[0].data;

//    qDebug() << data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
//    const int dimensions = 6;
    // 25200 for default size 416.
    const int rows = 19224; // 19224
//    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
//        qDebug() << i;
        float confidence = data[4];
//        std::cout << confidence << '\n';
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD && confidence < 1)
        {
            float* classes_scores = data + 5;
            // Create a 1x1 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 6;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);



    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        if (confidences[idx] < 0.99 && boxes[idx].x > 0 && boxes[idx].y > 0)
        {
            send_bbox = boxes[idx];

            int left = send_bbox.x;
            int top = send_bbox.y;
            int width = send_bbox.width;
            int height = send_bbox.height;
            int center_x = (left + left + width) / 2;
            int center_y = (top + top + height) / 2;

            // Draw bounding box.
            if (send_bbox.width * send_bbox.height > 100 && send_bbox.width > 0 && send_bbox.height > 0)
            {
                // Get the label for the class name and its confidence.
                std::string label = cv::format("%.2f", confidences[idx]);
                label = class_name[class_ids[idx]] + ":" + label;
                // Draw class labels.
                cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS_FOR_TEXT, &baseLine);

                class_indeces.insert(class_indeces.end(), class_ids[idx]);
                label_sizes.insert(label_sizes.end(), label_size);
                labels.insert(labels.end(), label);
                centres_x.insert(centres_x.end(), center_x);
                centres_y.insert(centres_y.end(), center_y);
                lefts.insert(lefts.end(), left);
                tops.insert(tops.end(), top);
                widths.insert(widths.end(), width);
                heights.insert(heights.end(), height);
            }
        }
    }
}

void MainWindow::draw_neuro_frame_new(cv::Mat &input_image) // поступает на вход frame из функции main
{
    std::vector<cv::Mat> detections;
    detections = pre_process(input_image, net); // далее этот frame идет в функцию pre_process
//    cv::Mat image = post_process(input_image, detections, class_list);        // вызывается функция, которая на вход принимает frame, detections, в котором хранятся
                                                                 // в котором хранятся outputs, возвращаемые функцией pre_process и class_list из функции
                                                                 // возвращает изображение разрисованное
    post_process(input_image, detections, class_list);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes).
//    std::vector<double> layersTimes;
//    double freq = cv::getTickFrequency() / 1000;
//    double t = net.getPerfProfile(layersTimes) / freq;
}

void MainWindow::start_detecting()
{
    if (flag == true)
    {
        class_list = load_class_list(); // вызывается функция load_class_list(), которая возвращает переменную class_list,
                                        // являющаяся объектом класса vector<std::string>, в которой хранятся лейблы классов,
                                        // загруженных из файла "/home/shine/Projects_Qt/Neuro/Kirill.txt";

        auto model = cv::dnn::DetectionModel(net); // эта строка создает экземпляр класса DetectionModel с использованием предоставленной нейронной сети net.
                                                   // DetectionModel - это класс, который предоставляет интерфейс для обнаружения объектов с использованием нейронной сети.

        model.setInputParams(1./255, cv::Size(416, 416), cv::Scalar(), true); // эта строка устанавливает параметры входного слоя модели. Параметры включают масштабирование пикселей (1/255),
                                                                              // размер изображения (416x416 пикселей), значение смещения (0, представленное как cv::Scalar())
                                                                              // и флаг, указывающий, что изображение должно быть нормализовано (true)

        capture.read(frame_1);

        frame = frame_1.clone();

        auto begin = std::chrono::steady_clock::now();

        if (frame.empty())
        {
            std::cout << "End of stream\n";
            QApplication::quit();
        }

//        cv::resize(frame, frame, cv::Size(1920, 1080), cv::INTER_LINEAR);

        draw_neuro_frame_new(frame);

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); // конвертирование картинки из BGR в RGB

        QImage qimg(frame.data,
                    frame.cols,
                    frame.rows,
                    frame.step,
                    QImage::Format_RGB888);

        QPainter painter(&qimg);
        QPen pen(Qt::black);

        pen.setWidth(5); // устанавливаем толщину линии
        painter.setPen(pen); // устанавливаем перо на QPainter
//        painter.drawLine(1270, 0, 1270, 1080);
//        painter.drawLine(650, 0, 650, 1080);
        painter.drawLine(0, 340, 1920, 340);
        painter.drawLine(0, 740, 1920, 740);

//        int center_of_blue_detail_x = 0;
//        int center_of_blue_detail_y = 0;
//        int center_of_bearing_x = 0;
//        int center_of_bearing_y = 0;
//        int center_of_screw_x = 0;
//        int center_of_screw_y = 0;

        int index_blue;
        int index_bearing;

        for (int i = 0; i < class_indeces.size(); i++)
        {
            if (class_indeces[i] == 0)
            {
                index_blue = i;
            }

            if (class_indeces[i] == 1)
            {
                index_bearing = i;
            }
        }

        bool position = false;
//        bool between_lines_blue = false;
//        bool between_lines_bearing = false;

        if (centres_x.size() - 1 >= std::max(index_blue, index_bearing))
        {
            position = (centres_x[index_blue] < centres_x[index_bearing]);
//            between_lines_blue = (340 <= centres_y[index_blue] && centres_y[index_blue] <= 740);
//            between_lines_bearing = (340 <= centres_y[index_bearing] && centres_y[index_bearing] <= 740);
        }


        for (int i = 0; i < labels.size(); i++)
        {
            QRect rect(QPoint(lefts.at(i), tops.at(i)), QPoint(lefts[i] + widths[i], tops[i] + heights[i]));
            pen.setColor(Qt::red);

            if (position)
            {
                pen.setColor(Qt::green);
            }

            pen.setWidth(THICKNESS_FOR_RECT);
            painter.setPen(pen);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(rect);

            int top = cv::max(tops[i], label_sizes[i].height);

            // Top left corner.
            QPoint tlc = QPoint(lefts[i], top);
            // Bottom right corner.
            QPoint brc = QPoint(lefts[i] + label_sizes[i].width, top - label_sizes[i].height);

            painter.setBrush(Qt::red);

            if (position)
            {
                painter.setBrush(Qt::green);
            }

            QRect rect_1(tlc, brc);
            painter.drawRect(rect_1);

            painter.setPen(Qt::black);
            painter.setFont(QFont("Arial", FONT_SIZE));
            painter.drawText(lefts[i] + 4, top - 2, QString::fromStdString(labels[i]));
        }

        lefts.clear();
        tops.clear();
        widths.clear();
        heights.clear();
        indices_blue.clear();
        indices_bearing.clear();
        centres_y.clear();
        centres_x.clear();
        label_sizes.clear();
        labels.clear();
        class_indeces.clear();

        pixmap.setPixmap(QPixmap::fromImage(qimg));                     // вывод картинки 'qimg' с детекцией в graphics view
        ui->graphicsView->fitInView(&pixmap, Qt::IgnoreAspectRatio);

        auto end = std::chrono::steady_clock::now();

        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        qDebug() << "The time: " << elapsed_ms.count() << " ms\n";
    }
}

void MainWindow::initialization()
{
    flag = !flag; // меняет значение булевской переменной на противоположное при нажатии на кнопку

    if (flag == true) // здесь происходит инициализация нейронной сети
    {
        ui->initialization->setText("Завершить детектирование");

        if(!capture.isOpened())
        {
            std::cerr << "Error opening video file\n";
            QApplication::quit();
        }

        bool is_cuda = false;

         load_net(net, is_cuda);
    }

    else // если значение переменной flag == false, то изображение в graphicsView меняется на белый фон
    {
        ui->initialization->setText("Начать детектирование");
        QImage img;
        img.fill(Qt::white);
        pixmap.setPixmap(QPixmap::fromImage(img));
        ui->graphicsView->fitInView(&pixmap, Qt::IgnoreAspectRatio);
    }
}
