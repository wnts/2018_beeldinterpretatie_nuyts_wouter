#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::ml;

const String keys("{help h usage ? | | print this message }"
                  "{@train         |<none>| training file}"
                  "{@test          |<none>| test file}");

Mat img_train;
bool fg = true;
vector<Point> pts_fg, pts_bg;

static void onMouse( int event, int x, int y, int, void* userdata)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        if(fg)
            pts_fg.push_back(Point(x, y));
        else
            pts_bg.push_back(Point(x, y));
        return;
    }

    if(event == EVENT_MBUTTONDOWN)
    {
        vector<Point>& pts = fg ? pts_fg : pts_bg;
        String str = fg ? "=====FG PTS=====" : "===== BG PTS =====";

        cout << str << endl;
        for(Point pt : pts)
            cout << pt << endl;
        return;
    }

    if(event == EVENT_RBUTTONDOWN)
    {
        vector<Point>& pts = fg ? pts_fg : pts_bg;

        if(pts.size() > 0)
            pts.pop_back();
        return;
    }
}

int main(int argc, char * argv[])
{
    CommandLineParser parser(argc, argv, keys);
    Mat img_test;


    String path_train = parser.get<String>("@train");
    String path_test = parser.get<String>("@test");

    if(path_train.empty() or path_test.empty())
    {
        cerr << "Please provide all arguments" << endl;
        return 1;
    }

    img_train = imread(path_train);
    if(img_train.empty())
    {
        cerr <<  "Could not open or find the input image with path '" + path_train + "'" << std::endl ;
        return -1;
    }
    img_test = imread(path_test);
    if(img_test.empty())
    {
        cerr <<  "Could not open or find the template image with path '" + path_train + "'" << std::endl ;
        return -1;
    }
    /* Gaussian blur toevoegen om te vermijden dat je niet op groene pitten kan klikken */
    GaussianBlur(img_test, img_test, Size(5, 5), 0);
    GaussianBlur(img_train, img_train, Size(5, 5), 0);

    /** Training **/
    /// Voorgrond pixels kiezen
    namedWindow("train");
    imshow("train", img_train);
    setMouseCallback("train", onMouse, &img_train);
    waitKey(0);
    /// Achtergrond pixels kiezen
    fg = false;
    waitKey(0);

    /** Trainingsdata maken: als descriptors van een pixel nemen we de HSV waarde van de pixel **/
    Mat img_train_hsv;
    cvtColor(img_train, img_train_hsv, COLOR_BGR2HSV);

    Mat desc_fg(pts_fg.size(), 3, CV_32FC1);
    /// foreground pixels worden als 1 geclassificeerd
    Mat labels_fg = Mat::ones(pts_fg.size(), 1, CV_32SC1);
    for(int i = 0; i < pts_fg.size(); i++)
    {
        Vec3b hsv = img_train_hsv.at<Vec3b>(pts_fg[i].y, pts_fg[i].x);
        desc_fg.at<float>(i, 0) = hsv[0];
        desc_fg.at<float>(i, 1) = hsv[1];
        desc_fg.at<float>(i, 2) = hsv[2];
    }
    Mat desc_bg(pts_bg.size(), 3, CV_32FC1);
    /// background pixels worden als 0 geclassifeerd
    Mat labels_bg = Mat::zeros(pts_bg.size(), 1, CV_32SC1);
    for(int i = 0; i < pts_bg.size(); i++)
    {
        Vec3b hsv = img_train_hsv.at<Vec3b>(pts_bg[i].y, pts_bg[i].x);
        desc_bg.at<float>(i, 0) = hsv[0];
        desc_bg.at<float>(i, 1) = hsv[1];
        desc_bg.at<float>(i, 2) = hsv[2];
    }
    /// trainData maken volgens formaat dat API verwacht
    Mat trainingData, labels;
    // hierboven gemaakte descriptor matrices (fg en bg) concateneren
    vconcat(desc_fg, desc_bg, trainingData);
    vconcat(labels_fg, labels_bg, labels);
    Ptr<TrainData> trainData = TrainData::create(trainingData, ROW_SAMPLE, labels);
    cout << "Training data: " << endl
         << "getNSamples\t" << trainData->getNSamples() <<endl
         << "getSamples\n"  << trainData->getSamples()  <<endl
         << endl;

    /** Classifiers maken en trainen **/
    /// K-Nearest-Neighbours
    Ptr<KNearest> knn = KNearest::create();
    knn->setIsClassifier(true);
    knn->setDefaultK(3);
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);
    knn->train(trainData);


    /// Normal Bayes
    Ptr<NormalBayesClassifier> bayes = NormalBayesClassifier::create();
    bayes->train(trainData);
    /// SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainData);

    /** Classifiers toepasssen op input afbeelding **/
    Mat img_test_hsv;
    cvtColor(img_test, img_test_hsv, COLOR_BGR2HSV);
    Mat desc_test(1, 3, CV_32FC1);
    Mat results_knn, results_bayes;
    Mat outputProbs;
    Mat img_result_knn(img_test.rows, img_test.cols, CV_8UC3);
    Mat img_result_bayes(img_test.rows, img_test.cols, CV_8UC3);
    Mat img_result_svm(img_test.rows, img_test.cols, CV_8UC3);
    for(int i = 0; i < img_test.rows; i++)
    {
        for(int j = 0; j < img_test.cols; j++)
        {
            Vec3b hsv = img_test_hsv.at<Vec3b>(i, j);
            desc_test.at<float>(0, 0) = hsv[0];
            desc_test.at<float>(0, 1) = hsv[1];
            desc_test.at<float>(0, 2) = hsv[2];
            /// KNN toepassen
            knn->findNearest(desc_test, knn->getDefaultK(), results_knn);
            if(results_knn.at<float>(0, 0) == 1)
                img_result_knn.at<Vec3b>(i, j) = img_test.at<Vec3b>(i, j);
            else
                img_result_knn.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            /// Bayes toepassen

            if((int)bayes->predict(desc_test) == 1)
            {
            	img_result_bayes.at<Vec3b>(i, j) = img_test.at<Vec3b>(i, j);
            }
            /// SVM toepassen
            //cerr << svm->predict(desc_test) << endl;
            if((int)svm->predict(desc_test) == 1)
                img_result_svm.at<Vec3b>(i, j) = img_test.at<Vec3b>(i, j);
            else
                img_result_svm.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

        }
    }

    imshow("Resultaat KNN", img_result_knn);
    imshow("Resultaat Bayes", img_result_bayes);
    imshow("Resultaat SVM", img_result_svm);

    /* extra opdracht: groene pixels eruit filteren */

    Mat tmp, mask;

    cvtColor(img_result_knn, tmp, COLOR_BGR2HSV);
    // Hue tussen 30 en 90 (= van 60 graden tot 180 in Hue cirkel) komen overeen met groen
    inRange(tmp, Scalar(30, 0, 0), Scalar(90, 255, 255), mask);
    img_result_knn.setTo(Scalar(0,0,0), mask);
    cvtColor(img_result_bayes, tmp, COLOR_BGR2HSV);
	inRange(tmp, Scalar(30, 0, 0), Scalar(90, 255, 255), mask);
	img_result_bayes.setTo(Scalar(0,0,0), mask);
	cvtColor(img_result_svm, tmp, COLOR_BGR2HSV);
	inRange(tmp, Scalar(30, 0, 0), Scalar(90, 255, 255), mask);
	img_result_svm.setTo(Scalar(0,0,0), mask);


    imshow("Resultaat KNN (met extra kleursgementatie)", img_result_knn);
    imshow("Resultaat Bayes (met extra kleursgementatie)", img_result_bayes);
    imshow("Resultaat SVM (met extra kleursgementatie)", img_result_svm);


    waitKey(0);
    return 0;
}
