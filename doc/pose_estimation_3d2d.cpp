//
// Created by hu on 2019/5/22.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

// ��������ת�����һ������
Point2d pixel2cam(const Point2d &p, const Mat &K);
//FileStorage config("../config.yaml", FileStorage::READ);
int main(int argc, char **argv) {
    //-- ��ȡͼ��

    Mat img_1 = imread("../data/test/trans_0_rgb.png");
    Mat img_2 = imread("../data/test/trans_5_rgb.png");
    Mat d1 = imread("../data/test/trans_0_depth.png",IMREAD_ANYDEPTH);       // ���ͼΪ16λ�޷���������ͨ��ͼ��
//    imshow("depth",d1*15);
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "һ���ҵ���" << matches.size() << "��ƥ���" << endl;
    // ����3D��
    Mat K = (Mat_<double>(3, 3) << 615.3, 0, 318.1, 0, 615.4, 241.8, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 1000.0;

        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
//        cout<<keypoints_2[m.trainIdx].pt<<endl;
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d good matches: " << pts_3d.size() << endl;

    Mat r, t;
    Mat r1,t1;
    solvePnPRansac(pts_3d,pts_2d,K,Mat(),r1,t1,false);
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false,SOLVEPNP_EPNP); // ����OpenCV �� PnP ��⣬��ѡ��EPNP��DLS�ȷ���
    Mat R;
    cv::Rodrigues(r, R); // rΪ��ת������ʽ����Rodrigues��ʽת��Ϊ����
    Mat R1;
    cv::Rodrigues(r1, R1); // rΪ��ת������ʽ����Rodrigues��ʽת��Ϊ����

    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "R1=" << endl << R1 << endl;
    cout << "t1=" << endl << t1 << endl;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch);
    imshow("�Ż���ƥ����", img_goodmatch);
    waitKey(0);



//    cout<<"calling bundle adjustment"<<endl;

//    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- ��ʼ��
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- ��һ��:��� Oriented FAST �ǵ�λ��
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ����
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- ���Ĳ�:ƥ����ɸѡ
    double min_dist = 10000, max_dist = 0;

    //�ҳ�����ƥ��֮�����С�����������, ���������Ƶĺ�����Ƶ������֮��ľ���
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);

    //��������֮��ľ��������������С����ʱ,����Ϊƥ������.����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }


}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}