#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "function.h"
#include <math.h>
using namespace std;
using namespace cv;
#include <iostream>
#include <fstream>
#include<time.h>
#include <ml.h>		  // opencv machine learning include file
#include<vector>  
#include <dirent.h>
#include <iterator>  
#include <time.h>
bool fileExists(const std::string& filename)
{
    std::ifstream file(filename);
    return file.good();
}
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
		string filename=string(dirp->d_name);
		if((filename.compare(".")!=0)&&(filename.compare("..")!=0))
		{
             files.push_back(string(dirp->d_name));
		}
    }
    closedir(dp);
    return 0;
}

FILE *stream_predict;
template <class RandomAccessIterator, class URNG>
void shuffle (RandomAccessIterator first, RandomAccessIterator last, URNG&& g)
{
  for (auto i = (last-first) - 1; i > 0; --i) {
    std::uniform_int_distribution<decltype(i)> d (0,i);
    swap (first[i], first[d (g)]);
  }
}


int main()
{
	
   
   CvRTrees* rtree = new CvRTrees;
	 rtree->load("/data9/nepal_predict/model_2025-v1.txt");

	//generate corresponding feature vector for each pixel
  //stream_predict = fopen("/data5/supplement_data/feature_nepal_predict.txt","w");
  std::string geology_path="/data5/supplement_data/geology_nepal_v2_0filled.tif";
  std::string slope_path="/data5/supplement_data/nepal_slopev2_filled0.tif";
  std::string pga_mean_path="/data5/sort2/pga_p50.tif";//p50;p10;p90
  std::string num_earq_path="/data5/sort2/num_p50.tif";
  std::string ssp126lcc_path="/data5/supplement_data/ssp126lcc-2030.tif";
  std::string ssp245lcc_path="/data5/supplement_data/ssp245lcc-2030.tif";
  std::string ssp370lcc_path="/data5/supplement_data/ssp370lcc-2030.tif";
  std::string ssp585lcc_path="/data5/supplement_data/ssp585lcc-2030.tif";
  std::string ssp126_mean_precipitation="/data5/supplement_data/ssp126_2030_means.tif";
  std::string ssp245_mean_precipitation="/data5/supplement_data/ssp245_2030_means.tif";
  std::string ssp370_mean_precipitation="/data5/supplement_data/ssp370_2030_means.tif";
  std::string ssp585_mean_precipitation="/data5/supplement_data/ssp585_2030_means.tif";
  std::string ssp126_1d_precipitation="/data5/supplement_data/ssp126_2030_1.tif";
  std::string ssp126_3d_precipitation="/data5/supplement_data/ssp126_2030_3.tif";
  std::string ssp126_5d_precipitation="/data5/supplement_data/ssp126_2030_5.tif";
  std::string ssp126_7d_precipitation="/data5/supplement_data/ssp126_2030_7.tif";
  std::string ssp126_10d_precipitation="/data5/supplement_data/ssp126_2030_10.tif";
  std::string ssp245_1d_precipitation="/data5/supplement_data/ssp245_2030_1.tif";
  std::string ssp245_3d_precipitation="/data5/supplement_data/ssp245_2030_3.tif";
  std::string ssp245_5d_precipitation="/data5/supplement_data/ssp245_2030_5.tif";
  std::string ssp245_7d_precipitation="/data5/supplement_data/ssp245_2030_7.tif";
  std::string ssp245_10d_precipitation="/data5/supplement_data/ssp245_2030_10.tif";
  std::string ssp370_1d_precipitation="/data5/supplement_data/ssp370_2030_1.tif";
  std::string ssp370_3d_precipitation="/data5/supplement_data/ssp370_2030_3.tif";
  std::string ssp370_5d_precipitation="/data5/supplement_data/ssp370_2030_5.tif";
  std::string ssp370_7d_precipitation="/data5/supplement_data/ssp370_2030_7.tif";
  std::string ssp370_10d_precipitation="/data5/supplement_data/ssp370_2030_10.tif";
  std::string ssp585_1d_precipitation="/data5/supplement_data/ssp585_2030_1.tif";
  std::string ssp585_3d_precipitation="/data5/supplement_data/ssp585_2030_3.tif";
  std::string ssp585_5d_precipitation="/data5/supplement_data/ssp585_2030_5.tif";
  std::string ssp585_7d_precipitation="/data5/supplement_data/ssp585_2030_7.tif";
  std::string ssp585_10d_precipitation="/data5/supplement_data/ssp585_2030_10.tif";
  
  std::string save_path="/data5/supplement_data/predict_ssp585_p50-2025v1.png";
  
  
  cv::Mat geology_mat=imread(geology_path.c_str(),0);
  cv::Mat slope_mat=imread(slope_path.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pga_mat=imread(pga_mean_path.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat num_earq_mat=imread(num_earq_path.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat lcc_mat=imread(ssp585lcc_path.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_mat=imread(ssp585_mean_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_1d=imread(ssp585_1d_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_3d=imread(ssp585_3d_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_5d=imread(ssp585_5d_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_7d=imread(ssp585_7d_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat pre_10d=imread(ssp585_10d_precipitation.c_str(),cv::IMREAD_UNCHANGED);
  cv::Mat result_mat(geology_mat.rows,geology_mat.cols,CV_16UC1,Scalar(0));         
          
  cout<<num_earq_path<<num_earq_mat.rows<<" "<<num_earq_mat.cols<<endl;
  //feature
  for(int i_index=0;i_index<geology_mat.rows;i_index++)
  {
     //cout<<i_index<<endl;
			for(int j_index=0;j_index<geology_mat.cols;j_index++)
			{
           int lcc_value=lcc_mat.at<uchar>(i_index,j_index);
           //cout<<lcc_value<<endl;
           if(lcc_value==255)
           {
              continue;
           }
            if(lcc_value==6)
           {
              continue;
           }
           
           //cout<<j_index<<endl;
           int geo_value=geology_mat.at<uchar>(i_index,j_index);
           //cout<<geo_value<<endl;
           if(geo_value==0)
           {
              continue;
           }
                     
  	       vector<float>features;
           float slope_value=slope_mat.at<float>(i_index,j_index);
           
           //cout<<slope_value<<endl;
           float pga_value=pga_mat.at<float>(i_index,j_index);
           if(pga_value>20)
           {
             continue;
           }
           //cout<<"pga    "<<pga_value<<endl;
           float num_earthquake_value=num_earq_mat.at<float>(i_index,j_index);
           //cout<<"num    "<<num_earthquake_value<<endl;
           float pre_value=pre_mat.at<float>(i_index,j_index);
           float pre_1d_value=pre_1d.at<float>(i_index,j_index);
           float pre_3d_value=pre_3d.at<float>(i_index,j_index);
           float pre_5d_value=pre_5d.at<float>(i_index,j_index);
           float pre_7d_value=pre_7d.at<float>(i_index,j_index);
           float pre_10d_value=pre_10d.at<float>(i_index,j_index);
          
  
           //cout<<"pre    "<<pre_value<<endl;
           features.push_back(float(geo_value));      
           features.push_back(num_earthquake_value);      
           features.push_back(pre_value);      
           features.push_back(pre_1d_value);      
           features.push_back(pre_3d_value); 
           features.push_back(pre_5d_value); 
           features.push_back(pre_7d_value); 
           features.push_back(pre_10d_value); 
           features.push_back(slope_value); 
           features.push_back(lcc_value); 
           features.push_back(pga_value); 
          
          
           cv::Mat feature_mat = cv::Mat::zeros(1, features.size(), CV_32F);
           for(int i_f=0;i_f<features.size();i_f++)
           {
               feature_mat.at<float>(0,i_f)=features[i_f];
           }
           //cout<<"finish loading feature"<<endl;          
   				 cv::Mat test_sample;
   				 test_sample=feature_mat.row(0);         
           float predict_prob=0.0;    
           predict_prob=rtree->predict_prob(test_sample,Mat());
          
           
           int predict_label=predict_prob*10000;      
			     result_mat.at<ushort>(i_index,j_index)=predict_label;
            //cout<<predict_label<<endl;
			 }
   }
 		imwrite(save_path.c_str(),result_mat);
   
    
  
	return 1;
}
