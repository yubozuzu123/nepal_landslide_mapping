
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


int main()
{
	
 //load rf model
	 CvRTrees* rtree = new CvRTrees;
	 rtree->load("/data2/nepal_national_30year/origin_image/sub_with_gt/classification_model.txt");/
   
   std::string root_path="/data2/nepal_national_30year/origin_image/";

  int year_start=2018;
  string data_scale[3]={"79E27N","83E26N","85E26N"};
  for(int scale_index=0;scale_index<3;scale_index++)
  {
        cout<<"processing index:"<<scale_index<<endl;
      for(int year_index=0;year_index<1;year_index++)
    {
       int year=year_start+year_index;
       int year_next=year_start+(year_index+2);
       cout<<"processing year: "<<year<<endl;
       stringstream year_ss;
       year_ss<<year;
       string year_str=year_ss.str();
       stringstream year_next_ss;
       year_next_ss<<year_next;
       string year_next_str=year_next_ss.str();
         //preprocess
       std::string ppl_standard_path=root_path+data_scale[scale_index]+"2015_ppl.tif";
       Mat ppl_standard_mat=imread(ppl_standard_path.c_str(),cv::IMREAD_UNCHANGED);
       Mat ppl_standard_bi(ppl_standard_mat.rows,ppl_standard_mat.cols,CV_8UC1,Scalar(0));
       Mat ppl_standard_bi2(ppl_standard_mat.rows,ppl_standard_mat.cols,CV_8UC1,Scalar(0));
       
       std::string ppl_next_path=root_path+data_scale[scale_index]+year_next_str+"_potential_landslide_origin.tif";
       cv::Mat ppl_next_mat=imread(ppl_next_path.c_str(),cv::IMREAD_UNCHANGED);
       Mat ppl_binary(ppl_next_mat.rows,ppl_next_mat.cols,CV_8UC1,Scalar(0));
       for(int i=0;i<ppl_next_mat.rows;i++)
       {
         for(int j=0;j<ppl_next_mat.cols;j++)
         {
            if(ppl_next_mat.at<float>(i,j)>0)
            {
              ppl_binary.at<uchar>(i,j)=255;
            }
            if(ppl_standard_mat.at<float>(i,j)>0.55)
            {
              ppl_standard_bi.at<uchar>(i,j)=255;
            }
         }
       }
       
       std::string ls_poten_path=root_path+data_scale[scale_index]+year_next_str+"_potential_landslide_int.tif";
       cv::Mat ls_poten_result=imread(ls_poten_path.c_str(),cv::IMREAD_UNCHANGED);
       
       vector<vector<Point>> contours_std;
       vector<Vec4i> hierarchy_std;  
       findContours(ppl_standard_bi, contours_std,hierarchy_std, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
       for (size_t i = 0; i < contours_std.size(); i++)
     	{
     		double area = cv::contourArea(contours_std[i]);
         cv::Rect rect = cv::boundingRect(contours_std[i]);
         int height=rect.height;
         int width=rect.width;
         float ratio=area/float(height*width);
         if((area>5000)&&(ratio<=0.1))
         {
            cout<<"area"<<area<<endl;
            cout<<"ratio"<<ratio<<endl;
            drawContours(ppl_standard_bi2, contours_std, i, Scalar(255), CV_FILLED, 8,hierarchy_std);
         }
       }
       
    
        for(int i=0;i<ls_poten_result.rows;i++)
       {
          for(int j=0;j<ls_poten_result.cols;j++)
          {
             if(ppl_standard_bi2.at<uchar>(i,j))
             {
                ls_poten_result.at<uchar>(i,j)=0;
                ppl_binary.at<uchar>(i,j)=0;
             }
          }
       }
   	   vector<vector<Point>> contours;
       vector<Vec4i> hierarchy;  
       findContours(ppl_binary, contours,hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
      	for (size_t i = 0; i < contours.size(); i++)
     	{
     	   double area = cv::contourArea(contours[i]);
         cv::Rect rect = cv::boundingRect(contours[i]);
         int height=rect.height;
         int width=rect.width;
         float ratio=area/float(height*width);
          
           if(area==1)
          {   
               drawContours(ls_poten_result, contours, i, Scalar(0), CV_FILLED, 8,hierarchy);               
           }
       }
    
       cv::Mat result_mat=Mat(ls_poten_result.rows,ls_poten_result.cols, CV_8UC1,Scalar(0));  
       
       //load in original image and feature of year and year_next for feature generation
       cout<<"loading data..."<<endl;
      cv::Mat origin_year_mat[6];
      cv::Mat feature_year_mat[11];
      cv::Mat origin_year_next_mat[6];
      cv::Mat feature_year_next_mat[11];
      cv::Mat pca_feature_mat[4];
      
        for(int i_channel=0;i_channel<6;i_channel++)
      {
           stringstream channel_ss;
           channel_ss<<i_channel;
           string channel_str=channel_ss.str();
           std::string origin_landsat_year_path=root_path+"image_band_feature/"+data_scale[scale_index]+year_str+"-01-01comp_"+channel_str+".tif";
           std::string origin_landsat_year_next_path=root_path+"image_band_feature/"+data_scale[scale_index]+year_next_str+"-01-01comp_"+channel_str+".tif";
           origin_year_mat[i_channel]=imread(origin_landsat_year_path.c_str(),cv::IMREAD_UNCHANGED);
           origin_year_next_mat[i_channel]=imread(origin_landsat_year_next_path.c_str(),cv::IMREAD_UNCHANGED);
            cout<<"image no problem "<< origin_year_mat[i_channel].rows<<" "<< origin_year_next_mat[i_channel].cols<<endl;
      }
       cout<<"finish initializing..origin image."<<endl;
       cout<<"start loading...pca..."<<endl;
        for(int i_channel=10;i_channel<14;i_channel++)
      {
           stringstream channel_ss;
           channel_ss<<i_channel;
           string channel_str=channel_ss.str();
           std::string feature_landsat_year_next_path=root_path+"image_band_feature/"+data_scale[scale_index]+year_next_str+"-01-01comp_feature_file_"+channel_str+".tif";
           pca_feature_mat[i_channel-10]=imread(feature_landsat_year_next_path.c_str(),cv::IMREAD_UNCHANGED);
           cout<<"image no problem "<<  pca_feature_mat[i_channel-10].rows<<" "<<  pca_feature_mat[i_channel-10].cols<<endl;
      }
        cout<<"finish initializing..pca image."<<endl;
        cout<<"start to load feature...."<<endl;
        for(int i_channel=0;i_channel<11;i_channel++)
      {
          int i_channel_use=i_channel;
           stringstream channel_ss;
           if(i_channel==10)
           {
             i_channel=14;
           }
           channel_ss<<i_channel;
           string channel_str=channel_ss.str();
           std::string feature_landsat_year_path=root_path+"image_band_feature/"+data_scale[scale_index]+year_str+"-01-01comp_feature_file_"+channel_str+".tif";
           std::string feature_landsat_year_next_path=root_path+"image_band_feature/"+data_scale[scale_index]+year_next_str+"-01-01comp_feature_file_"+channel_str+".tif";
           feature_year_mat[i_channel_use]=imread(feature_landsat_year_path.c_str(),cv::IMREAD_UNCHANGED);
           feature_year_next_mat[i_channel_use]=imread(feature_landsat_year_next_path.c_str(),cv::IMREAD_UNCHANGED);
           cout<<"image no problem "<<feature_year_mat[i_channel_use].rows<<" "<<feature_year_next_mat[i_channel].cols<<endl;
      }
      cout<<"finish initializing..feature image."<<endl;
      
   for(int i_index=4;i_index<(ls_poten_result.rows-5);i_index++)
       {
         for(int j_index=4;j_index<(ls_poten_result.cols-5);j_index++)
         {
            int b3_bi_value=ls_poten_result.at<uchar>(i_index,j_index);
            if(!b3_bi_value)
            {
               continue;
            }
       			vector<float>features;
          for(int i_channel=0;i_channel<4;i_channel++)
        {
          float pca_feature_temp=pca_feature_mat[i_channel].at<float>(i_index,j_index);
          features.push_back(pca_feature_temp);
        }
        for(int i_channel=0;i_channel<6;i_channel++)
        {
          
           float origin_float=origin_year_mat[i_channel].at<float>(i_index,j_index);
           float origin_next_float=origin_year_next_mat[i_channel].at<float>(i_index,j_index);
           float origin_diff=origin_next_float-origin_float;
           float origin_diff_inv=origin_float-origin_next_float;
           if(origin_diff<0)
           {
              origin_diff=0.0;
           }
           
           if(origin_diff_inv<0)
           {
              origin_diff_inv=0.0;
           }
           
           features.push_back(origin_float);
           features.push_back(origin_next_float);
           features.push_back(origin_diff);
           features.push_back(origin_diff_inv);
           
           vector<float> origin_diff_nei;
           vector<float> origin_diff_inv_nei;
           float mean_origin_diff=0.0;
           float origin_diff_var=0.0;
           float mean_origin_diff_inv=0.0;
           float origin_diff_inv_var=0.0;
          
           	for(int i_ind=-4;i_ind<=4;i_ind++)
  				{
  					for(int j_ind=-4;j_ind<=4;j_ind++)
  					{
              
  						float origin_nei_float=origin_year_mat[i_channel].at<float>(i_index+i_ind,j_index+j_ind);                                                                            
              float origin_nei_next_float=origin_year_next_mat[i_channel].at<float>(i_index+i_ind,j_index+j_ind);     
              float origin_nei_diff=origin_nei_next_float-origin_nei_float;             
              float origin_nei_diff_inv=origin_nei_float-origin_nei_next_float;
          
              if(origin_nei_diff<0)
              {
                 origin_nei_diff=0.0;
              }

               if(origin_nei_diff_inv<0)
              {
                 origin_nei_diff_inv=0.0;
              }

              if(origin_nei_diff>1000)
              {
                cout<<"index: "<<i_index+i_ind<<" "<<j_index+j_ind<<endl;
                 break;
              }
             
              origin_diff_nei.push_back(origin_nei_diff);
               origin_diff_inv_nei.push_back(origin_nei_diff_inv);
              features.push_back(origin_nei_diff);
              features.push_back(origin_nei_diff_inv);
  					  mean_origin_diff=mean_origin_diff+origin_nei_diff;  
              mean_origin_diff_inv=mean_origin_diff_inv+origin_nei_diff_inv;                                                     
  					}
  				}
         
          mean_origin_diff=mean_origin_diff/float(origin_diff_nei.size());
           mean_origin_diff_inv=mean_origin_diff_inv/float(origin_diff_nei.size());
          features.push_back(mean_origin_diff);
          features.push_back(mean_origin_diff_inv);
      
          for(int i_vec=0;i_vec<origin_diff_nei.size();i_vec++)
          {
              origin_diff_var=origin_diff_var+(origin_diff_nei[i_vec]-mean_origin_diff)*(origin_diff_nei[i_vec]-mean_origin_diff);  
              origin_diff_inv_var=origin_diff_inv_var+(origin_diff_inv_nei[i_vec]-mean_origin_diff_inv)*(origin_diff_inv_nei[i_vec]-mean_origin_diff_inv);
            
          }
          features.push_back(origin_diff_var);       
          features.push_back(origin_diff_inv_var);
      }
        
        
         for(int i_channel=0;i_channel<11;i_channel++)
        {
          
           float feature_float=feature_year_mat[i_channel].at<float>(i_index,j_index);
           float feature_next_float=feature_year_next_mat[i_channel].at<float>(i_index,j_index);
           float feature_diff=feature_next_float-feature_float;
           float feature_diff_inv=feature_float-feature_next_float;
        
            if(feature_diff<0)
           {
              feature_diff=0.0;
           }
          
            if(feature_diff_inv<0)
           {
              feature_diff_inv=0.0;
           }
                  
           features.push_back(feature_float);          
           features.push_back(feature_next_float);                
           features.push_back(feature_diff);
           features.push_back(feature_diff_inv);
           
        
       }
      
        cv::Mat feature_mat = cv::Mat::zeros(1, features.size(), CV_32F);
        for(int i_f=0;i_f<features.size();i_f++)
        {
            feature_mat.at<float>(0,i_f)=features[i_f];
        }
                
				cv::Mat test_sample;
				test_sample=feature_mat.row(0);
				int predict_label=int(rtree->predict(test_sample,Mat()));
				result_mat.at<uchar>(i_index,j_index)=predict_label;
		     
         }
      }
    
      
      

    
        string result_path=root_path+data_scale[scale_index]+year_next_str+"-result_classification.png";
      imwrite(result_path.c_str(),result_mat);
      
     
       
    }
   }
	  
	return 1;


}
