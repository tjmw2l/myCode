#pragma once
#ifdef KNU_SAMPLING_DLL_EXPORTS
#define KNU_SAMPLING_DLL_API __declspec(dllexport)
#else
#define KNU_SAMPLING_DLL_API __declspec(dllimport)
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <opencv2\opencv.hpp>

KNU_SAMPLING_DLL_API __global__ void find_3D_Corr(double *src_cloud_point, double* dst_cloud_point, int* frame_count_table, int* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col);
KNU_SAMPLING_DLL_API __global__ void find_3D_Corr(double *src_cloud_point, double* dst_cloud_point, double* difference_depth_arr, double* difference_depth_sum, int* frame_count_table, unsigned short* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col, int cur_cam_index, int max_src_cloud_point_size);
KNU_SAMPLING_DLL_API __global__ void get_nz_index_num(double *dst_cloud_point, int *index_table, int *nz_num,int *frame_count_table);
KNU_SAMPLING_DLL_API __global__ void compress_arr(double *compressed_arr, int *compressed_count_table, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
KNU_SAMPLING_DLL_API __global__ void compress_arr2(double *compressed_arr_src, double *compressed_arr, int *compressed_count_table, double *src_cloud_point, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
KNU_SAMPLING_DLL_API __global__ void average_XYZ_point(double *compressed_arr, int *compressed_count_table,int *nz_num);
KNU_SAMPLING_DLL_API __global__ void compute_weight(double *difference_depth_arr,double* difference_depth_sum, double *weight_arr,int camera_num, int max_src_cloud_point_size);
KNU_SAMPLING_DLL_API __global__ void compute_weighted_points(double *non_weighted_points, double *weight_arr, double *weighted_points, int camera_num,int max_src_cloud_point_size);
class crvlSamplingOtp
{
	/////////////////////////////////////////////�ؼ�//////////////////////////////////////////////////////
public:
	double *d_src_cloud_point_XYZ;
	int number_of_d_src_cloud_point_XYZ;
private:
	/////////////////////////////////////////////GPU MEM//////////////////////////////////////////	
	double *d_dst_cloud_point_XYZ;					//t+1�� ������ ����Ʈ (ũ�� : �ִ� ���� ���� X 3 X ���� ��(8))
	double *d_difference_depth_arr;					//3���� ����Ʈ�� z �� - ���� �� ���� ���� ���� (ũ�� : �ִ� ���� ���� X ���� ��(8))
	double *d_difference_depth_sum;					//��� �������� ���� ������ �� (ũ�� : �ִ� ���� ����)
	double *d_weight_arr;							//�� ���� �� ����Ʈ�� ����ġ (ũ�� : �ִ� ���� ���� X ���� ��(8))
	double *d_weighted_dst_cloud_point_XYZ;			//����ġ�� ����� t+1 �ð��� ������ ����Ʈ (ũ�� : �ִ� ���� ���� X 3)
	int *d_frame_count_table;						//��� �������� ������ �����ߴ��� ī���� (ũ�� : �ִ� ���� ����)
	unsigned short *d_opticalflow_image;
	unsigned short *d_cur_frame_depth_map_data;
	unsigned short *d_next_frame_depth_map_data;
	double *d_extrinsic_param;
	double *d_extrinsic_inv_param;
	double *d_intrinsic_param;
	double *d_intrinsic_inv_param;
	int *d_index_table;
	int *d_nz_num;
	double *d_compressed_arr_src;					//������ ������ src ����Ʈ�� ������ �迭
	double *d_compressed_arr;						//������ dst ����Ʈ�� ������ �迭
	int *d_compressed_count_table;


	/////////////////////////////////////////////CPU MEM//////////////////////////////////////////
	double *h_src_cloud_point_XYZ;
	//int *h_opticalflow_image;
	int *h_index_table;
	double *h_compressed_src_src;
	double *h_compressed_src_dst;
	
	double *h_extrinsic;
	double *h_extrinsic_inv;
	double *h_intrinsic;
	double *h_intrinsic_inv;
	unsigned short *h_src_frame_depth_data;
	unsigned short *h_dst_frame_depth_data;
	////////////////////////////////////////////////////////////////////////////////////////////
	bool start = true;
public:
	int MAX_CLOUD_POINT_SIZE = 400000;		//�ִ� ���� ����
	int XY_CHANNEL_NUM = 2;					
	int XYZ_CHANNEL_NUM = 3;
	int MAX_DST_POINT_SIZE = 20000;			//�ִ� ���� ���� ����Ʈ ����
	int SRC_DST_CHANNEL_NUM = 6;
	/////////////////////////////////////////////�ؼ�//////////////////////////////////////////////////////

public:
	crvlSamplingOtp();
	~crvlSamplingOtp();
	KNU_SAMPLING_DLL_API bool is_first();
	KNU_SAMPLING_DLL_API void set_h_mem_alloc_CPU(int image_row, int image_col, int camera_num);
	KNU_SAMPLING_DLL_API void set_h_mem_init_CPU(int image_row, int image_col, int camera_num);
	KNU_SAMPLING_DLL_API void set_h_mem_copy_CPU(unsigned short* _h_src_frame_depth_data, unsigned short* _h_dst_frame_depth_data, double* _h_extrinsic, double* _h_extrinsic_inv, double* _h_intrinsic, double* _h_intrinsic_inv);
	KNU_SAMPLING_DLL_API void set_h_mem_copy_CPU(double* _h_extrinsic, double* _h_extrinsic_inv, double* _h_intrinsic, double* _h_intrinsic_inv);
	KNU_SAMPLING_DLL_API void set_h_mem_copy_src_cloud_point_CPU(double* _h_cloud_point, int size);
	KNU_SAMPLING_DLL_API void set_d_mem_alloc_GPU(int image_row, int image_col,int cam_count);
	KNU_SAMPLING_DLL_API void set_d_mem_init_GPU(int image_row, int image_col);
	KNU_SAMPLING_DLL_API void set_d_mem_init_GPU(int image_row, int image_col, unsigned short *current_depth, unsigned short *next_depth, unsigned short *GPC_result,int cam_count);
	KNU_SAMPLING_DLL_API void set_d_mem_copy_GPU(int image_row,int image_col);
	KNU_SAMPLING_DLL_API void set_d_mem_copy_GPU(int image_row, int image_col, unsigned short *current_depth, unsigned short *next_depth, unsigned short *GPC_result);
	KNU_SAMPLING_DLL_API int get_XY_CHANNEL_NUM();
	KNU_SAMPLING_DLL_API int get_XYZ_CHANNEL_NUM();
	KNU_SAMPLING_DLL_API int get_MAX_DST_POINT_SIZE();
	KNU_SAMPLING_DLL_API int get_SRC_DST_CHANNEL_NUM();
	KNU_SAMPLING_DLL_API double* get_h_src_cloud_point_ptr();
	KNU_SAMPLING_DLL_API int* get_h_opticalflow_ptr();
	KNU_SAMPLING_DLL_API double* get_h_compressed_src_src_ptr();
	KNU_SAMPLING_DLL_API double* get_h_compressed_src_dst_ptr();
	KNU_SAMPLING_DLL_API int* get_h_index_table_ptr();
	KNU_SAMPLING_DLL_API bool opticalflow_vecTOimage(std::vector<std::vector<cv::Point2i>>& inSrcCorr, std::vector<std::vector<cv::Point2i>>& inDstCorr, int image_row, int image_col, int camera_num);

	KNU_SAMPLING_DLL_API double* get_d_src_cloud_point_XYZ();
	KNU_SAMPLING_DLL_API double* get_d_dst_cloud_point_XYZ();
	KNU_SAMPLING_DLL_API double* get_d_difference_depth_arr();
	KNU_SAMPLING_DLL_API double* get_d_difference_depth_sum();
	KNU_SAMPLING_DLL_API double* get_d_weight_arr();
	KNU_SAMPLING_DLL_API double* get_d_weighted_dst_cloud_point_XYZ();
	KNU_SAMPLING_DLL_API int* get_d_frame_count_table();
	KNU_SAMPLING_DLL_API unsigned short* get_d_opticalflow_image();
	KNU_SAMPLING_DLL_API unsigned short* get_d_cur_frame_depth_map_data();
	KNU_SAMPLING_DLL_API unsigned short* get_d_next_frame_depth_map_data();
	KNU_SAMPLING_DLL_API double* get_d_extrinsic_param();
	KNU_SAMPLING_DLL_API double* get_d_extrinsic_inv_param();
	KNU_SAMPLING_DLL_API double* get_d_intrinsic_param();
	KNU_SAMPLING_DLL_API double* get_d_intrinsic_inv_param();
	KNU_SAMPLING_DLL_API int* get_d_index_table();
	KNU_SAMPLING_DLL_API int* get_d_nz_num();
	KNU_SAMPLING_DLL_API double* get_d_compressed_arr();
	KNU_SAMPLING_DLL_API double* get_d_compressed_arr_src();
	KNU_SAMPLING_DLL_API int* get_d_compressed_count_table();


	/*	�Լ��� : find_3D_Corr_cpp
		����   : ��Ƽ���÷ο� ������ ���� ������ �̿��Ͽ� src ������ ���� ������ ������ Ž��
	*/
	KNU_SAMPLING_DLL_API void find_3D_Corr_cpp(double *src_cloud_point, double* dst_cloud_point, double* difference_depth_arr, double* difference_depth_sum,int* frame_count_table, unsigned short* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col, int cur_cam_index);
	
	/*	�Լ��� : get_nz_index_num_cpp
		����   : ������ Ž���� ������ ����Ʈ���� �����ϱ����� ������ Ž���� ������ ����Ʈ���� �ε����� ���� ���� 
	*/
	KNU_SAMPLING_DLL_API void get_nz_index_num_cpp(double *dst_cloud_point, int *index_table, int *nz_num,int *frame_count_table);
	
	/*	�Լ��� : compress_arr_cpp
		����   : get_nz_index_num_cpp�� �ۼ��� ������ Ž���� ������ �ε��� ���̺��� �̿��Ͽ� ������ ����Ʈ�� ���� �����Ͽ� ����
	*/
	KNU_SAMPLING_DLL_API void compress_arr_cpp(double *compressed_arr, int *compressed_count_table, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
	KNU_SAMPLING_DLL_API void compress_arr_cpp2(double *compressed_arr_src, double *compressed_arr, int *compressed_count_table, double *src_cloud_point, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
	
	/*	�Լ��� : average_XYZ_point_cpp
	����   : ������ ������� ������ ���� ���� �������� Ž���� ���������� ����Ͽ� ���� �������� ����ϴ� �Լ�
	*/
	KNU_SAMPLING_DLL_API void average_XYZ_point_cpp(double *compressed_arr, int *compressed_count_table, int *nz_num);
	
	/*	�Լ��� : compute_weight_cpp
	����   : find_3D_Corr_cpp���� ���� �� ������ ���� ���̿� �� ���� ������ ���� �̿��Ͽ� �� ������ ����ġ�� �����
	*/
	KNU_SAMPLING_DLL_API void compute_weight_cpp(double *difference_depth_arr, double* difference_depth_sum, double *weight_arr, int camera_num);
	
	/*	�Լ��� : compute_weighted_points_cpp
	����   : compute_weight_cpp���� ���� �� ������ ����ġ�� �̿��Ͽ� ����ġ�� ������ dst ����Ʈ�� �����
	*/
	KNU_SAMPLING_DLL_API void compute_weighted_points_cpp(double *non_weighted_points, double *weight_arr, double *weighted_points, int camera_num);
	//KNU_SAMPLING_DLL_API void find_3D_Corr_cpp(int image_row, int image_col);
	//KNU_SAMPLING_DLL_API void set_jun_mem_copy(int *h_opticalflow_image, unsigned short *h_cur_frame_depth_map_data, unsigned short *h_next_frame_depth_map_data, double *h_extrinsic_param, double *h_extrinsic_inv_param, double *h_intrinsic_param, double *h_intrinsic_inv_param);
	//void set_h_cloud_point(double *h_cloud_point_XYZ);
};

