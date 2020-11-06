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
	/////////////////////////////////////////////준수//////////////////////////////////////////////////////
public:
	double *d_src_cloud_point_XYZ;
	int number_of_d_src_cloud_point_XYZ;
private:
	/////////////////////////////////////////////GPU MEM//////////////////////////////////////////	
	double *d_dst_cloud_point_XYZ;					//t+1에 추적된 포인트 (크기 : 최대 점군 개수 X 3 X 시점 수(8))
	double *d_difference_depth_arr;					//3차원 포인트의 z 값 - 투영 후 깊이 맵의 깊이 (크기 : 최대 점군 개수 X 시점 수(8))
	double *d_difference_depth_sum;					//모든 시점에서 깊이 차이의 합 (크기 : 최대 점군 개수)
	double *d_weight_arr;							//각 시점 별 포인트의 가충치 (크기 : 최대 점군 개수 X 시점 수(8))
	double *d_weighted_dst_cloud_point_XYZ;			//가중치가 적용된 t+1 시간의 추적된 포인트 (크기 : 최대 점군 개수 X 3)
	int *d_frame_count_table;						//몇개의 시점에서 추적에 성공했는지 카운팅 (크기 : 최대 점군 개수)
	unsigned short *d_opticalflow_image;
	unsigned short *d_cur_frame_depth_map_data;
	unsigned short *d_next_frame_depth_map_data;
	double *d_extrinsic_param;
	double *d_extrinsic_inv_param;
	double *d_intrinsic_param;
	double *d_intrinsic_inv_param;
	int *d_index_table;
	int *d_nz_num;
	double *d_compressed_arr_src;					//추적에 성공한 src 포인트만 압축한 배열
	double *d_compressed_arr;						//추적된 dst 포인트만 압축한 배열
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
	int MAX_CLOUD_POINT_SIZE = 400000;		//최대 점군 개수
	int XY_CHANNEL_NUM = 2;					
	int XYZ_CHANNEL_NUM = 3;
	int MAX_DST_POINT_SIZE = 20000;			//최대 추적 가능 포인트 개수
	int SRC_DST_CHANNEL_NUM = 6;
	/////////////////////////////////////////////준수//////////////////////////////////////////////////////

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


	/*	함수명 : find_3D_Corr_cpp
		내용   : 옵티컬플로우 정보와 깊이 정보를 이용하여 src 점군의 다음 프레임 대응점 탐색
	*/
	KNU_SAMPLING_DLL_API void find_3D_Corr_cpp(double *src_cloud_point, double* dst_cloud_point, double* difference_depth_arr, double* difference_depth_sum,int* frame_count_table, unsigned short* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col, int cur_cam_index);
	
	/*	함수명 : get_nz_index_num_cpp
		내용   : 대응점 탐색에 실패한 포인트들을 제거하기위해 대응점 탐색에 성공한 포인트들의 인덱스를 따로 저장 
	*/
	KNU_SAMPLING_DLL_API void get_nz_index_num_cpp(double *dst_cloud_point, int *index_table, int *nz_num,int *frame_count_table);
	
	/*	함수명 : compress_arr_cpp
		내용   : get_nz_index_num_cpp로 작성한 대응점 탐색에 성공한 인덱스 테이블을 이용하여 성공한 포인트만 따로 압축하여 저장
	*/
	KNU_SAMPLING_DLL_API void compress_arr_cpp(double *compressed_arr, int *compressed_count_table, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
	KNU_SAMPLING_DLL_API void compress_arr_cpp2(double *compressed_arr_src, double *compressed_arr, int *compressed_count_table, double *src_cloud_point, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num);
	
	/*	함수명 : average_XYZ_point_cpp
	내용   : 지금은 사용하지 않지만 과거 여러 시점에서 탐색된 대응점들을 평균하여 최종 대응점을 계산하는 함수
	*/
	KNU_SAMPLING_DLL_API void average_XYZ_point_cpp(double *compressed_arr, int *compressed_count_table, int *nz_num);
	
	/*	함수명 : compute_weight_cpp
	내용   : find_3D_Corr_cpp에서 구한 각 시점별 깊이 차이와 총 깊이 차이의 합을 이용하여 각 시점별 가중치를 계산함
	*/
	KNU_SAMPLING_DLL_API void compute_weight_cpp(double *difference_depth_arr, double* difference_depth_sum, double *weight_arr, int camera_num);
	
	/*	함수명 : compute_weighted_points_cpp
	내용   : compute_weight_cpp에서 구한 각 시점별 가중치를 이용하여 가중치를 적용한 dst 포인트를 계산함
	*/
	KNU_SAMPLING_DLL_API void compute_weighted_points_cpp(double *non_weighted_points, double *weight_arr, double *weighted_points, int camera_num);
	//KNU_SAMPLING_DLL_API void find_3D_Corr_cpp(int image_row, int image_col);
	//KNU_SAMPLING_DLL_API void set_jun_mem_copy(int *h_opticalflow_image, unsigned short *h_cur_frame_depth_map_data, unsigned short *h_next_frame_depth_map_data, double *h_extrinsic_param, double *h_extrinsic_inv_param, double *h_intrinsic_param, double *h_intrinsic_inv_param);
	//void set_h_cloud_point(double *h_cloud_point_XYZ);
};

