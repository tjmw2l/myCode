#include
#include <iostream>
#include <string.h>

#define NOMINMAX	
#include <Windows.h>
class Timer {
public:
	inline void tic() {
		_tic = getTime();
	}
	inline long toc() {
		_toc = getTime();
		return _toc - _tic;
	}
private:
	inline long getTime() {
		SYSTEMTIME t;
		GetSystemTime(&t);
		return ((t.wHour * 60 + t.wMinute) * 60 + t.wSecond) * 1000 + t.wMilliseconds;
	}
private:
	long _tic, _toc;
};

crvlSamplingOtp::crvlSamplingOtp()
{

}
crvlSamplingOtp::~crvlSamplingOtp()
{
	cudaFree(this->d_src_cloud_point_XYZ);
	cudaFree(this->d_dst_cloud_point_XYZ);
	cudaFree(this->d_difference_depth_arr);
	cudaFree(this->d_difference_depth_sum);
	cudaFree(this->d_weight_arr);
	cudaFree(this->d_frame_count_table);
	cudaFree(this->d_opticalflow_image);
	cudaFree(this->d_cur_frame_depth_map_data);
	cudaFree(this->d_next_frame_depth_map_data);
	cudaFree(this->d_extrinsic_param);
	cudaFree(this->d_extrinsic_inv_param);
	cudaFree(this->d_intrinsic_param);
	cudaFree(this->d_intrinsic_inv_param);
	cudaFree(this->d_compressed_count_table);
	cudaFree(this->d_weighted_dst_cloud_point_XYZ);
	delete[] this->h_src_cloud_point_XYZ;
	//delete[] this->h_opticalflow_image;
	delete[] this->h_compressed_src_dst;
}

bool crvlSamplingOtp::is_first()
{
	if (this->start == true)
	{
		this->start = false;
		return true;
	}

	return false;
}


void crvlSamplingOtp::set_d_mem_alloc_GPU(int image_row,int image_col,int cam_count)
{
	cudaMalloc((double**)&(this->d_src_cloud_point_XYZ),this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE*sizeof(double));
	cudaMalloc((double**)&(this->d_dst_cloud_point_XYZ), this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE *cam_count* sizeof(double));
	cudaMalloc((double**)&(this->d_difference_depth_arr),this->MAX_CLOUD_POINT_SIZE *cam_count * sizeof(double));
	cudaMalloc((double**)&(this->d_difference_depth_sum), this->MAX_CLOUD_POINT_SIZE * sizeof(double));
	cudaMalloc((double**)&(this->d_weight_arr), this->MAX_CLOUD_POINT_SIZE *cam_count * sizeof(double));
	cudaMalloc((double**)&(this->d_weighted_dst_cloud_point_XYZ), this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE * sizeof(double));
	cudaMalloc((double**)&(this->d_frame_count_table), this->MAX_CLOUD_POINT_SIZE * sizeof(int));
	cudaMalloc((int**)&(this->d_opticalflow_image), image_row*image_col*this->XY_CHANNEL_NUM * sizeof(int));
	cudaMalloc((unsigned short**)&(this->d_cur_frame_depth_map_data), image_row*image_col*sizeof(unsigned short));
	cudaMalloc((unsigned short**)&(this->d_next_frame_depth_map_data), image_row*image_col * sizeof(unsigned short));
	cudaMalloc((double**)&(this->d_extrinsic_param), 16 * sizeof(double));
	cudaMalloc((double**)&(this->d_extrinsic_inv_param), 16 * sizeof(double));
	cudaMalloc((double**)&(this->d_intrinsic_param), 9 * sizeof(double));
	cudaMalloc((double**)&(this->d_intrinsic_inv_param), 9 * sizeof(double));
	cudaMalloc((int**)&(this->d_index_table), this->MAX_DST_POINT_SIZE * sizeof(int));
	cudaMalloc((int**)&(this->d_nz_num), sizeof(int));
	
	cudaMalloc((double**)&(this->d_compressed_arr_src), this->XYZ_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMalloc((double**)&(this->d_compressed_arr), this->XYZ_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMalloc((int**)&(this->d_compressed_count_table), this->MAX_DST_POINT_SIZE * sizeof(int));
}

void crvlSamplingOtp::set_d_mem_init_GPU(int image_row, int image_col)
{
	cudaMemcpy(this->d_src_cloud_point_XYZ, this->h_src_cloud_point_XYZ,this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(this->d_dst_cloud_point_XYZ, 0, this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_frame_count_table, 0,this->MAX_CLOUD_POINT_SIZE * sizeof(int));
	cudaMemset(this->d_index_table, 0, this->MAX_DST_POINT_SIZE * sizeof(int));
	cudaMemset(this->d_nz_num, 0, sizeof(int));
	//cudaMemset(this->d_compressed_arr, 0, this->SRC_DST_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_compressed_arr, 0, this->XYZ_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_compressed_count_table, 0, this->MAX_DST_POINT_SIZE * sizeof(int));
}

void crvlSamplingOtp::set_d_mem_init_GPU(int image_row, int image_col, unsigned short *current_depth, unsigned short *next_depth, unsigned short *GPC_result,int cam_count)
{
	//cudaMemcpy(this->d_src_cloud_point_XYZ, this->h_src_cloud_point_XYZ, this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(this->d_dst_cloud_point_XYZ, 0, this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE *cam_count* sizeof(double));
	cudaMemset(this->d_difference_depth_arr, 0, this->MAX_CLOUD_POINT_SIZE *cam_count * sizeof(double));
	cudaMemset(this->d_difference_depth_sum, 0, this->MAX_CLOUD_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_weight_arr, 0, this->MAX_CLOUD_POINT_SIZE *cam_count * sizeof(double));
	cudaMemset(this->d_weighted_dst_cloud_point_XYZ, 0, this->XYZ_CHANNEL_NUM*this->MAX_CLOUD_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_frame_count_table, 0, this->MAX_CLOUD_POINT_SIZE * sizeof(int));
	cudaMemset(this->d_index_table, 0, this->MAX_DST_POINT_SIZE * sizeof(int));
	cudaMemset(this->d_nz_num, 0, sizeof(int));
	//cudaMemset(this->d_compressed_arr, 0, this->SRC_DST_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_compressed_arr, 0, this->XYZ_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_compressed_arr_src, 0, this->XYZ_CHANNEL_NUM * this->MAX_DST_POINT_SIZE * sizeof(double));
	cudaMemset(this->d_compressed_count_table, 0, this->MAX_DST_POINT_SIZE * sizeof(int));

	/*this->d_cur_frame_depth_map_data = current_depth;
	this->d_next_frame_depth_map_data = next_depth;
	this->d_opticalflow_image = GPC_result;*/
}

void crvlSamplingOtp::set_d_mem_copy_GPU(int image_row, int image_col, unsigned short *current_depth, unsigned short *next_depth, unsigned short *GPC_result)
{
	//cudaMemcpy(this->d_opticalflow_image, this->h_opticalflow_image, image_row*image_col*this->XY_CHANNEL_NUM * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(this->d_cur_frame_depth_map_data, this->h_src_frame_depth_data, image_col*image_row * sizeof(unsigned short), cudaMemcpyHostToDevice);
	//cudaMemcpy(this->d_next_frame_depth_map_data, this->h_dst_frame_depth_data, image_row*image_col * sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_extrinsic_param, this->h_extrinsic, 16 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_extrinsic_inv_param, this->h_extrinsic_inv, 16 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_intrinsic_param, this->h_intrinsic, 9 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_intrinsic_inv_param, this->h_intrinsic_inv, 9 * sizeof(double), cudaMemcpyHostToDevice);
}

void crvlSamplingOtp::set_h_mem_alloc_CPU(int image_row,int image_col,int camera_num)
{
	this->h_src_cloud_point_XYZ = new double[this->MAX_CLOUD_POINT_SIZE*this->XYZ_CHANNEL_NUM];
	
	this->h_compressed_src_src = new double[this->MAX_DST_POINT_SIZE*this->XYZ_CHANNEL_NUM];
	this->h_compressed_src_dst = new double[this->MAX_DST_POINT_SIZE*this->XYZ_CHANNEL_NUM];
	this->h_index_table = new int[this->MAX_DST_POINT_SIZE];
}

void crvlSamplingOtp::set_h_mem_init_CPU(int image_row, int image_col, int camera_num)
{
	memset(this->h_src_cloud_point_XYZ, 0, this->MAX_CLOUD_POINT_SIZE*this->XYZ_CHANNEL_NUM * sizeof(double));
	//memset(this->h_opticalflow_image, 0, image_row*image_col*this->XY_CHANNEL_NUM*sizeof(int)*camera_num);
	memset(this->h_compressed_src_src, 0, this->MAX_DST_POINT_SIZE*this->XYZ_CHANNEL_NUM * sizeof(double));
	memset(this->h_compressed_src_dst, 0, this->MAX_DST_POINT_SIZE*this->XYZ_CHANNEL_NUM * sizeof(double));
	memset(this->h_index_table, 0, this->MAX_DST_POINT_SIZE * sizeof(int));
}

void crvlSamplingOtp::set_h_mem_copy_CPU(unsigned short* _h_src_frame_depth_data, unsigned short* _h_dst_frame_depth_data,double* _h_extrinsic,double* _h_extrinsic_inv,double* _h_intrinsic, double* _h_intrinsic_inv)
{
	this->h_src_frame_depth_data = _h_src_frame_depth_data;
	this->h_dst_frame_depth_data = _h_dst_frame_depth_data;
	this->h_extrinsic = _h_extrinsic;
	this->h_extrinsic_inv = _h_extrinsic_inv;
	this->h_intrinsic = _h_intrinsic;
	this->h_intrinsic_inv = _h_intrinsic_inv;
}

void crvlSamplingOtp::set_h_mem_copy_CPU(double* _h_extrinsic, double* _h_extrinsic_inv, double* _h_intrinsic, double* _h_intrinsic_inv)
{
	//this->h_src_frame_depth_data = _h_src_frame_depth_data;
	//this->h_dst_frame_depth_data = _h_dst_frame_depth_data;
	this->h_extrinsic = _h_extrinsic;
	this->h_extrinsic_inv = _h_extrinsic_inv;
	this->h_intrinsic = _h_intrinsic;
	this->h_intrinsic_inv = _h_intrinsic_inv;
}

void crvlSamplingOtp::set_h_mem_copy_src_cloud_point_CPU(double* _h_cloud_point, int size)
{
	memcpy(this->h_src_cloud_point_XYZ, _h_cloud_point, size);
}
int crvlSamplingOtp::get_XY_CHANNEL_NUM()
{
	return this->XY_CHANNEL_NUM;
}
int crvlSamplingOtp::get_XYZ_CHANNEL_NUM()
{
	return this->XYZ_CHANNEL_NUM;
}
int crvlSamplingOtp::get_MAX_DST_POINT_SIZE()
{
	return this->MAX_DST_POINT_SIZE;
}
int crvlSamplingOtp::get_SRC_DST_CHANNEL_NUM()
{
	return this->SRC_DST_CHANNEL_NUM;
}

double* crvlSamplingOtp::get_h_src_cloud_point_ptr()
{
	return this->h_src_cloud_point_XYZ;
}

//int* crvlSamplingOtp::get_h_opticalflow_ptr()
//{
//	return this->h_opticalflow_image;
//}

double* crvlSamplingOtp::get_h_compressed_src_src_ptr()
{
	return this->h_compressed_src_src;
}

double* crvlSamplingOtp::get_h_compressed_src_dst_ptr()
{
	return this->h_compressed_src_dst;
}
int* crvlSamplingOtp::get_h_index_table_ptr()
{
	return this->h_index_table;
}
double* crvlSamplingOtp::get_d_src_cloud_point_XYZ()
{
	return this->d_src_cloud_point_XYZ;
}
double* crvlSamplingOtp::get_d_dst_cloud_point_XYZ()
{
	return this->d_dst_cloud_point_XYZ;
}
double* crvlSamplingOtp::get_d_difference_depth_arr()
{
	return this->d_difference_depth_arr;
}
double* crvlSamplingOtp::get_d_difference_depth_sum()
{
	return this->d_difference_depth_sum;
}
double* crvlSamplingOtp::get_d_weight_arr()
{
	return this->d_weight_arr;
}
double* crvlSamplingOtp::get_d_weighted_dst_cloud_point_XYZ()
{
	return this->d_weighted_dst_cloud_point_XYZ;
}
int* crvlSamplingOtp::get_d_frame_count_table()
{
	return this->d_frame_count_table;
}
unsigned short* crvlSamplingOtp::get_d_opticalflow_image()
{
	return this->d_opticalflow_image;
}
unsigned short* crvlSamplingOtp::get_d_cur_frame_depth_map_data()
{
	return this->d_cur_frame_depth_map_data;
}
unsigned short* crvlSamplingOtp::get_d_next_frame_depth_map_data()
{
	return this->d_next_frame_depth_map_data;
}
double* crvlSamplingOtp::get_d_extrinsic_param()
{
	return this->d_extrinsic_param;
}
double* crvlSamplingOtp::get_d_extrinsic_inv_param()
{
	return this->d_extrinsic_inv_param;
}
double* crvlSamplingOtp::get_d_intrinsic_param()
{
	return this->d_intrinsic_param;
}
double* crvlSamplingOtp::get_d_intrinsic_inv_param()
{
	return this->d_intrinsic_inv_param;
}
int* crvlSamplingOtp::get_d_index_table()
{
	return this->d_index_table;
}
int* crvlSamplingOtp::get_d_nz_num()
{
	return this->d_nz_num;
}
double* crvlSamplingOtp::get_d_compressed_arr()
{
	return this->d_compressed_arr;
}

double* crvlSamplingOtp::get_d_compressed_arr_src()
{
	return this->d_compressed_arr_src;
}

int* crvlSamplingOtp::get_d_compressed_count_table()
{
	return this->d_compressed_count_table;
}

bool crvlSamplingOtp::opticalflow_vecTOimage(std::vector<std::vector<cv::Point2i>>& inSrcCorr, std::vector<std::vector<cv::Point2i>>& inDstCorr, int image_row, int image_col, int camera_num)
{
	/*memset(this->h_opticalflow_image, 0, image_col*image_row * this->XY_CHANNEL_NUM*sizeof(int));

	int src_x = 0;
	int src_y = 0;
	int dst_x = 0;
	int dst_y = 0;
	int shifted_index = 0;

	for (int j = 0; j < inSrcCorr.at(camera_num).size(); j++)
	{
		src_x = inSrcCorr.at(camera_num).at(j).x;
		src_y = inSrcCorr.at(camera_num).at(j).y;
		dst_x = inDstCorr.at(camera_num).at(j).x;
		dst_y = inDstCorr.at(camera_num).at(j).y;

		shifted_index = (src_y*image_col + src_x) * 2;
		this->h_opticalflow_image[shifted_index] = dst_x;
		this->h_opticalflow_image[shifted_index + 1] = dst_y;
	}
	return true;*/
	return false;
}


void crvlSamplingOtp::find_3D_Corr_cpp(double *src_cloud_point, double* dst_cloud_point, double* difference_depth_arr, double* difference_depth_sum, int* frame_count_table, unsigned short* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col,int cur_cam_index)
{
	int blockDim = 32 * 25;	
	int gridDim = (this->MAX_CLOUD_POINT_SIZE + blockDim - 1) / blockDim;	
	//find_3D_Corr << <gridDim, blockDim >> > (src_cloud_point, dst_cloud_point, frame_count_table, opticalflow_image, cur_frame_depth, next_frame_depth, extrinsic, inv_extrinsic, intrinsic, inv_intrinsic, image_row, image_col,camera_num);
	find_3D_Corr << <gridDim, blockDim >> > (src_cloud_point, dst_cloud_point, difference_depth_arr,difference_depth_sum,frame_count_table, opticalflow_image, cur_frame_depth, next_frame_depth, extrinsic, inv_extrinsic, intrinsic, inv_intrinsic, image_row, image_col, cur_cam_index, this->MAX_CLOUD_POINT_SIZE);
}

void crvlSamplingOtp::get_nz_index_num_cpp(double *dst_cloud_point,int *index_table, int *nz_num,int *frame_count_table)
{
	//??
	int blockDim = 32 * 25;
	int gridDim = (this->MAX_CLOUD_POINT_SIZE + blockDim - 1) / blockDim;	
	get_nz_index_num<<<gridDim,blockDim>>>(dst_cloud_point, index_table, nz_num, frame_count_table);
}

void crvlSamplingOtp::compress_arr_cpp(double *compressed_arr,int *compressed_count_table, double *dst_cloud_point,int* frame_count_table, int *index_table, int *nz_num)
{
	int blockDim = 32 * 15;
	int gridDim = (this->MAX_DST_POINT_SIZE + blockDim - 1) / blockDim;

	//compress_arr<<<gridDim,blockDim>>>(compressed_arr, src_cloud_point, dst_cloud_point, index_table, nz_num);
	compress_arr << <gridDim, blockDim >> >(compressed_arr,compressed_count_table, dst_cloud_point,frame_count_table, index_table, nz_num);
}

void crvlSamplingOtp::compress_arr_cpp2(double *compressed_arr_src, double *compressed_arr, int *compressed_count_table, double *src_cloud_point, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num)
{
	int blockDim = 32 * 15;
	int gridDim = (this->MAX_DST_POINT_SIZE + blockDim - 1) / blockDim;

	//compress_arr<<<gridDim,blockDim>>>(compressed_arr, src_cloud_point, dst_cloud_point, index_table, nz_num);
	//compress_arr << <gridDim, blockDim >> >(compressed_arr, compressed_count_table, dst_cloud_point, frame_count_table, index_table, nz_num);
	compress_arr2 << <gridDim, blockDim >> >(compressed_arr_src, compressed_arr, compressed_count_table, src_cloud_point, dst_cloud_point, frame_count_table, index_table, nz_num);
}

void crvlSamplingOtp::average_XYZ_point_cpp(double *compressed_arr, int *compressed_count_table, int *nz_num)
{
	int blockDim = 32*2;
	int gridDim = (this->MAX_DST_POINT_SIZE + blockDim - 1) / blockDim;
	average_XYZ_point << <gridDim, blockDim >> > (compressed_arr, compressed_count_table, nz_num);
}

void crvlSamplingOtp::compute_weight_cpp(double *difference_depth_arr, double *difference_depth_sum, double *weight_arr,int camera_num)
{
	int blockDim = 32 * 25;
	int gridDim = (this->MAX_CLOUD_POINT_SIZE + blockDim - 1) / blockDim;
	compute_weight<<<gridDim,blockDim>>>(difference_depth_arr, difference_depth_sum,weight_arr, camera_num, this->MAX_CLOUD_POINT_SIZE);
}

void crvlSamplingOtp::compute_weighted_points_cpp(double *non_weighted_points, double *weight_arr, double *weighted_points, int camera_num)
{
	int blockDim = 32 * 25;
	int gridDim = (this->MAX_CLOUD_POINT_SIZE + blockDim - 1) / blockDim;
	compute_weighted_points << <gridDim, blockDim >> > (non_weighted_points, weight_arr, weighted_points, camera_num, this->MAX_CLOUD_POINT_SIZE);
}

__global__ void find_3D_Corr(double *src_cloud_point, double* dst_cloud_point, double* difference_depth_arr, double* difference_depth_sum,int* frame_count_table, unsigned short* opticalflow_image, unsigned short* cur_frame_depth, unsigned short* next_frame_depth, double* extrinsic, double* inv_extrinsic, double* intrinsic, double* inv_intrinsic, int image_row, int image_col, int cur_cam_index,int max_src_cloud_point_size)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	const double diff_margin = 3;      // 3D point�� z���� depth������ distance ���� // (Volume size) / (Voxel dimension x) // 3072 / 256 = 12 �ε� 12�ϸ� �װ� 8 �����ؾߵ�
	const double diff_Euclidian = 100; // 3D optical flow ���̿� ���� �Ÿ�	
	const double cur_W_point_x = src_cloud_point[3 * index];
	const double cur_W_point_y = src_cloud_point[3 * index + 1];
	const double cur_W_point_z = src_cloud_point[3 * index + 2];
	if (index % 3 == 0)
	{
		if ((cur_W_point_x != 0) || (cur_W_point_y != 0) || (cur_W_point_z != 0))
		{
			double cur_C_point_x = inv_extrinsic[0] * cur_W_point_x + inv_extrinsic[1] * cur_W_point_y + inv_extrinsic[2] * cur_W_point_z + inv_extrinsic[3]; // CurrentFame�� 3D-point�� World->Cam ��ǥ���(X)
			double cur_C_point_y = inv_extrinsic[4] * cur_W_point_x + inv_extrinsic[5] * cur_W_point_y + inv_extrinsic[6] * cur_W_point_z + inv_extrinsic[7]; // CurrentFame�� 3D-point�� World->Cam ��ǥ���(Y)
			double cur_C_point_z = inv_extrinsic[8] * cur_W_point_x + inv_extrinsic[9] * cur_W_point_y + inv_extrinsic[10] * cur_W_point_z + inv_extrinsic[11]; // CurrentFame�� 3D-point�� World->Cam ��ǥ���(Z)

			double cur_2D_point_x = (intrinsic[0] * cur_C_point_x + intrinsic[1] * cur_C_point_y + intrinsic[2] * cur_C_point_z) / cur_C_point_z;
			double cur_2D_point_y = (intrinsic[3] * cur_C_point_x + intrinsic[4] * cur_C_point_y + intrinsic[5] * cur_C_point_z) / cur_C_point_z;

			if ((0 < cur_2D_point_y && cur_2D_point_y < image_row)     // �̹��� ũ�� �ȿ� ������ ��
				&& (0 < cur_2D_point_x && cur_2D_point_x < image_col)) // �̹��� ũ�� �ȿ� ������ ��
			{
				const int cur_pixel_location = image_row*image_col*cur_cam_index + image_col*(int)cur_2D_point_y + (int)cur_2D_point_x;
				double cur_pixel_depth = (double)cur_frame_depth[cur_pixel_location];
				double distance_difference = std::abs(cur_C_point_z - cur_pixel_depth);	//src ����Ʈ�� z �� - ������ �ȼ��� ���� �� (����ġ ��꿡 ����)
								
				if (distance_difference < diff_margin) // 3���� pointcloud�� point�� z���� �̹������� depth ���̰� "diff_magin" ���� �϶�
				{
					unsigned short next_2D_point_x = opticalflow_image[2 * cur_pixel_location];
					unsigned short next_2D_point_y = opticalflow_image[2 * cur_pixel_location + 1];
					{
						const int next_pixel_location = image_row*image_col*cur_cam_index + image_col*(int)next_2D_point_y + (int)next_2D_point_x; // opticalflow�� �����Ǵ� �ȼ��� �ε���
						double next_pixel_depth = (double)next_frame_depth[next_pixel_location]; // opticalfow�� �����Ǵ� �ȼ��� depth ����

						if (next_pixel_depth > 100) // opticalfow�� �����Ǵ� �ȼ��� depth�� 0~100 ������ �����ʹ� ����(������ ���Ÿ� ���ؼ�), 100 �̻��� �����͸� ���
						{
							double next_C_point_x = (inv_intrinsic[0] * next_2D_point_x + inv_intrinsic[1] * next_2D_point_y + inv_intrinsic[2])*next_pixel_depth; // opticalflow�� �����Ǵ� �ȼ��� 3��������(X)
							double next_C_point_y = (inv_intrinsic[3] * next_2D_point_x + inv_intrinsic[4] * next_2D_point_y + inv_intrinsic[5])*next_pixel_depth; // opticalflow�� �����Ǵ� �ȼ��� 3��������(Y)
							double next_C_point_z = (inv_intrinsic[6] * next_2D_point_x + inv_intrinsic[7] * next_2D_point_y + inv_intrinsic[8])*next_pixel_depth; // opticalflow�� �����Ǵ� �ȼ��� 3��������(Z)

							if (next_C_point_z > 100 && cur_pixel_depth > 100) // ���� �������� 3���� point�� �����Ǵ� 3���� point�� Z���� ��� 100�̻� �϶�
							{
								double next_distance_difference = next_C_point_z - next_pixel_depth; // ���� deph�� �����Ǵ� depth�� ����

								if (std::abs(next_distance_difference) < diff_margin) // ���� deph�� �����Ǵ� depth�� ���̰� "diff_magin" ���� ������
								{
									double next_W_point_x = extrinsic[0] * next_C_point_x + extrinsic[1] * next_C_point_y + extrinsic[2] * next_C_point_z + extrinsic[3]; // NextFrame�� 3D-point�� World��ǥ���(X)
									double next_W_point_y = extrinsic[4] * next_C_point_x + extrinsic[5] * next_C_point_y + extrinsic[6] * next_C_point_z + extrinsic[7]; // NextFrame�� 3D-point�� World��ǥ���(Y)
									double next_W_point_z = extrinsic[8] * next_C_point_x + extrinsic[9] * next_C_point_y + extrinsic[10] * next_C_point_z + extrinsic[11]; // NextFrame�� 3D-point�� World��ǥ���(Z)

									double GPC_diff_x = cur_W_point_x - next_W_point_x; // Current-3D point�� Next-3D point�� ����(X)
									double GPC_diff_y = cur_W_point_y - next_W_point_y; // Current-3D point�� Next-3D point�� ����(Y)
									double GPC_diff_z = cur_W_point_z - next_W_point_z; // Current-3D point�� Next-3D point�� ����(Z)
									double GPC_diff = sqrt(GPC_diff_x*GPC_diff_x + GPC_diff_y*GPC_diff_y + GPC_diff_z*GPC_diff_z);

									if (GPC_diff < diff_Euclidian) // 3D optical flow ���̿� ���� �Ÿ�
									{
										dst_cloud_point[3*max_src_cloud_point_size*cur_cam_index + 3 * index] = next_W_point_x;			//cur_cam_index ��° �������� ������ ����Ʈ ���� X
										dst_cloud_point[3* max_src_cloud_point_size*cur_cam_index + 3 * index+1] = next_W_point_y;		//cur_cam_index ��° �������� ������ ����Ʈ ���� Y
										dst_cloud_point[3* max_src_cloud_point_size*cur_cam_index + 3 * index+2] = next_W_point_z;		//cur_cam_index ��° �������� ������ ����Ʈ ���� Z

										if (distance_difference == 0)
											distance_difference = 0.000001;

										difference_depth_arr[max_src_cloud_point_size*cur_cam_index + index] = distance_difference;		//cur_cam_index ��° �������� ���� ���� ��
										difference_depth_sum[index] += 1/distance_difference;											//��� �������� index ��° ����Ʈ�� ���� ���� ���� ���� �� 

										frame_count_table[index] += 1;		//index ��° ����Ʈ�� ������ ������ Ƚ��
									}
								}
							}
						}
					}

				}
			}
		}
	}

}

__global__ void get_nz_index_num(double *dst_cloud_point, int *index_table, int *nz_num,int *frame_count_table)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;		//���� ����Ʈ �ε���
	//atomicAdd(&nz_num[0], 1);
	if (dst_cloud_point[3 * index]&&frame_count_table[index]>=1)	//�ּ� 1���̶� ������ ����������
	{
		int _index = atomicAdd(&nz_num[0], 1);						//���� ����Ʈ�� ���� �ε��� ����
		index_table[_index - 1] = index;							//������ ���� ���ؽ� ���̺� ����
	}
}

__global__ void compress_arr(double *compressed_arr,int *compressed_count_table,double *dst_cloud_point,int* frame_count_table,int *index_table, int *nz_num)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;		//�ε��� ���̺��� ���� �ε���
	if (*nz_num > index)									//������ ������ ����Ʈ�� �������� ���� �ε��� ���̺��� �ε����� ������
	{
		int nz_index = index_table[index];					//������ ������ ����Ʈ�� �ε��� ����
		compressed_arr[3 * index] = dst_cloud_point[3 * nz_index];	//������ ������ ����Ʈ�� ���� �迭�� �������� ����
		compressed_arr[3 * index + 1] = dst_cloud_point[3 * nz_index + 1];
		compressed_arr[3 * index + 2] = dst_cloud_point[3 * nz_index + 2];
		compressed_count_table[index] = frame_count_table[nz_index];
	}
}

__global__ void compress_arr2(double *compressed_arr_src, double *compressed_arr, int *compressed_count_table, double *src_cloud_point, double *dst_cloud_point, int* frame_count_table, int *index_table, int *nz_num)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;		//�ε��� ���̺��� ���� �ε���
	if (*nz_num > index)									//������ ������ ����Ʈ�� �������� ���� �ε��� ���̺��� �ε����� ������
	{
		int nz_index = index_table[index];					//������ ������ ����Ʈ�� �ε��� ����

		compressed_arr_src[3 * index] = src_cloud_point[3 * nz_index];				//������ ������ src ����Ʈ�� ���� �迭�� �������� ����
		compressed_arr_src[3 * index + 1] = src_cloud_point[3 * nz_index + 1];
		compressed_arr_src[3 * index + 2] = src_cloud_point[3 * nz_index + 2];

		compressed_arr[3 * index] = dst_cloud_point[3 * nz_index];					//������ dst ����Ʈ�� ���� �迭�� ����
		compressed_arr[3 * index + 1] = dst_cloud_point[3 * nz_index + 1];
		compressed_arr[3 * index + 2] = dst_cloud_point[3 * nz_index + 2];
		compressed_count_table[index] = frame_count_table[nz_index];				//�ش� ����Ʈ�� ��� ������ �����ߴ����� ���� ����
	}
}

__global__ void average_XYZ_point(double *compressed_arr, int *compressed_count_table,int *nz_num)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (*nz_num > index)
	{
		int frame_count = compressed_count_table[index];
		
		compressed_arr[3 * index] /= frame_count;
		compressed_arr[3 * index + 1] /= frame_count;
		compressed_arr[3 * index + 2] /= frame_count;	
	}
}

__global__ void compute_weight(double *difference_depth_arr, double *difference_depth_sum,double *weight_arr, int camera_num, int max_src_cloud_point_size)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;		//���� ����Ʈ �ε���
	/*
	double sum_diff_depth_r = 0;

	for (int i = 0; i < camera_num; i++)
	{
		double cur_diff_depth=difference_depth_arr[max_src_cloud_point_size*i + index];
		if (cur_diff_depth > 0)
		{
			double cur_diff_depth_r = 1 / cur_diff_depth;
			sum_diff_depth_r += cur_diff_depth_r;
		}
	}
	*/

	double sum_diff_depth = difference_depth_sum[index];			//index��° ����Ʈ�� ���� ���� ������ �� 
	for (int i = 0; i < camera_num; i++)							//0�� �������� 8�� �������� �ݺ�
	{
		double cur_diff_depth = difference_depth_arr[max_src_cloud_point_size*i + index];	//i��° ���������� ���� ���� ������ ��
		if (cur_diff_depth > 0)
		{
			double cur_diff_depth_r = 1 / cur_diff_depth;									//���� ������ ���� ������ ���
			weight_arr[max_src_cloud_point_size*i + index] = cur_diff_depth_r / sum_diff_depth;	//����ġ ���
		}
	}
}

__global__ void compute_weighted_points(double *non_weighted_points, double *weight_arr, double *weighted_points, int camera_num, int max_src_cloud_point_size)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;	//���� ����Ʈ �ε���

	double weighted_point_x = 0;
	double weighted_point_y = 0;
	double weighted_point_z = 0;

	for (int i = 0; i < camera_num; i++)
	{
		double non_weighted_point_x = non_weighted_points[3 * max_src_cloud_point_size*i + 3 * index];			//i��° ������ dst ����Ʈ
		double non_weighted_point_y = non_weighted_points[3 * max_src_cloud_point_size*i + 3 * index + 1];
		double non_weighted_point_z = non_weighted_points[3 * max_src_cloud_point_size*i + 3 * index + 2];
		double weight = weight_arr[max_src_cloud_point_size*i + index];											//i �������� index��° ����Ʈ�� ����ġ
		if (weight>0)
		{
			weighted_point_x += weight*non_weighted_point_x;				//i ������ dst����Ʈ�� ����ġ�� ���� ���� dst����Ʈ ���
			weighted_point_y += weight*non_weighted_point_y;
			weighted_point_z += weight*non_weighted_point_z;
		}
	}

	weighted_points[3 * index] = weighted_point_x;
	weighted_points[3 * index+1] = weighted_point_y;
	weighted_points[3 * index+2] = weighted_point_z;
}