#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
//#include <cutil.h>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
#define REDUCE_BLOCK_SIZE 128

struct Matrix {
	Matrix() : elements(NULL), width(0), height(0), pitch(0) {}
	~Matrix() { if (elements) delete[] elements; }
	unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
};

void matrixMulKernel(float*, float*, float*, int, int, int, int,
                     const sycl::nd_item<3> &item_ct1);
void sigmoidKernel(float*, int, const sycl::nd_item<3> &item_ct1);
void matrixAbsErrorKernel(float*, float*, float*, int, int,
                          const sycl::nd_item<3> &item_ct1);
void absErrorKernel(float*, float*, float*, int,
                    const sycl::nd_item<3> &item_ct1);
void updateParamsAbsErrorKernel(float*, float*, float*, float*, int, float,
                                const sycl::nd_item<3> &item_ct1);
void crossEntropyKernel(float*, float*, float*, int,
                        const sycl::nd_item<3> &item_ct1);
void reduceKernel(float*, float*, int, const sycl::nd_item<3> &item_ct1,
                  float *partialSum);

inline static void InitializeMatrix(Matrix *mat, int x, int y, float val) {
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Initialize Matrix");
	}
	mat->elements[y * mat->width + x] = val;
}

inline static float Matrix_Element_Required(Matrix *mat, int x, int y)
{
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Matrix Element Required");
	}
	return mat->elements[y * mat->width + x];
}

static void AllocateMatrix(Matrix *mat, int height, int width)
{
	mat->elements = new float[height * width];
	mat->width = width;
	mat->height = height;
	for (int i = 0; i < mat->width; i++) {
		for (int j = 0; j < mat->height; j++) {
			InitializeMatrix(mat, i, j, 0.0f);
		}
	}
}

static void DisplayMatrix(Matrix &mat, bool force = false)
{
	std::cout << "Dim: " << mat.height << ", " << mat.width << "\n";
	if ((mat.width < 10 && mat.height < 10) || force)
	{
		for (int j = 0; j < mat.height; j++) {
			for (int i = 0; i < mat.width; i++) {
				std::cout << Matrix_Element_Required(&mat, i, j) << "\t";
			}
			std::cout << "\n";
		}
	}
	std::cout << std::endl;
}

static bool setup_data (string file_name, Matrix *X, Matrix *y) {

	ifstream s(file_name.c_str());
	//ifstream s(file_name);
	if (!s.is_open()) {
		//throw runtime_error(file_name + " doesn't exist");
		printf("The file does not exist\n");
	}

	int rows = 0;
	int cols = 0;
	string line;
	while (getline(s, line)) {
		// if we read first line, check how many columns
		if (rows++ == 0) {
			stringstream ss(line);

			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				cols++;
			}
		}
	}
	std::cout << "Found " << rows << " rows with " << cols << " columns." << std::endl;
	s.clear() ;
	s.seekg(0, ios::beg);

	AllocateMatrix (X, rows - 1,cols - 2);
	AllocateMatrix (y, rows - 1, 1);

	// go to second line
	getline(s, line);
	int ya = 0;
	while (getline(s, line)) {
		stringstream ss(line);

		int xa = 0;
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			// first column is uninteresting
			// second column is target values
			if (xa == 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(y, 0, ya, val);
			} else if (xa > 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(X, (xa - 2), ya, val);
			}
			xa++;
		}
		ya++;
	}

	return true;
}

static void Normalize_Matrix_min_max(Matrix *m)
{
	for (int x = 0; x < m->width; ++x) {
		// calculate std for each column
		float min = Matrix_Element_Required(m, x, 0);
		float max = Matrix_Element_Required(m, x, 0);
		for (int y = 1; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			if (val < min) {
				min = val;
			} else if (val > max) {
				max = val;
			}
		}

		for (int y = 0; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			InitializeMatrix(m, x, y, (val - min) / max);
		}
	}
}

static void InitializeRandom(Matrix *mat, float LO, float HI)
{
	for (int i = 0; i < mat->width; ++i) {
		for (int j = 0; j < mat->height; ++j) {
			float r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
			InitializeMatrix(mat, i, j, r);
		}
	}
}

void matrixMulKernel(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh,
                     const sycl::nd_item<3> &item_ct1)
{
        int row = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);
        int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);

        if ((row < rh) && (col < rw)) {
		// dot product
		float accum = 0.0f;
		for (int c = 0; c < m1w; c++)
		{
			float v1 = m1[row * m1w + c];
			float v2 = m2[c * m2w + col];
			accum += (v1 *  v2);
		}

		r[row * rw + col] = accum;
	}
}

void sigmoidKernel(float *r, int m, const sycl::nd_item<3> &item_ct1)
{
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        if (index < m) {
		float val = r[index];
                r[index] = 1.0 / (1.0 + sycl::exp(-val));
        }
}

void matrixAbsErrorKernel(float *p, float *ys, float *r, int rw, int rh,
                          const sycl::nd_item<3> &item_ct1)
{
        int row = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);
        int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);

        if ((row < rh) && (col < rw)) {
		float pval = p[row * rw + col];
		float ysval = ys[row * rw + col];

		float v = pval - ysval;
		r[row * rw + col] = v * v;
	}
}

void absErrorKernel(float *p, float *ys, float *r, int m,
                    const sycl::nd_item<3> &item_ct1)
{
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float v = pval - ysval;
		r[index] = v * v;
	}
}

void updateParamsAbsErrorKernel(float *p, float *ys, float *th, float *xs, int m, float alpha,
                                const sycl::nd_item<3> &item_ct1)
{
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        if (index < m) {
		float h = *p;
		float y = *ys;

		float x = xs[index];

		th[index] = th[index] - alpha * (h - y) * x;
	}
}

void crossEntropyKernel(float *p, float *ys, float *r, int m,
                        const sycl::nd_item<3> &item_ct1)
{
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

                float ex = sycl::log1p(sycl::exp(-ysval * pval));
                r[index] = ex;
	}
}

void reduceKernel(float * input, float * output, int len,
                  const sycl::nd_item<3> &item_ct1, float *partialSum) {
    //@@ Load a segment of the input vector into shared memory

    unsigned int t = item_ct1.get_local_id(2),
                 start = 2 * item_ct1.get_group(2) * REDUCE_BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + REDUCE_BLOCK_SIZE + t < len)
       partialSum[REDUCE_BLOCK_SIZE + t] = input[start + REDUCE_BLOCK_SIZE + t];
    else
       partialSum[REDUCE_BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = REDUCE_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       item_ct1.barrier(sycl::access::fence_space::local_space);
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[item_ct1.get_group(2)] = partialSum[0];
}

static void Logistic_Regression_SYCL(Matrix *X, Matrix *y, Matrix *Parameters, Matrix *Train_Parameters, int maxIterations, float alpha, vector<float> &cost_function)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        // put stuff into gpu
	float *gpu_X;
	float *gpu_y;

	float *gpu_prediction;

	float *gpu_params;
	float *gpu_abs_error;
	float *gpu_err_cost;

	float *gpu_predictions;
	Matrix predictions;
	AllocateMatrix(&predictions, y->height, y->width);

	Matrix absErrors;
	AllocateMatrix(&absErrors, y->height, y->width);

	float mean_error;
	float sum=0;
	int quantity = 1;

	int m = y->height;

	int numOutputElements;
	numOutputElements = m / (REDUCE_BLOCK_SIZE<<1);
	if (m % (REDUCE_BLOCK_SIZE<<1)) {
		numOutputElements++;
	}

        SAFE_CALL((gpu_X = (float *)sycl::malloc_device(
                       sizeof(float) * X->width * X->height, q_ct1),
                   0));
        
        SAFE_CALL((gpu_y = (float *)sycl::malloc_device(
                       sizeof(float) * y->width * y->height, q_ct1),
                   0));
        
        SAFE_CALL((gpu_prediction = sycl::malloc_device<float>(1, q_ct1), 0));
        
        SAFE_CALL((gpu_predictions = (float *)sycl::malloc_device(
                       sizeof(float) * y->width * y->height, q_ct1),
                   0));
        
        SAFE_CALL((gpu_abs_error = (float *)sycl::malloc_device(
                       sizeof(float) * y->width * y->height, q_ct1),
                   0));
       
        SAFE_CALL(
            (gpu_params = (float *)sycl::malloc_device(
                 sizeof(float) * Parameters->width * Parameters->height, q_ct1),
             0));
        
        SAFE_CALL((gpu_err_cost =
                       sycl::malloc_device<float>(numOutputElements, q_ct1),
                   0));

        
        SAFE_CALL((q_ct1
                       .memcpy(gpu_X, X->elements,
                               sizeof(float) * X->width * X->height)
                       .wait(),
                   0));
        
        SAFE_CALL((q_ct1
                       .memcpy(gpu_y, y->elements,
                               sizeof(float) * y->width * y->height)
                       .wait(),
                   0));
        
        SAFE_CALL(
            (q_ct1
                 .memcpy(gpu_params, Parameters->elements,
                         sizeof(float) * Parameters->width * Parameters->height)
                 .wait(),
             0));

        // invoke kernel
	static const int blockWidth = 16;
	static const int blockHeight = blockWidth;
	int numBlocksW = X->width / blockWidth;
	int numBlocksH = X->height / blockHeight;
	if (X->width % blockWidth) numBlocksW++;
	if (X->height % blockHeight) numBlocksH++;

        sycl::range<3> dimGrid(1, numBlocksH, numBlocksW);
        sycl::range<3> dimBlock(1, blockHeight, blockWidth);

        sycl::range<3> dimReduce(1, 1, (m - 1) / REDUCE_BLOCK_SIZE + 1);
        sycl::range<3> dimReduceBlock(1, 1, REDUCE_BLOCK_SIZE);

        sycl::range<3> dimVectorGrid(1, 1, ((m - 1) / blockWidth * blockWidth) + 1);
        sycl::range<3> dimVectorBlock(1, 1, blockWidth * blockWidth);

        float* error_accum = new float[numOutputElements];
	for (int iter = 0; iter < maxIterations; ++iter) {
		for (int i = 0; i < m; ++i) {
                        
                        q_ct1.submit([&](sycl::handler &cgh) {
                                auto gpu_X_i_X_width_ct0 = &gpu_X[i * X->width];
                                auto X_width_ct3 = X->width;
                                auto Parameters_width_ct4 = Parameters->width;

                                cgh.parallel_for(
                                    sycl::nd_range<3>(dimGrid * dimBlock,
                                                      dimBlock),
                                    [=](sycl::nd_item<3> item_ct1) {
                                            matrixMulKernel(
                                                gpu_X_i_X_width_ct0, gpu_params,
                                                gpu_prediction, X_width_ct3,
                                                Parameters_width_ct4, 1, 1,
                                                item_ct1);
                                    });
                        });
                        
                        q_ct1.parallel_for(
                            sycl::nd_range<3>(dimVectorGrid * dimVectorBlock,
                                              dimVectorBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    sigmoidKernel(gpu_prediction, 1, item_ct1);
                            });
                        
                        q_ct1.submit([&](sycl::handler &cgh) {
                                auto gpu_y_i_ct1 = &gpu_y[i];
                                auto gpu_X_i_X_width_ct3 = &gpu_X[i * X->width];
                                auto Parameters_height_ct4 = Parameters->height;

                                cgh.parallel_for(
                                    sycl::nd_range<3>(dimVectorGrid *
                                                          dimVectorBlock,
                                                      dimVectorBlock),
                                    [=](sycl::nd_item<3> item_ct1) {
                                            updateParamsAbsErrorKernel(
                                                gpu_prediction, gpu_y_i_ct1,
                                                gpu_params, gpu_X_i_X_width_ct3,
                                                Parameters_height_ct4, alpha,
                                                item_ct1);
                                    });
                        });
                }
                
                q_ct1.submit([&](sycl::handler &cgh) {
                        auto X_width_ct3 = X->width;
                        auto Parameters_width_ct4 = Parameters->width;
                        auto predictions_width_ct5 = predictions.width;
                        auto predictions_height_ct6 = predictions.height;

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    matrixMulKernel(
                                        gpu_X, gpu_params, gpu_predictions,
                                        X_width_ct3, Parameters_width_ct4,
                                        predictions_width_ct5,
                                        predictions_height_ct6, item_ct1);
                            });
                });
                
                q_ct1.parallel_for(
                    sycl::nd_range<3>(dimVectorGrid * dimVectorBlock,
                                      dimVectorBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                            sigmoidKernel(gpu_predictions, m, item_ct1);
                    });

                // calculate error
                
                q_ct1.parallel_for(
                    sycl::nd_range<3>(dimVectorGrid * dimVectorBlock,
                                      dimVectorBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                            absErrorKernel(gpu_predictions, gpu_y,
                                           gpu_abs_error, m, item_ct1);
                    });
                
                q_ct1.submit([&](sycl::handler &cgh) {
                        
                        sycl::local_accessor<float, 1> partialSum_acc_ct1(
                            sycl::range<1>(256 /*2 * REDUCE_BLOCK_SIZE*/), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(dimReduce * dimReduceBlock,
                                              dimReduceBlock),
                            [=](sycl::nd_item<3> item_ct1) {
                                    reduceKernel(
                                        gpu_abs_error, gpu_err_cost, m,
                                        item_ct1,
                                        partialSum_acc_ct1.get_pointer());
                            });
                });
               
                SAFE_CALL((q_ct1
                               .memcpy(error_accum, gpu_err_cost,
                                       sizeof(float) * numOutputElements)
                               .wait(),
                           0));
                float g_sum = 0;
		for (int i = 0; i < numOutputElements; ++i)
		{
			g_sum += error_accum[i];
		}

		g_sum /= (2*m);

		cost_function.push_back(g_sum);
		sum += g_sum;
		quantity++;
		cout << g_sum << "\n";
	}

	mean_error = sum/quantity;
	printf("\n The mean error is %f\n", mean_error);
	cout << endl;

	delete[] error_accum;
        
        SAFE_CALL((sycl::free(gpu_X, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_y, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_abs_error, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_prediction, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_predictions, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_params, q_ct1), 0));
        
        SAFE_CALL((sycl::free(gpu_err_cost, q_ct1), 0));
}

int main(int argc, char *argv[])
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        string input_file = "";
	cout << "Please enter a valid file to run test for logistic regression on SYCL:\n>";
	getline(cin, input_file);
 	cout << "You entered: " << input_file << endl << endl;
    Matrix X,y;
    setup_data (input_file, &X, &y);
    cout <<"\n The X - Squiggle Matrix." << endl;
    DisplayMatrix (X,true);
    cout <<"\n The y - Matrix." << endl;
    DisplayMatrix (y,true);

    Matrix Parameters, Train_Parameters;
    //Setup matrices with 1 as value initially
    AllocateMatrix(&Parameters, X.width, 1);
    AllocateMatrix(&Train_Parameters, X.width, 1);
    //Initialize with random +1 and -1 parameters.
    InitializeRandom(&Parameters, -1.0, 1.0);

    Normalize_Matrix_min_max(&X);

    vector<float> cost_function;

    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
        start = new sycl::event();
        stop = new sycl::event();

    start_ct1 = std::chrono::steady_clock::now();
    Logistic_Regression_SYCL(&X, &y, &Parameters, &Train_Parameters, 150, 0.03, cost_function);
    
    stop_ct1 = std::chrono::steady_clock::now();

        float milliseconds = 0;
        milliseconds =
            std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                .count();
	printf("\nProcessing time: %f (ms)\n", milliseconds);
    std::cout << "============succeed!============" << std::endl;

	return 0;
}