#include "ex1.h"


#define HIST_LENGTH 256
#define IMG_SIZE IMG_WIDTH*IMG_HEIGHT

__device__ void prefix_sum(int arr[], int arr_size) {
    int threadID = threadIdx.x;
    int offset = 1;
    int last = arr[arr_size-1];
    for(int level = arr_size / 2; level > 0; level /= 2)
    {
    	if(threadID < level)
    	{
    		arr[offset * (2 * threadID + 2) - 1] += arr[offset * (2 * threadID + 1) - 1];
    	}
    	offset *= 2;
    	__syncthreads(); 
    }
    if(threadID == 0)
    {
    	arr[arr_size - 1] = 0;
    }
    for(int level = 1; level < arr_size; level *= 2)
    {
    	offset /= 2;
    	__syncthreads();
    	if(threadID < level)
    	{
    		int temp = arr[offset * (2 * threadID + 1) - 1];
    		arr[offset * (2 * threadID + 1) - 1] = arr[offset * (2 * threadID + 2) - 1];
    		arr[offset * (2 * threadID + 2) - 1] += temp;
    	}
    }
	__syncthreads(); 
    if(threadID == 0){
        for(int i=0; i<arr_size-1;i++){
        	arr[i]=arr[i+1];
        }
        arr[arr_size-1]= arr[arr_size-1]+last;
    }
	return;
}

//serial 
/*__global__ void process_image_kernel(uchar *all_in, uchar *all_out) {
    __shared__ int hist[HIST_LENGTH];

    int threadID = threadIdx.x;
    int blockSize = blockDim.x;

    if(threadID < HIST_LENGTH)
    {
    	hist[threadID] = 0;
    }

    //	Create the histogram
    for(int i = threadID; i < IMG_SIZE; i += blockSize)
    {
    	atomicAdd(&hist[all_in[i]], 1);
    }
    __syncthreads();

    //	Create the CDF
    prefix_sum(hist, HIST_LENGTH);
    __syncthreads();

    //	Create the map
    if(threadID < HIST_LENGTH)
    {
        hist[threadID] = (HIST_LENGTH / N_COLORS) * (int)(N_COLORS * (float)hist[threadID] / (IMG_WIDTH * IMG_HEIGHT));
    }
    __syncthreads();

    //	Compute the new image
    for(int i = threadID; i < IMG_SIZE; i += blockSize)
        {
        	all_out[i] = hist[all_in[i]];
        }
	return;
}*/


//block
__global__ void process_image_kernel(uchar *all_in, uchar *all_out) {
    __shared__ int hist[HIST_LENGTH];

    int threadID  =  threadIdx.x;
    int blockID   =  blockIdx.x;
    int blockSize =  blockDim.x;

    if(threadID < HIST_LENGTH)
    {
    	hist[threadID] = 0;
    }
    __syncthreads();
    //	Create the histogram
    for(int i = threadID; i < IMG_SIZE; i += blockSize)
    {
    	atomicAdd(&hist[all_in[i + blockID*IMG_SIZE]], 1);
    }
    __syncthreads();

    //	Create the CDF
    prefix_sum(hist, HIST_LENGTH);
    __syncthreads();

    //	Create the map
    if(threadID < HIST_LENGTH)
    {
        hist[threadID] = (HIST_LENGTH / N_COLORS) * (int)(N_COLORS * (float)hist[threadID] / (IMG_SIZE));
    }
    __syncthreads();

    //	Compute the new image
    for(int i = threadID; i < IMG_SIZE; i += blockSize)
        {
        	all_out[i + blockID*IMG_SIZE] = hist[all_in[i + blockID*IMG_SIZE]];
        }
	return;
}
/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *all_in, *all_out;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    CUDA_CHECK(cudaMalloc((void**)&context->all_in, IMG_SIZE*sizeof(uchar)));
    CUDA_CHECK(cudaMalloc((void**)&context->all_out, IMG_SIZE*sizeof(uchar)));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    for(int imageIdx = 0; imageIdx < N_IMAGES; imageIdx++)
    {
    	CUDA_CHECK(cudaMemcpy((void*)context->all_in, (void*)(images_in + imageIdx * IMG_SIZE), IMG_SIZE*sizeof(uchar), cudaMemcpyHostToDevice));
    	process_image_kernel<<<1, 1024>>>(context->all_in, context->all_out);
    	CUDA_CHECK(cudaDeviceSynchronize());
    	CUDA_CHECK(cudaMemcpy((void*)(images_out + imageIdx * IMG_SIZE), (void*)context->all_out, IMG_SIZE*sizeof(uchar), cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree((void*)context->all_in));
    CUDA_CHECK(cudaFree((void*)context->all_out));

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
	uchar *all_in, *all_out;
};

/* Allocate GPU memory for all the input and output images.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    CUDA_CHECK(cudaMalloc((void**)&context->all_in,  N_IMAGES*IMG_SIZE*sizeof(uchar)));
    CUDA_CHECK(cudaMalloc((void**)&context->all_out, N_IMAGES*IMG_SIZE*sizeof(uchar)));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
	CUDA_CHECK(cudaMemcpy((void*)context->all_in, (void*)(images_in), N_IMAGES*IMG_SIZE*sizeof(uchar), cudaMemcpyHostToDevice));
	process_image_kernel<<<N_IMAGES, 1024>>>(context->all_in, context->all_out);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy((void*)(images_out), (void*)context->all_out, N_IMAGES*IMG_SIZE*sizeof(uchar), cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree((void*)context->all_in));
    CUDA_CHECK(cudaFree((void*)context->all_out));
    free(context);
}

