

__global__ void conv1(float * input,float * output,float * kernel)
{
	float value = 0;
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			for(int k=0;k<128;k++){
				value += *(input + (threadIdx.x + i) * 18 *128 +(threadIdx.y + j) * 128 + k) * \
				*(kernel + i * 3 * 128 * 256 + j * 128 * 256 + k * 256 + blockIdx.x); 
			} 
		}
	}
	*(output + threadIdx.x * 16 * 256 + threadIdx.y * 256 + blockIdx.x) = value;
}

#include <stdio.h>


int main()
{
	// conv param
	int H0 = 18 , W0 = 18 , C0 = 128;
	int H1 = 16 , W1 = 16 , C1 = 256;
	int K = 3; 


	float * input, * output, * kernel;
	int input_size = H0 * W0 * C0;
	int output_size = H1 * W1 * C1;
	int kernel_size = K * K * C0 * C1;


	FILE * f_input = fopen("./data/input.bin","rb");
	FILE * f_kernel = fopen("./data/kernel.bin","rb");

	if(f_input && f_kernel){
		input = (float *)malloc(input_size * sizeof(float));
		output = (float *)malloc(output_size * sizeof(float));
		kernel = (float *)malloc(kernel_size * sizeof(float));
		fread(input,sizeof(float),input_size,f_input);
		fread(kernel,sizeof(float),kernel_size,f_kernel);	
		fclose(f_input);
		fclose(f_kernel);	
	}else{
		printf("task is fail\n");
		return 0;
	}

	float * dev_input, * dev_output, * dev_kernel;
	cudaMalloc((void**)&dev_input, input_size * sizeof(float));	
	cudaMalloc((void**)&dev_output, output_size * sizeof(float));	
	cudaMalloc((void**)&dev_kernel, kernel_size * sizeof(float));	

	cudaMemcpy(dev_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

	dim3 b(256),t(16,16);
	conv1<<<b,t>>>(dev_input,dev_output,dev_kernel);

	cudaMemcpy(output, dev_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

	FILE * f_output = fopen("./data/output.bin","wb");
	if(f_output){
		fwrite(output,sizeof(float),output_size,f_output);
		fclose(f_output);
	} 

	free(input);
	free(output);
	free(kernel);

    cudaFree(dev_input);  
    cudaFree(dev_output);  
    cudaFree(dev_kernel);  
}