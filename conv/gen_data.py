import numpy as np 

def out_shape(in_shape,kernel_shape,stride = 1,pading = 0):
	H0,W0,C0 = in_shape
	K,K,C0,C1 = kernel_shape
	H1 = (H0 - K + 1 + pading)//stride
	W1 = (W0 - K + 1 + pading)//stride
	return [H1,W1,C1]	


def conv(input_data,kernel,stride = 1,pading = 0):
	output_shape = out_shape(input_data.shape,kernel.shape,stride,pading)
	output_data =np.zeros(output_shape,dtype = np.float32)
	for i in range(output_shape[0]):
		for j in range(output_shape[1]):		 
			for k in range(output_shape[2]):
				value = 0
				for m in range(kernel.shape[0]):
					for n in range(kernel.shape[1]):
						for l in range(kernel.shape[2]):
							value += input_data[stride * i + m,stride * j + n, l] *  kernel[m,n,l,k]
				output_data[i,j,k] = value
	return output_data


def gen_data(in_shape,kernel_shape,stride = 1,pading = 0,min_value = 0,max_value = 10):
	print("inshape =",in_shape)
	print("kernel_shape",kernel_shape)
	output_shape = out_shape(in_shape ,kernel_shape ,stride ,pading)
	print("output_shape",output_shape)
	input_data = np.random.rand(*in_shape).astype(np.float32) * (max_value-min_value) - min_value
	kernel_data = np.random.rand(*kernel_shape).astype(np.float32) * (max_value-min_value) - min_value
	except_data = conv(input_data,kernel_data ,stride ,pading)
	input_data.tofile("./data/input.bin")
	kernel_data.tofile("./data/kernel.bin")	
	except_data.tofile("./data/except.bin")


def check_data(output_data,except_data):
	diff = np.abs(output_data - except_data)
	print("="*40,"output_data","="*40)
	print(output_data)
	print("="*40,"except_data","="*40)
	print(except_data)	
	print("="*40,"max value of diff","="*40)
	print(np.max(diff))


def verify_algo(input_data,kernel_data,stride = 1,pading = 0): 
	import os
	import tensorflow as tf

	input_tensor = tf.Variable(input_data,dtype = np.float32)
	filter_data = tf.Variable(kernel_data, dtype=np.float32)
	output_tensor = tf.nn.conv2d(input_data, filter_data, strides=[1, stride, stride, 1], padding="VALID") 
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init) 
		res=sess.run(output_tensor) 
		res.tofile("./data/tf_output.bin")


if __name__=="__main__":
	"""
	in_shape = [18,18,128]
	kernel_shape = [3,3,128,256]
	gen_data(in_shape,kernel_shape)  	
	
	input_data = np.fromfile("input.bin",dtype = np.float32).reshape([1,18,18,128])
	kernel_data = np.fromfile("kernel.bin",dtype = np.float32).reshape([3,3,128,256])
	verify_algo(input_data,kernel_data)
	"""
	output_data = np.fromfile("./data/except.bin",dtype = np.float32).reshape([16,16,256])
	except_data = np.fromfile("./data/output.bin",dtype = np.float32).reshape([16,16,256])
	check_data(output_data,except_data)

