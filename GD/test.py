caffe_root = '../../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def warp_image(im):
	im = im - [129,104,93]
	im = im.transpose(2,0,1) #WHC to CWH
	im = im[::-1] #RGB to BGR
	im = np.require(im,dtype = np.float32)
	return im

def unwarp_image(im):
	im = im.copy()
	im = im[::-1] #BGR to RBG
	im = im.transpose(1,2,0) #CWH to WHC
	im = im + [129,104,93] # reverse the averaging
	im[im<0],im[im>255]= 0,255
	im = np.round(im)
	im = np.require(im,dtype = np.uint8)
	return im

import tooptimize as obj
import GD

def progress(x,g,f_x,xnorm,gnorm,step,k,ls):
	"""Report optimization procedure """
	print("step:{:d}, xnorm{:f}, gnorm{:f}, f(x):{:f}".format(step,xnorm, gnorm, f_x))

	
#preloading
caffe.set_device(3)
caffe.set_mode_gpu()
net = caffe.Net('./VGG_Attr_deploy.prototxt', 
		'../CelebA_from_VGGFace/model_iter_50000.caffemodel'
		,caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

for i in range(40):
	im = Image.open('../test.png')
	#im = Image.open('/home/yiliny1/faceAttributes/img_align_celeba/' + str(i+1).zfill(6) + '.jpg')
	#im = Image.open('/home/yiliny1/faceAttributes/img_align_celeba/000007.jpg')
	#im = Image.open('/home/yiliny1/faceAttributes/LFW_from_VGGFace/lfw/David_Beckham/David_Beckham_0001.jpg')
	im = im.resize([224,224],Image.ANTIALIAS)
	im = np.array(im)

	#process the input image
	im = warp_image(im)

	#set desired output
	im_input = im[np.newaxis,:,:,:]
	net.blobs['data'].data[...] = im_input
	target_out = net.forward()['fc8_CelebA'].copy()
	print target_out
	target_out[0][i] = -40 * np.sign(target_out[0][i])

	#set parameters
	Lambda = 1e-8
	im0 = im.copy()

	#optimize!
	#im_output = lbfgs.fmin_lbfgs(lambda x,g:obj.f(x,g,im0,target_out,net,Lambda),
	#                            im0,progress)
	im_output = GD.GD(lambda x:obj.f(x,im0,target_out,net,Lambda),im0,10,5000)

	result = unwarp_image(im_output)
	plt.imsave('out/out_L1_attr' + str(i) + '_test.png', result)
	#result.save('out.png')

