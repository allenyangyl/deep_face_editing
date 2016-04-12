import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = 'TestLabels.txt'
result_file = 'cls_results_200000.txt'

caffe.set_device(4)
caffe.set_mode_gpu()
net = caffe.Net('/home/yiliny1/faceAttributes/CelebA_from_VGGFace/test.prototxt',
                '/home/yiliny1/faceAttributes/CelebA_from_VGGFace/model_iter_200000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
label_count = 40
accuracy = np.zeros(label_count)

f = open(result_file, 'w')
print(batch_count)
for i in range(batch_count):

	out = net.forward()
	print(i)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break
		
		fname = test_list[id].split()[0]
		f.write(fname)
		for l in range(label_count):
			lbl = int(test_list[id].split()[l + 1])
		
			prop = out['fc8_CelebA'][j][l]
			pred_lbl = int(prop >= 0)
			if pred_lbl == lbl:
				accuracy[l] = accuracy[l] + 1

			f.write('{0: d}'.format(pred_lbl))
		f.write('\n')

for l in range(label_count):
	accuracy[l] = accuracy[l] * 1.0 / data_counts
	f.write('{0: d}'.format(l))
	f.write('{0: f}'.format(accuracy[l]))
	f.write('\n')
accuracy = np.sum(accuracy) / (label_count)

print accuracy
f.write('accuracy: {0: f}'.format(accuracy))

f.close()
