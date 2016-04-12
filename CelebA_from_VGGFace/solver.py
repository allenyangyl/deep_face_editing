from __future__ import division
import sys

caffe_root = '../../caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(6)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('/home/yiliny1/faceAttributes/vgg_face_caffe/VGG_FACE.caffemodel')

niter = 200000
train_loss = np.zeros(niter)

f = open('loss2.txt', 'w')

for it in range(niter): 
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: d} '.format(it))
    f.write('{0: f}\n'.format(train_loss[it]))

f.close()

# solver.step(80000)
