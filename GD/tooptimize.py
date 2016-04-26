import numpy as np
import caffe

def f(im,im0,target,net,Lambda):
    #target ter
    mul = 1
    im_input = im[np.newaxis,:,:,:]
    #print im_input.shape
    net.blobs['data'].data[...] = im_input
    netoutput = net.forward()
    #print netoutput
    #print target
    netoutput = netoutput['fc8_CelebA']
    top_diff = 2*(netoutput - target)
    #print top_diff
    net.blobs['fc8_CelebA'].diff[...] = top_diff
    #print net.blobs['fc8_CelebA'].diff
    bottom_diff = net.backward()
    bottom_diff = bottom_diff['data']
    der = bottom_diff[0,:,:,:]
    output = np.sum((netoutput - target)**2)
  
    #regularization term
    #der = der + Lambda*2*(im - im0)
    der = der + Lambda * np.sign(im - im0)
    #output = output + Lambda*np.sum((im-im0)**2)  
    #der *= mul
    #output *= mul
    #np.copyto(g,der)
    return der
