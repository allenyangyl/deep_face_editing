import numpy as np
import matplotlib.pyplot as plt

def GD(gradient, I0, ita = 1, max_iter = 10000, min_loss = 1e-10):
	iter = 0
	I = I0
	while True:
		gI = gradient(I)
		I = I - ita * gI
		loss = np.mean(np.square(gI))
		iter = iter + 1
		print 'iter: ' + str(iter) + '; loss: ' + str(loss)
#		print iter
#		print I
#		print gI
#		print loss
		if iter >= max_iter or loss < min_loss: 
			return I

#def g2(I):
#	return 2 * (I - 5)
#
#img = GD(gradient = g2, I0 = np.array([[1,2],[3,4]]))
