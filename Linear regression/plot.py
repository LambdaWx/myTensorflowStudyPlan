# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
import numpy as np
import os
import time
import datetime
import random
import matplotlib.pyplot as plt 

def inputs():
	X = [x*2+random.random() for x in range(50)]
	Y = [0.638*x+0.55+np.random.normal(0, 1.0, 1) for x in X]
	return X, Y
X_in,Y_in = inputs()
plot1 = plt.plot(X_in, Y_in, 'or',label='traing dataset')

W=0.645892
b=0.023363
Y_predict = [W*x+b for x in X_in]
plot2 = plt.plot(X_in, Y_predict,label='model prediction')

legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()
