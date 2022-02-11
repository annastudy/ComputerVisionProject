# import numpy
import numpy as np


# Create and print a 3 by 3 array where every number is a 15
print (15*np.ones((3,3)))
# print out what are the largest and smalled values in the array below
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print ('max =', arr.max())
# import pyplot lib from matplotlib and Image lib from PIL
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
# use PIL and matplotlib to read and display the ../data/zebra.jpg image
pic = plt.imread('../data/zebra.jpg')
plt.imshow (pic)
plt.show()
# convert the image to a numpy arrary and print the shape of the arrary
pic_arr = np.array (pic)
print (pic_arr.shape)
# use slicing to set the RED and GREEN channels of the picture to 0, then use imshow() to show the isolated blue channel
pic_new = pic_arr.copy()
pic_new [:,:,1] = 0
pic_new[:,:,0] = 0
plt.imshow (pic_new)
plt.show()
