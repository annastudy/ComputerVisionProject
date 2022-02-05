# import numpy, matplotlib.pyplot and PIL's Image libs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print('open image file using Image open function in pillow library')
pic = Image.open('../data/bird.jpg')
print (pic)
print (type(pic))
print('cast the image data into numpy array')
pic_arr = np.array(pic)

print('use plt image show function to show the image')

plt.imshow (pic_arr)
plt.show()
print('copy picture array into a new variable so that we can adjust the RGB channel later on.')
pic_red =  pic_arr.copy()
plt.imshow (pic_red)
plt.show()
print('zero out green and blue channels\' value and only keep color value in Red channel.')
pic_red[:,:,1]=0 #Zero out contribution from green
pic_red[:,:,2]=0 #Zero out contribution from blue

plt.imshow (pic_red)
plt.show()
print('jfkldf;try to debug the pic_red variable and see what are the RGB value examples in 3 channels after zero out.')

pic_red = pic_arr.copy()
print('show the result using imshow function')
