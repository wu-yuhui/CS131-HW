import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import color
import sys

# Works for arbitrary pixel value image, and scale it to 1 byte([0,255])
def histogram(img):
	img_max = img.max()
	h = np.zeros(256)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			val = int(img[i, j]/img_max * 256)
			if val == 256:	# img_max pixel will be out-of-bounds
				val = 255
			h[val] += 1
	plt.plot(np.arange(256), h)
	plt.show()


def main():
	img = io.imread(sys.argv[1])
	#img = io.imread(f)
	img_gray = color.rgb2gray(img)
	histogram(img_gray)

if __name__ == "__main__":
	main()
