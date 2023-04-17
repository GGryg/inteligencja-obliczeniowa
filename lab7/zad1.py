import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)

# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

def apply_convolution(img, kernel, stride):
    height, width = img.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (height - kernel_height) // stride + 1
    output_width = (width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))
    for i in range(0, height - kernel_height + 1, stride):
        for j in range(0, width - kernel_width + 1, stride):
            patch = img[i:i+kernel_height, j:j+kernel_width]
            output[i//stride, j//stride] = np.sum(patch * kernel)
    return output

gray_data = np.mean(data, axis=2)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
# b
# result = apply_convolution(gray_data, kernel, 1)
# result = convolve2d(gray_data, kernel, mode='valid')

# c
# result = apply_convolution(gray_data, kernel, 2)

# d
kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# result = apply_convolution(gray_data, kernel, 1)
result = apply_convolution(gray_data, kernel, 2)

plt.subplot(1, 2, 1)
plt.imshow(data, interpolation='nearest')
plt.title('Obrazek przed konwolucją')
plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray', interpolation='nearest')
plt.title('Obrazek po konwolucji')
plt.show()