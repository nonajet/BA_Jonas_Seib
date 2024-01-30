from matplotlib import pyplot as plt

import numpy as np
from skimage import measure
from skimage.draw import disk

a = np.arange(36).reshape(6, 6)

b = np.array([[0, 0, 0, 0, 0, 0],
              [0, 7, 8, 9, 10, 0],
              [0, 13, 14, 15, 16, 0],
              [0, 19, 20, 21, 22, 0],
              [0, 25, 26, 27, 28, 0],
              [0, 0, 0, 0, 0, 0]])

image = np.zeros((100, 100), dtype=np.uint8)
rr, cc = disk((30, 30), radius=15)
image[rr, cc] = 1

rr, cc = disk((70, 70), radius=20)
image[rr, cc] = 2

# Zeige das Beispielbild
plt.imshow(image, cmap='gray')
plt.title('Beispielbild mit zwei Regionen')
plt.show()
regions = measure.regionprops(image)

for region in regions:
    print(f"\nRegion mit Label {region.label}:")
    print(f"Fläche: {region.area}")
    print(f"Schwerpunkt: {region.centroid}")
    print(f"Bounding Box: {region.bbox}")
    print(f"Perimeter: {region.perimeter}")

if __name__ == '__main__':
    pass
