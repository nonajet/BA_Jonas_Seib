import numpy as np

a = np.arange(36).reshape(6, 6)

b = np.array([[0, 0, 0, 0, 0, 0],
              [0, 7, 8, 9, 10, 0],
              [0, 13, 14, 15, 16, 0],
              [1, 19, 20, 21, 22, 0],
              [1, 25, 26, 27, 28, 0],
              [1, 0, 0, 0, 0, 0]])

c = np.array([1, 2, 3])
d = np.array([0, 1, 1])
print(c.any())
print(d.any())

arr = [c, d]
arr = [np.where(i < 2, 0, i) for i in arr]
print(arr)

if __name__ == '__main__':
    pass
