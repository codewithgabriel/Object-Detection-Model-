# %%
import cv2

# %%
img = cv2.imread("images/me.png", 0)
print(img)
print(img.shape)
print(img.dtype)


import matplotlib.pyplot as plt

plt.imshow(img)
# %%
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(img, cmap='gray')

# %%
coke_img = cv2.imread("images/me.png", cv2.IMREAD_COLOR)

# split color

b, g, r = cv2.split(coke_img)

plt.figure(1, figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(b)
plt.title("Blue")

plt.subplot(1, 4, 2)
plt.imshow(g)
plt.title("Green")

plt.subplot(1, 4, 3)
plt.imshow(r)
plt.title("Red")

# %%
m = cv2.merge((b, g, r))
plt.imshow(m[:, :, ::-1])


import numpy as np
arr1 = np.array([200, 250], dtype=np.uint8).reshape(-1, 1)
arr2 = np.array([40, 40], dtype=np.uint8).reshape(-1, 1)
add_numpy = arr1+arr2
add_cv2 = cv2.add(arr1, arr2)