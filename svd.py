import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread("image0.jpg") #Read image
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255 #Turn it from [0,255] to [0,1] and into grayscale

def svd(image):
    u, s, v = np.linalg.svd(image) #SVD decomposition
    return u, s, v

def truncate(u,s,v,k):
    u_k=u[:,:k] #Take the first k columns of u
    s_k=np.diag(s[:k]) #Takgie the first k singular values and make a diagonal matrix
    v_k=v[:k,:] #Take the first k rows of v 
    return u_k, s_k, v_k

def create_img(u,s,v):
    return np.matmul(u, np.matmul(s, v)) #Recreate the image from the SVD decomposition


#Show truncated images
fig1 = plt.figure(figsize=(10, 8))

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.02)



plt.subplot(3,2,1)
plt.imshow(create_img(*truncate(*svd(grey), 5)), cmap='gray', interpolation='antialiased')
plt.title("k=5")


plt.subplot(3,2,2)
plt.imshow(create_img(*truncate(*svd(grey), 15)), cmap='gray', interpolation='antialiased')
plt.title("k=15")


plt.subplot(3,2,3)
plt.imshow(create_img(*truncate(*svd(grey), 50)), cmap='gray', interpolation='antialiased')
plt.title("k=50")


plt.subplot(3,2,4)
plt.imshow(create_img(*truncate(*svd(grey), 200)), cmap='gray', interpolation='antialiased')
plt.title("k=200")


plt.subplot(3,2,5)
plt.imshow(grey, cmap='gray', interpolation='antialiased')
plt.title("Original")



#Show singular values, linearly
fig2 = plt.figure(figsize=(10, 8))

plt.plot(np.diag(svd(grey)[1]))




#Show singular values, logarithmically
fig3 = plt.figure(figsize=(10, 8))

plt.plot(np.diag(svd(grey)[1]))
plt.yscale('log')


def frobenius_norm(img):
    return np.linalg.norm(img)

print("Frobenius norm of original image: ", frobenius_norm(grey))

print("Error at k=5: ", frobenius_norm(grey-create_img(*truncate(*svd(grey), 5)))/frobenius_norm(grey))
print("Error at k=15: ", frobenius_norm(grey-create_img(*truncate(*svd(grey), 15)))/frobenius_norm(grey))
print("Error at k=50: ", frobenius_norm(grey-create_img(*truncate(*svd(grey), 50)))/frobenius_norm(grey))
print("Error at k=200: ", frobenius_norm(grey-create_img(*truncate(*svd(grey), 200)))/frobenius_norm(grey))




plt.show()
