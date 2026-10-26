import numpy as np
import cv2
from matplotlib import pyplot as plt



img = cv2.imread("figures/sunset.png") #Read image
grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255 #Turn it from [0,255] to [0,1] and into grayscale

def svd(image):
    u, s, v = np.linalg.svd(image) #SVD decomposition
    return u, s, v

def truncate(u,s,v,k):
    u_k=u[:,:k] #Take the first k columns of u
    s_k=np.diag(s[:k]) #Take the first k singular values and make a diagonal matrix
    v_k=v[:k,:] #Take the first k rows of v 
    return u_k, s_k, v_k

def create_img(u,s,v):
    return np.matmul(u, np.matmul(s, v)) #Recreate the image from the SVD decomposition


#Show truncated images
fig1 = plt.figure(figsize=(10, 8))

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.02)

svd_grey = svd(grey)

plt.subplot(3,2,1)
plt.imshow(create_img(*truncate(*svd_grey, 5)), cmap='gray', interpolation='antialiased')
plt.title("k=5")


plt.subplot(3,2,2)
plt.imshow(create_img(*truncate(*svd_grey, 15)), cmap='gray', interpolation='antialiased')
plt.title("k=15")


plt.subplot(3,2,3)
plt.imshow(create_img(*truncate(*svd_grey, 50)), cmap='gray', interpolation='antialiased')
plt.title("k=50")


plt.subplot(3,2,4)
plt.imshow(create_img(*truncate(*svd_grey, 200)), cmap='gray', interpolation='antialiased')
plt.title("k=200")


plt.subplot(3,2,5)
plt.imshow(grey, cmap='gray', interpolation='antialiased')
plt.title("Original")



#Show singular values, linearly
fig2 = plt.figure(figsize=(10, 8))

plt.plot(np.diag(svd_grey[1]))




#Show singular values, logarithmically
fig3 = plt.figure(figsize=(10, 8))

plt.plot(np.diag(svd_grey[1]))
plt.yscale('log')


def frobenius_norm(img):
    return np.linalg.norm(img)

print("Frobenius norm of original image: ", frobenius_norm(grey))

print("Error at k=5: ", frobenius_norm(grey-create_img(*truncate(*svd_grey, 5)))/frobenius_norm(grey))
print("Error at k=15: ", frobenius_norm(grey-create_img(*truncate(*svd_grey, 15)))/frobenius_norm(grey))
print("Error at k=50: ", frobenius_norm(grey-create_img(*truncate(*svd_grey, 50)))/frobenius_norm(grey))
print("Error at k=200: ", frobenius_norm(grey-create_img(*truncate(*svd_grey, 200)))/frobenius_norm(grey))

sum_sigma=0
u,s,v=svd_grey
for i in range(50, min(grey.shape)):
    sum_sigma+=s[i]**2



print("Verifying numerically: ", frobenius_norm(grey-create_img(*truncate(*svd_grey, 50)))**2,"=", sum_sigma)


p,q=grey.shape

print("The original image stores ", p*q, " numbers.")
print("At k=5 we store ",5*(p+q+1), "numbers. The ratio of storage is ", 5*(p+q+1)/(p*q)*100,"%")
print("At k=15 we store ",15*(p+q+1), "numbers. The ratio of storage is ", 15*(p+q+1)/(p*q)*100,"%")
print("At k=50 we store ",50*(p+q+1), "numbers. The ratio of storage is ", 50*(p+q+1)/(p*q)*100,"%")
print("At k=200 we store ",200*(p+q+1), "numbers. The ratio of storage is ", 200*(p+q+1)/(p*q)*100,"%")



def retained_frobenius(s, k):
    return np.sum(s[:k]**2)/np.sum(s**2)

for i in range(1,1000,1):
    if (retained_frobenius(s, i)-0.90)<0.1 and (retained_frobenius(s, i)-0.90)>0:
        print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, i)*100:.2f}%")
    elif (retained_frobenius(s, i)-0.95)<0.1 and (retained_frobenius(s, i)-0.95)>0:
        print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, i)*100:.2f}%")
    elif (retained_frobenius(s, i)-0.99)<0.01 and (retained_frobenius(s, i)-0.99)>0:
        print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, i)*100:.2f}%")
print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, i)*100:.2f}%")



#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 1)*100:.2f}%")
#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 2)*100:.2f}%")
#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 7)*100:.2f}%")
#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 8)*100:.2f}%")
#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 91)*100:.2f}%")
#print(f"Retained Frobenius norm at k={i}: {retained_frobenius(s, 92)*100:.2f}%")




plt.show()
