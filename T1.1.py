import cv2
import numpy as np
import math

def custom_canny_edge_detection(image, low_threshold, high_threshold, kernel_size=5, sigma=1.4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    Gx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    
    G=np.sqrt(Gx**2+Gy**2)
    angle=np.arctan2(Gy,Gx)*(180/np.pi)
    matrix=np.zeros_like(G)
    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):
            if (-22.5<=angle[i,j]<22.5)or(157.5<angle[i,j]<=180)or(-180<angle[i,j]<-157.5):
                N1=G[i+1,j]
                N2=G[i-1,j]
            elif (22.5 <= angle[i, j]<67.5) or (-157.5<angle[i,j]<=-112.5):
                N1=G[i+1,j-1]
                N2=G[i-1,j+1]
            elif (112.5 <= angle[i, j]<157.5) or (-67.5<angle[i,j]<=-22.5):
                N1=G[i-1,j-1]
                N2=G[i+1,j+1]
            elif (67.5<=angle[i,j]<112.5)or(-112.5<angle[i,j]<=-67.5):
                N1=G[i,j-1]
                N2=G[i,j+1]
            
            if G[i,j] >= N1 and G[i, j] >= N2:
                matrix[i, j] = G[i, j]
                
    strong_edges=matrix>high_threshold
    weak_edges=(matrix<high_threshold) &( matrix>low_threshold)
    
    
    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            if weak_edges[i, j]:
                if (strong_edges[i+1, j+1] or strong_edges[i+1, j] or strong_edges[i+1, j-1] or
                strong_edges[i, j+1] or strong_edges[i, j-1] or
                strong_edges[i-1, j+1] or strong_edges[i-1, j] or strong_edges[i-1, j-1]):
                    strong_edges[i, j] = True
                    
    return strong_edges.astype(np.uint8) * 255


image = cv2.imread('table.png')

low_threshold = 89
high_threshold = 180

point=[]


edges = custom_canny_edge_detection(image, low_threshold, high_threshold)
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if edges[i, j] == 255:
            point.append((i, j))  
print(point)




start = point[4]
end = point[-1]




m = (end[1]-start[1])/(end[0]-start[0])
b = end[1]-m*end[0]

x_top = 0
x_bottom = image.shape[1]-1

y_top = int(m*x_top+b)
y_bottom = int(m*x_bottom+b)

cv2.line(image,(y_top,x_top),( y_bottom,x_bottom),(0, 0, 255),thickness=4)



cv2.imshow('Image with Marked Edges', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('hello',edges)
cv2.waitKey()
cv2.destroyAllWindows()
            
    
    
    

