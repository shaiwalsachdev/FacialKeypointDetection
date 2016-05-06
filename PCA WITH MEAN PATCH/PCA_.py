import numpy as np 
import pandas as pd
from sklearn import datasets, decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from numpy import linalg
import scipy
from scipy import misc
from matplotlib import pyplot as plt 
from numpy.linalg import pinv
from numpy import matmul
import csv
import math


#cleanedData.csv is the input file  given by Kaggle

numberofImages = 5049
variance = 0.95
sizeofrow = 96
sizeofcolumn  = 96 
numberofpixels = sizeofrow * sizeofcolumn


def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r



def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

   


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
   # print r 
  #  print c
    m = np.mean(A,1)
    A = A - np.transpose(np.tile(m, (c,1)))
    B = np.dot(np.transpose(A), A)
    [d,v] = linalg.eigh(B)
    
  # print d.shape
   # print v.shape
    # v is in descending sorted order
    sort_perm = d.argsort()
    sort_perm = sort_perm[::-1]
    d.sort()     # <-- This sorts the list in place.
    d = d[::-1]
    v = v[sort_perm]
    
    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
 	
 	#print W2.shape
    #LL = d[0:-1]

    #W = W2[:,0:-1]      #omit last column, which is the nullspace
    #print W.shape
    #print LL.shape

    LL = d[0:]

    W = W2[:,0:]    
    return W, LL, m




def float2int8(A):
    # function im = float2int8(A)
    # convert an float array image into grayscale image with contrast from 0-255
    amin = np.amin(A)
    amax = np.amax(A)
    [r,c] = A.shape
    im = ((A - amin) / (amax - amin)) * 255

    return im


#Forfeatures
def normfeatures(A):
    # function im = float2int8(A)
    # convert an float array image into grayscale image with contrast from 0-255
    amin = np.amin(A)
    amax = np.amax(A)
    [r,c] = A.shape
    im = ((A - amin) / (amax - amin)) * 95

    return im




#Reading using pandas
datafromfile = pd.read_csv("cleanedData.csv");



image = datafromfile["Image"]

#print image.shape
coordinate = datafromfile.iloc[:,0:30]
#print coordinate


features = coordinate.as_matrix()


#print m1
#print W1

numpyMatrix = image.as_matrix() 


newnumpyMatrix = [] 
for i in range(numberofImages):
	single_image = numpyMatrix[i].split(" ")
	for j in range(numberofpixels):
		single_image[j] = int(single_image[j])
	newnumpyMatrix.append(single_image)
	if i == numberofImages-1:
		break


matrix = np.array(newnumpyMatrix)

X = matrix
matrix = matrix.transpose()
#print matrix

#W eigenvector
#LL eigenvalues
#m mean

[W,LL,m] = myPCA(matrix)

W = W.transpose()
W = float2int8(W)
print LL.shape
print W.shape


"""
#Normalize
mi = 1000000000000000000000
ma = -100000000000000000000

for i in range(150):
	for j in range(96*96):
		if (W[i][j] > ma) :
			ma = W[i][j]

		if (W[i][j] < mi) :
			mi = W[i][j]

print mi
print ma

for i in range(150):
	for j in range(96*96):
		x = ((W[i][j] - mi) / (ma - mi)) * 255
		W[i][j] = x;

"""


sumofeigen = 0
for i in range(numberofImages):
    sumofeigen = sumofeigen + LL[i]




cumulativevar = []
xcordinate = [] 
curr = 0
flag = 0
face = 0
for i in range(numberofImages):
	xcordinate.append(i)
	curr  = curr + LL[i]
	cumulativevar.append(curr/sumofeigen)
	if cumulativevar[i] > variance:
		if flag == 0:
			face = i
			flag = 1
		


#plt.plot(xcordinate, cumulativevar, linewidth=2.0)
#plt.show()



#print face 



#print W.shape

for ii in range(face):
	st = "face" + str(ii) + ".jpg" 
	st = "eiface1/" + st
	a = np.zeros(shape=(sizeofrow,sizeofcolumn),dtype = np.uint8)
	c = 0
	for i in range(sizeofrow):
		d = []
		for j in range(sizeofcolumn):
			d.append(W[ii][c])
			c = c + 1
		a[i] = d
	misc.imsave(st, a)




#Find the phi matrix

pc = W[0:face,:]
#print pc.shape

X = X[0:face,:]
#print X.shape

Xinv = pinv(X)
#print Xinv.shape
#for ii in range(face):

phi = matmul(pc,Xinv)
#print phi.shape

#print phi
#phi = phi**2
#print phi

#NOw Trimmed X
newX = features[0:face,:]
#print newX.shape
finalcoordinates = matmul(phi,newX)
#print finalcoordinates.shape
#print finalcoordinates

transfinalcoordinates = finalcoordinates.transpose()
#print a

#plt.imshow(a,cmap = cm.Greys_r)
#plt.show()
#print W.shape
#print LL.shape


#Now We will write the final co-ordinates and image to the transtraining.csv


finaldf = datafromfile.iloc[:face,:]
#print finaldf




list_str = []
for i in range(face):
	s1 = ''
	for j in range(numberofpixels):
		s1 = s1 + str(int(math.ceil(pc[i][j] - 0.5))) + " "
	s1 = s1[:-1]
	list_str.append(s1)


#print pc
#print list_str
#print finaldf['Image'].shape




finalcsv = pd.DataFrame({'left_eye_center_x':finalcoordinates[:,0],'left_eye_center_y':finalcoordinates[:,1],'right_eye_center_x':finalcoordinates[:,2],'right_eye_center_y':finalcoordinates[:,3],'left_eye_inner_corner_x':finalcoordinates[:,4],
	'left_eye_inner_corner_y':finalcoordinates[:,5],'left_eye_outer_corner_x':finalcoordinates[:,6],'left_eye_outer_corner_y':finalcoordinates[:,7],'right_eye_inner_corner_x':finalcoordinates[:,8],
	'right_eye_inner_corner_y':finalcoordinates[:,9],'right_eye_outer_corner_x':finalcoordinates[:,10],'right_eye_outer_corner_y':finalcoordinates[:,11],'left_eyebrow_inner_end_x':finalcoordinates[:,12],
	'left_eyebrow_inner_end_y':finalcoordinates[:,13],'left_eyebrow_outer_end_x':finalcoordinates[:,14],'left_eyebrow_outer_end_y':finalcoordinates[:,15],'right_eyebrow_inner_end_x':finalcoordinates[:,16],
	'right_eyebrow_inner_end_y':finalcoordinates[:,17],'right_eyebrow_outer_end_x':finalcoordinates[:,18],'right_eyebrow_outer_end_y':finalcoordinates[:,19],'nose_tip_x':finalcoordinates[:,20],'nose_tip_y':finalcoordinates[:,21],
	'mouth_left_corner_x':finalcoordinates[:,22],'mouth_left_corner_y':finalcoordinates[:,23],'mouth_right_corner_x':finalcoordinates[:,24],'mouth_right_corner_y':finalcoordinates[:,25],'mouth_center_top_lip_x':finalcoordinates[:,26],
	'mouth_center_top_lip_y':finalcoordinates[:,27],'mouth_center_bottom_lip_x':finalcoordinates[:,28],'mouth_center_bottom_lip_y':finalcoordinates[:,29]})

finalcsv['Image'] = list_str

"""
left_eye_center_x
left_eye_center_y
right_eye_center_x
right_eye_center_y
left_eye_inner_corner_x
left_eye_inner_corner_y
left_eye_outer_corner_x
left_eye_outer_corner_y
right_eye_inner_corner_x
right_eye_inner_corner_y
right_eye_outer_corner_x
right_eye_outer_corner_y
left_eyebrow_inner_end_x
left_eyebrow_inner_end_y
left_eyebrow_outer_end_x
left_eyebrow_outer_end_y
right_eyebrow_inner_end_x
right_eyebrow_inner_end_y
right_eyebrow_outer_end_x
right_eyebrow_outer_end_y
nose_tip_x
nose_tip_y
mouth_left_corner_x
mouth_left_corner_y
mouth_right_corner_x
mouth_right_corner_y
mouth_center_top_lip_x
mouth_center_top_lip_y
mouth_center_bottom_lip_x
mouth_center_bottom_lip_y
Image
#transfinalcoordinates = finalcoordinates.transpose()

#Variables
left_eye_center_x = []
left_eye_center_y = []

right_eye_center_x = []
right_eye_center_y = []



"""


#for i in range(face):
	#left_eye_center_x.append(transfinalcoordinates[0][i])

#print left_eye_center_x
#finaldf['Image'] = list_str
#finaldf['left_eye_center_x'] = left_eye_center_x



finalcsv.to_csv('transtraining1.csv',index = False)

plt.plot(xcordinate, cumulativevar, linewidth=2.0)
plt.show()


