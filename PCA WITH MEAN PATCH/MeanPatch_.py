import numpy as np 
import pandas as pd
from sklearn import datasets, decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from numpy import linalg
from numpy import std
import scipy
from scipy import misc
from  scipy import stats
from matplotlib import pyplot as plt 
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error
import csv
import math
from sklearn import metrics
search_size = 5
patch_size = 3
numberofpixels = 9216

def cov(A,B):
	covariance = 0
	size = patch_size
	a1 = np.mean(A)
	a2 = np.mean(B)

	for i in range(size):
		for j in range(size):
			covariance = covariance + (A[i][j] - a1)*(B[i][j] - a2)

	return covariance/(size * size)


def corr2_coeff(A,B):
	#print np.std(A)
	#print np.std(B)
	if (np.std(A) == 0 or np.std(B) == 0):
		return cov(A,B)
	return cov(A,B)/(np.std(A)*np.std(B))



#Reading using pandas
datafromfile = pd.read_csv("transtraining1.csv");

data = datafromfile.iloc[:,0:30]
finalcoordinates = data.as_matrix()
transfinalcoordinates = finalcoordinates.transpose()
face = len(finalcoordinates);


#First 30 means
means = []
for i in range(30):
	summ = 0
	for j in range(face):
		summ = summ + transfinalcoordinates[i][j]

	summ = summ/face

	means.append(summ)


#print means



#Now 30  Mean Patch

#Read Image



im = datafromfile['Image'].as_matrix()


val = im

images = np.zeros((face,96,96), np.uint8)

for i in range(face):
    val[i] = im[i].split(' ')



#Each row has a 2D matrix
cn = 0

for i in range(face) :
    cn = 0
    for j in range(96) :
        for k in range(96):
            images[i][j][k] = int(val[i][cn])
            cn = cn + 1






#print images

#Find Mean Patch
#print face
patch = np.zeros((15,patch_size,patch_size),int)



for i in range(0,30,2):
	m = (int)(i/2)
	for j in range(face):
		x = finalcoordinates[j][i]
		y = finalcoordinates[j][i+1]
		leftcorner_x = (int)(x - (int)(patch_size/2))
		leftcorner_y = (int)(y- (int)(patch_size/2))


		for k in range(patch_size):
			for l in range(patch_size):
				p_x = leftcorner_x + k
				p_y = leftcorner_y + l 
				
				if(p_x >=0 and p_x <=95 and p_y>=0 and p_y<=95):
					patch[m][k][l] = (patch[m][k][l] + images[j][p_x][p_y])
				




for i in range(15):
	for k in range(patch_size):
			for l in range(patch_size):
				patch[i][k][l] = (int)(patch[i][k][l]/face)



#print patch



#Training Done


#Reading Test_Size of data
n_test = 100
testdata = pd.read_csv("cleanedData.csv");

im1 = testdata['Image'].as_matrix()
testdata = testdata.iloc[5639:,:30]
val1 = im1

test_images = np.zeros((n_test,96,96))

for i in range(n_test):
    val1[i] = im1[i].split(' ')


real_coordinates = testdata.iloc[:n_test,0:30]

#Each row has a 2D matrix
cn1 = 0

for i in range(n_test) :
    cn1 = 0
    for j in range(96) :
        for k in range(96):
            test_images[i][j][k] = int(val1[i][cn1])
            cn1 = cn1 + 1

#print test_images



result = np.zeros((n_test,30),int)


for i in range(n_test):
	for j in range(0,30,2):
		mean_x = means[j]
		mean_y = means[j+1]
		left_new_corner_x = (int)(mean_x - (int)(search_size/2))
		left_new_corner_y = (int)(mean_y - (int)(search_size/2))


		th = 0
		#flag = 0
		ans_x = mean_x
		ans_y = mean_y

		for k in range(search_size):
			for l in range(search_size):
				point_x = left_new_corner_x + k
				point_y = left_new_corner_y + l

				patch_corner_x = (int)(point_x - (int)(patch_size/2))
				patch_corner_y = (int)(point_y - (int)(patch_size/2))

				patchi = np.zeros((patch_size,patch_size),int)
				
				for m in range(patch_size):
					for n in range(patch_size):
						ppoint_x = patch_corner_x + m
						ppoint_y = patch_corner_y + n 
						if(ppoint_x >=0 and ppoint_x <=95 and ppoint_y>=0 and ppoint_y<=95):
							patchi[m][n] = test_images[i][ppoint_x][ppoint_y]

				#print patchi
				#print patch[(int)(j/2)]
				corr = corr2_coeff(patchi,patch[(int)(j/2)])
				#	corr = 0
				#print corr
				if corr > th or corr == th:
					th = corr
					#flag = 1
					ans_x = point_x
					ans_y = point_y

		result[i][j] = ans_x
		result[i][j+1] = ans_y


#print result.shape

finalcsv = pd.DataFrame({'left_eye_center_x':result[:,0],'left_eye_center_y':result[:,1],'right_eye_center_x':result[:,2],'right_eye_center_y':result[:,3],'left_eye_inner_corner_x':result[:,4],
	'left_eye_inner_corner_y':result[:,5],'left_eye_outer_corner_x':result[:,6],'left_eye_outer_corner_y':result[:,7],'right_eye_inner_corner_x':result[:,8],
	'right_eye_inner_corner_y':result[:,9],'right_eye_outer_corner_x':result[:,10],'right_eye_outer_corner_y':result[:,11],'left_eyebrow_inner_end_x':result[:,12],
	'left_eyebrow_inner_end_y':result[:,13],'left_eyebrow_outer_end_x':result[:,14],'left_eyebrow_outer_end_y':result[:,15],'right_eyebrow_inner_end_x':result[:,16],
	'right_eyebrow_inner_end_y':result[:,17],'right_eyebrow_outer_end_x':result[:,18],'right_eyebrow_outer_end_y':result[:,19],'nose_tip_x':result[:,20],'nose_tip_y':result[:,21],
	'mouth_left_corner_x':result[:,22],'mouth_left_corner_y':result[:,23],'mouth_right_corner_x':result[:,24],'mouth_right_corner_y':result[:,25],'mouth_center_top_lip_x':result[:,26],
	'mouth_center_top_lip_y':result[:,27],'mouth_center_bottom_lip_x':result[:,28],'mouth_center_bottom_lip_y':result[:,29]})

finalcsv.to_csv('test_result.csv',index = False)




newimages = test_images
for i in range(n_test):
	for j in range(0,30,2):
		x = result[i][j]
		y = result[i][j+1]
		newimages[i][x][y] = 0

	#plt.imshow(newimages[i],cmap = cm.Greys_r)
	#plt.show()	

print "The RMSE obtained is =  "
print np.sqrt(metrics.mean_squared_error(real_coordinates,result))

#print real_coordinates.shape
#	print result.shape