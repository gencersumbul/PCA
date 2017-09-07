import numpy as np
import csv

class PCA:
	def __init__(self,d):
		self.data = d
		self.final_data = self.run_PCA(self.data)

	def get_result(self):
		return self.final_data
		
	def read_data(self,file_name):
		data = []
		with open(file_name, 'rt') as data_file:
			reader = csv.reader(data_file)
			for row in reader:
				data.append(row)
		return np.array(data,dtype=float)

	def save_data(self,new_data,file_name):
		with open(file_name, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',',quotechar=' ', quoting=csv.QUOTE_MINIMAL)
			for i in range(len(new_data)):
				row=[]
				for j in range(len(new_data[0])):					
					row.append(new_data[i][j])
				writer.writerow(row)

	def calc_covariance(self,data):
		column = len(data[0])
		covariance = np.zeros((column,column))
		covariance = np.cov(np.array(data).astype(np.float),rowvar=0)

		return covariance

	def find_eigens(self,covariance):
		eigen_values = np.zeros((len(covariance[0])))
		eigen_vectors = np.zeros((len(covariance),len(covariance[0])))
		eigenvalues, eigenvectors = np.linalg.eig(covariance)
		eigen_vectors = eigenvectors
		eigen_values = eigenvalues

		return eigen_values,eigen_vectors

	def order_eigens(self,eigen_values,eigen_vectors,length):
		ordered_indices = np.zeros((len(eigen_values)))
		ordered_eigen_vectors = np.zeros((len(eigen_vectors),length))
		ordered_indices=np.argsort(eigen_values)
		for i in range(len(eigen_vectors)):
			for k in range(length):
				t = len(eigen_vectors[0])-k-1
				ordered_eigen_vectors[i][k]=eigen_vectors[i][ordered_indices[t]]
		return ordered_eigen_vectors

	def find_final_data(self,data,eigen_vectors,reduced_feature_size):
		new_data=np.zeros((len(data),reduced_feature_size))
		new_data=np.transpose(np.dot(np.transpose(eigen_vectors),np.transpose(data)))

		return new_data

	def run_PCA(self,data):
		#data = read_data('H.csv')
		reduced_feature_size=2
		covariance = self.calc_covariance(data)
		eigen_values, eigen_vectors = self.find_eigens(covariance)
		ordered_eigen_vectors=self.order_eigens(eigen_values,eigen_vectors,2)
		new_data=self.find_final_data(data,ordered_eigen_vectors,reduced_feature_size)
		return new_data
		#save_data(new_data,'H-PCA.csv')