import numpy as np
from sklearn import preprocessing

def paint_rectangle(rectangle_list, width, height, title=""):
	'''
	Changes in data before and after 2D anonymous rendering
	1. It is possible that the predicted values and eigenvalues are uncertain data that have been processed anonymously
	2. The predicted value is accurate, and the eigenvalue is anonymized
	Scale the image according to the canvas size to make the image look better
	(the origin is in the top left corner of the canvas)
	
	&Paras:
	Rectangle_list: the rectangle coordinate list ([top left x, top left y, bottom right x, bottom right y])
	Width: width of the canvas
	Height: canvas height
	'''
	import tkinter as TK
	tk = TK.Tk()
	tk.title = title
	canvas = TK.Canvas(tk, width=width, height=height, bg="White")
	canvas.pack()
	for r in rectangle_list:
		canvas.create_rectangle(r[0], int(height-r[1]), r[2], int(height-r[3]))
	tk.mainloop()


def calculate_uncertainty(masked_result):
	from sklearn import preprocessing
	np_result = np.array(masked_result)
	matrix = np.zeros((len(masked_result), len(masked_result[0])*2+1))
	matrix[:, -1] = 0
	for col in range(len(masked_result[0])):
		for row in range(len(masked_result)):
			values = masked_result[row][col].split("~")
			if len(values) < 2:
				matrix[row, 2*col:2*col+2] = [values[0], values[0]]
			else:
				matrix[row, 2*col:2*col+2] = values
	value_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	for col in range(len(masked_result[0])):
		value_scaler.fit(np.vstack((matrix[:, 2*col].reshape(-1,1), matrix[:, 2*col+1].reshape(-1,1))))
		matrix[:, 2*col] = value_scaler.transform(matrix[:, 2*col].reshape(-1,1)).flatten()
		matrix[:, 2*col+1] = value_scaler.transform(matrix[:, 2*col+1].reshape(-1,1)).flatten()
	for col in range(len(masked_result[0])):
		delta = np.abs(matrix[:, 2*col+1]-matrix[:, 2*col])
		matrix[:, -1] = matrix[:, -1] + delta
	return matrix[:, -1].flatten()



def rectangel_sampling_from_masked_data(masked_result, by_size=False, ratio=0.8):
	'''
	Paras:
		by_size: If true, then calculate global uncertainty with multiplication, else by addition.
		ratio: remainder-ratio
	'''
	from sklearn import preprocessing
	np_result = np.array(masked_result)
	matrix = np.zeros((len(masked_result), len(masked_result[0])*2+1))
	matrix[:, -1] = 0
	for col in range(len(masked_result[0])):
		for row in range(len(masked_result)):
			values = masked_result[row][col].split("~")
			if len(values) < 2:
				matrix[row, 2*col:2*col+2] = [values[0], values[0]]
			else:
				matrix[row, 2*col:2*col+2] = values
	value_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	for col in range(len(masked_result[0])):
		value_scaler.fit(np.vstack((matrix[:, 2*col].reshape(-1,1), matrix[:, 2*col+1].reshape(-1,1))))
		matrix[:, 2*col] = value_scaler.transform(matrix[:, 2*col].reshape(-1,1)).flatten()
		matrix[:, 2*col+1] = value_scaler.transform(matrix[:, 2*col+1].reshape(-1,1)).flatten()
	for col in range(len(masked_result[0])):
		delta = np.abs(matrix[:, 2*col+1]-matrix[:, 2*col])
		matrix[:, -1] = matrix[:, -1] + delta
	left_ratio = ratio
	if not by_size:
		left_num = max(100, int(matrix.shape[0]*left_ratio))
		real_left_num = min(np_result.shape[0], left_num)
		np_result = np_result[matrix[:, -1].argsort()]
		print(real_left_num,"/",len(masked_result)," are left after sampling!")
		return np_result[:left_num]
	else:
		size_vec = np.unique(matrix[:, -1])
		left_num = int(len(size_vec) * left_ratio)
		thresh_hold = np.sort(size_vec)[left_num]
		result = np_result[matrix[:,-1]<=thresh_hold]
		real_left_num =len(result)
		print(real_left_num,"/",len(masked_result)," are left after sampling!")
		return result


def paint_rectangle_for_masked_2d_data(masked_data, title=""):
	'''The 2-dimensional hypercube is drawn according to the output of the anonymous algorithm'''
	CANVAS_WIDTH = 1000
	CANVAS_HEIGHT = 1000
	x_scaler = preprocessing.MinMaxScaler(feature_range=(0, CANVAS_WIDTH))
	y_scaler = preprocessing.MinMaxScaler(feature_range=(0, CANVAS_HEIGHT))
	def make_rectangle(x, y):
		if len(x.split("~"))>1:
			x1, x2 = map(float, x.split("~"))
		else:
			x1, x2 = float(x), float(x)
		if len(y.split("~")) > 1:
			y1, y2 = map(float, y.split("~"))
		else:
			y1, y2 = float(y), float(y)
		return [x1, y1, x2, y2]
	rectangles = map(
		lambda xy:(make_rectangle(xy[0], xy[1])), 
		masked_data
		)
	np_rectangles = np.array(list(rectangles))
	x_scaler.fit(np.vstack((np_rectangles[:, 0].reshape(-1,1), np_rectangles[:, 2].reshape(-1,1))))
	y_scaler.fit(np.vstack((np_rectangles[:, 1].reshape(-1,1), np_rectangles[:, 3].reshape(-1,1))))
	np_rectangles[:, 0] = x_scaler.transform(np_rectangles[:, 0].reshape(-1,1)).flatten()
	np_rectangles[:, 2] = x_scaler.transform(np_rectangles[:, 2].reshape(-1,1)).flatten()
	np_rectangles[:, 1] = y_scaler.transform(np_rectangles[:, 1].reshape(-1,1)).flatten()
	np_rectangles[:, 3] = y_scaler.transform(np_rectangles[:, 3].reshape(-1,1)).flatten()
	paint_rectangle(np_rectangles, CANVAS_WIDTH, CANVAS_HEIGHT)


if __name__ == '__main__':
	'''
		module test
	'''
	from mondrian.masker import mask
	np.random.seed(2018)
	l = list(range(1, 500, 10))
	l.reverse()
	ORIGINAL_DATA_X = np.array(l, dtype=np.float64)
	MAP_FUNC = lambda x: 0.5 * x + 5 + 0.005 * x**2 + 0.001 * x + 0.01 * np.log(x+1)
	ORIGINAL_DATA_Y = np.array([MAP_FUNC(num) for num in ORIGINAL_DATA_X]) + 3 * np.random.randn(len(ORIGINAL_DATA_X))
	TEST_DATA_X = np.array(list(range(0, 500, 10)))
	TEST_DATA_Y = np.array([MAP_FUNC(num) for num in TEST_DATA_X]) + 3 * np.random.randn(len(TEST_DATA_X))
	xy_data = [[ORIGINAL_DATA_X[i], ORIGINAL_DATA_Y[i]] for i, value in enumerate(ORIGINAL_DATA_X)]
	result, eval_result = mask(xy_data, 3, 2)
	sampled_masked_data = rectangel_sampling_from_masked_data(result, by_size=True)
	paint_rectangle_for_masked_2d_data(sampled_masked_data)