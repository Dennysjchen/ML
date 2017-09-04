# --coding:utf-8 --
# --author:dennychen
# --date:2017.07.10
#  kNN demo
#  1.计算所有类别与未知类别的距离,并按升序排列.
#  2.找到k个最靠近未知类别的类型（少数服从多数），来决定未知类别的类别。
#  d = 」(xA0 - xB0 )2 + (xA{ - xBt )2
#  main function:tile
    #b = np.array([[1, 2], [3, 4]])
	# np.tile(b, 2)
	# array([[1, 2, 1, 2],
	#         [3, 4, 3, 4]])
	# np.tile(b, (2, 1))
	# array([[1, 2],
	#        [3, 4],
	#        [1, 2],
	#        [3, 4]])
	
import numpy as np
import operator
from sklearn.datasets import load_iris  
from sklearn import neighbors  
import sklearn

def createDataSet():
	dtrain =np.array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.0]])
	dtest  =['A','A','B','B']
	print(dtrain,dtest)
	return dtrain,dtest

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]        # 得到数组的行数，即知道有几个数据集
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 样本与训练集的差值矩阵  #tile(a,(3,2)) 构造3X2个copy
    sqDiffMat = diffMat ** 2               # 各元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)    # 每一行元素求和
    distances = sqDistances ** 0.5         # 开方，得到距离
    sortedDistances = distances.argsort()  # 按照distances中元素进行升序排列后得到相应下表的列表
    print("distances",distances)
    print("sortedDistances",sortedDistances)
    classCount = {}                        # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        print("sortedDistances",sortedDistances[i])
        print("voteIlabel",voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按classCount字典的value（类别出现次数）降序排列
    return sortedClassCount[0][0]

def testsklearnKNN():
	iris = load_iris()
	#print(iris)

	knn = neighbors.KNeighborsClassifier()  
	#训练数据集  
	knn.fit(iris.data, iris.target)
	print(iris.target)
	#训练准确率
	score = knn.score(iris.data, iris.target)

	#预测
	predict = knn.predict([[0.1,0.2,0.3,0.4]])
	#预测，返回概率数组
	predict2 = knn.predict_proba([[0.1,0.2,0.3,0.4]])
	 
	print(predict)
	print(iris.target[predict])


testsklearnKNN()

# inX =[0.1,0.1]
# dataSet,labels =createDataSet()
# labels = classify0(inX,dataSet,labels,3)
# print(inX," labels is :",labels)

