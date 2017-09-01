# --coding:utf-8 --
# --author:dennychen
# --date:2017.07.10
#  kNN demo
#  
import numpy as np
import operator

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
    classCount = {}                        # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #按classCount字典的value（类别出现次数）降序排列
    return sortedClassCount[0][0]



inX =[0.1,0.1]
dataSet,labels =createDataSet()
labels = classify0(inX,dataSet,labels,3)
print(inX," labels is :",labels)

