
KNN

1.计算所有类别与未知类别的距离,并按升序排列.
2.找到k个最靠近未知类别的类型（少数服从多数），来决定未知类别的类别。


目录

1 算法概述
1.1 算法特点：
    简单地说，k-近邻算法采用测量不同特征值之间的距离方法进行分类。

    优点：精度高、对异常值不敏感、无数据输入假定
    缺点：计算复杂度高、空间复杂度高
    适用数据范围：数值型和标称型

1.2 工作原理
    存在一个训练样本集，并且每个样本都存在标签（有监督学习）。输入没有标签的新样本数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取出与样本集中特征最相似的数据（最近邻）的分类标签。一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，而且k通常不大于20。最后选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

1.3 实例解释
    以电影分类为例子，使用k-近邻算法分类爱情片和动作片。有人曾经统计过很多电影的打斗镜头和接吻镜头，下图显示了6部电影的打斗和接吻镜头数。 假如有一部未看过的电影，如何确定它是爱情片还是动作片呢？ 
      diffMat = tile(inX, (dataSetSize, 1)) - dataset
      sqDiffMat = diffMat ** 2
      sqDistance = sqDiffMat.sum(axis=1)
      distance = sqDistance ** 0.5
      sortedDistIndicies = distance.argsort() #升序排序后值的索引


2 代码实现
2.1 k-近邻简单分类的应用


2.2 在约会网站上使用k-近邻算法
  从文件中读入训练数据，并存储为矩阵：
    numberOfLines = len(arrayOlines)   #获取 n=样本的行数
    returnMat = zeros((numberOfLines,3))   #创建一个2维矩阵用于存放训练样本数据，一共有n行，每一行存放3个数据
    returnMat[index,:] = listFromLine[0:3]，把list分割好的数据放至数据集就是放到第几行
    classLabelVector.append(int(listFromLine[-1])),该样本对应的标签放至标签集，顺序与样本集对应,-1表示最后一个。

  训练数据归一化：
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]

2.3 手写识别系统实例
3 应用 scikit-learn 库实现k近邻算法
 knn = neighbors.KNeighborsClassifier() 
knn.fit(iris.data, iris.target)
score = knn.score(iris.data, iris.target)
#预测
predict = knn.predict([[0.1,0.2,0.3,0.4]])
#预测，返回概率数组
predict2 = knn.predict_proba([[0.1,0.2,0.3,0.4]])

print(predict)
print(iris.target_names[predict])

b = np.array([[1, 2], [3, 4]])
np.tile(b, 2)
array([[1, 2, 1, 2],
        [3, 4, 3, 4]])
np.tile(b, (2, 1))
array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])