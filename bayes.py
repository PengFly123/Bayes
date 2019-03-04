import os
import re
from numpy import *
from math import *

def textParse(doc):             # 将文本解析成单词列表
    reTxt = re.compile(r'\W')
    wordList = [word.lower() for word in reTxt.split(doc) if len(word)>2]       # 将长度大于2的单词存入列表
    return wordList


def words2Vec(vocaList, wordList):
    retVec = zeros(len(vocaList))
    for word in wordList:
        if word in vocaList:
            retVec[vocaList.index(word)] += 1
    return retVec


def trainNB0(dataSet, labelSet):
    p0Num = ones(len(dataSet[0]))
    p1Num = ones(len(dataSet[0]))
    p0Sum = 2.0
    p1Sum = 2.0
    p0 = 0
    sampNum = len(dataSet)
    for i in range(sampNum):
        if labelSet[i] == 0:
            p0Num += dataSet[i]
            p0Sum += sum(dataSet[i])
            p0 += 1
        else:
            p1Num += dataSet[i]
            p1Sum += sum(dataSet[i])
    p0V = [log(x/p0Sum) for x in p0Num]
    p1V = [log(x/p1Sum) for x in p1Num]
    return p0V, p1V, float(p0/sampNum)


def classifyNB(testdata, p0V, p1V, p0):
    P0 = sum(testdata*p0V) + log(p0)
    P1 = sum(testdata*p1V) + log(1.0-p0)
    if P0>P1:
        return 0
    else:
        return 1


def main():
    className = os.listdir('email')
    class0List = os.listdir(os.path.join('email', className[0]))
    class1List = os.listdir(os.path.join('email', className[1]))
    class0Nums = len(class0List)            # 由于两个类别的训练样本数量相等，取一个就可以
    classLabelAll = []
    vocaList = []
    wordList = []
    for i in range(class0Nums):
        words = textParse(open(os.path.join(os.path.join('email', className[0]), class0List[i]), 'r').read())
        wordList.append(words)
        vocaList.extend(words)
        classLabelAll.append(0)
        words = textParse(open(os.path.join(os.path.join('email', className[1]), class0List[i]), 'r').read())
        wordList.append(words)
        vocaList.extend(words)
        classLabelAll.append(1)
    vocaList = set(vocaList)            # 所有单词不重复集合
    vocaList = list(vocaList)
    non_trainSet = []
    testSet = []
    testClass = []
    while len(non_trainSet) < 10:
        randIndex = int(random.uniform(0, 50))
        if randIndex not in non_trainSet:               # 防止生成相同的随机数
            testSet.append(wordList[randIndex])
            non_trainSet.append(randIndex)
            testClass.append(classLabelAll[randIndex])
    trainSet = []
    trainLabel = []
    for i in range(50):
        if i not in  non_trainSet:
            trainSet.append(words2Vec(vocaList, wordList[i]))
            trainLabel.append(classLabelAll[i])
    p0V, p1V, p0 = trainNB0(array(trainSet), array(trainLabel))
    error = 0
    for i in range(len(testSet)):
        data = words2Vec(vocaList, testSet[i])
        if classifyNB(array(data), p0V, p1V, p0) != testClass[i]:
            error += 1
    print('错误率是: ' , error/len(testSet))


if __name__ == '__main__':
    main()