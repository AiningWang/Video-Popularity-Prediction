# -*- coding: utf-8 -*-

"""
    Predict online video popularity according time series
    Implement SH model, ML model, Pop model, Altman model...
    Author: Aining Wang
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from loadInfo import AllVideoInfo

"""
    @refer_d: the popularity before refer_d is known
    @target_d: the popularity before target_d need to be predict
"""

REFER_D_MIN = 7
REFER_D_MAX = 22
TARGET_D = 23
FILE = "new_upload_sampling10_14.txt"



class Data:
    """
        load and store data
    """
    def __init__(self):
        self.trainx = []
        self.trainy = []
        self.trainxCDF = []
        self.testx = []
        self.testy = []
        self.testxCDF = []
        self.num_of_train = 0
        self.num_of_test = 0


    def load(self, file, refer_d, target_d):
        """
            the file is as follows:
            vid1 \t day1 \t day2 ... dayN \n
            vid2 ...
        """
        i = 0
        for line in file:
            i += 1
            series = map(int, line.strip("\n").split("\t")[1:])
            if i % 4 != 0:
                self.trainx.append(series[: refer_d])
                self.trainy.append(sum(series[: target_d]))
            else:
                self.testx.append(series[: refer_d])
                self.testy.append(sum(series[: target_d]))
        file.close()
        """
            if N(target_d) == 0, delete this video
        """
        k = len(self.testy)
        for i in range(k):
            if self.testy[k - i - 1] == 0:
                del self.testy[k - i - 1]
                del self.testx[k - i - 1]
        k = len(self.trainy)
        for i in range(k):
            if self.trainy[k - i - 1] == 0:
                del self.trainy[k - i - 1]
                del self.trainx[k - i - 1]
        self.num_of_test = len(self.testy)
        self.num_of_train = len(self.trainy)


    def load2(self, info_dict, refer_d, target_d):
        """
            load the data from class AllVideoInfo
        """
        i = 0
        for vid in info_dict:
            i += 1
            if i % 4 != 0:
                self.trainx.append(info_dict[vid].pop[: refer_d])
                self.trainy.append(sum(info_dict[vid].pop[: target_d]))
            else:
                self.testx.append(info_dict[vid].pop[: refer_d])
                self.testy.append(sum(info_dict[vid].pop[: target_d]))
        """
            if N(target_d) == 0, delete this video
        """
        k = len(self.testy)
        for i in range(k):
            if self.testy[k - i - 1] == 0:
                del self.testy[k - i - 1]
                del self.testx[k - i - 1]
        k = len(self.trainy)
        for i in range(k):
            if self.trainy[k - i - 1] == 0:
                del self.trainy[k - i - 1]
                del self.trainx[k - i - 1]
        self.num_of_test = len(self.testy)
        self.num_of_train = len(self.trainy)


    def turnCDF(self):
        self.testxCDF = []
        self.trainxCDF = []
        refer_d = len(self.testx[0])
        for i in range(self.num_of_test):
            x = []
            for j in range(refer_d):
                x.append(sum(self.testx[i][:j + 1]))
            self.testxCDF.append(x)
        for i in range(self.num_of_train):
            x = []
            for j in range(refer_d):
                x.append(sum(self.trainx[i][:j + 1]))
            self.trainxCDF.append(x)


    def addTrainData(self, x, y):
        if y != 0:
            self.trainx.append(x)
            self.trainy.append(y)
            self.num_of_train += 1


    def addTestData(self, x, y):
        if y != 0:
            self.testx.append(x)
            self.testy.append(y)
            self.num_of_test += 1




class ClassicPredictModel:
    """
        implement SH model, ML model
    """
    
    def __init__(self):
        self.mRSE = 0
        self.data = Data()
    
    
    def load(self, refer_d, target_d):
        self.data.load(open(FILE, "r"), refer_d, target_d)
    

    def SHModel(self):
        alpha = 0
        sum1 = 0
        sum2 = 0
        """
            calculate alpha
            According S-H model: alpha * N(refer_d) = N(target_d)
        """
        for i in range(self.data.num_of_train):
            sum1 += (sum(self.data.trainx[i]) + 0.0) / self.data.trainy[i]
            sum2 += ((sum(self.data.trainx[i]) + 0.0) / self.data.trainy[i]) ** 2
        alpha = sum1 / sum2
        """
            calculate mRSE
        """
        RSE = 0
        for i in range(self.data.num_of_test):
            RSE += ((alpha * sum(self.data.testx[i]) / self.data.testy[i]) - 1) ** 2
        self.mRSE = RSE / self.data.num_of_test


    def MLmRSE(self, alpha, x, y):
        return (np.dot(alpha, x) / y - 1)
    
    
    def MLModel(self):
        """
            According M-L model: alpha1*day1 + ... + alphan*dayn = N(target_d)
            get alpha by using leastsq
        """
        alpha = [0] * len(self.data.trainx[0])
        x     = np.transpose(np.array(self.data.trainx))
        y     = np.array(self.data.trainy)
        alpha = leastsq(self.MLmRSE, alpha, args = (x, y))[0]
        """
            calculate mRSE
        """
        RSE = 0
        for i in range(self.data.num_of_test):
            RSE += ((np.dot(alpha, self.data.testx[i]) / self.data.testy[i]) - 1) ** 2
        self.mRSE = RSE / self.data.num_of_test



class Pop:
    """
        > 10^4      : Very popular (VP)
        10^3 - 10^4 : Popular (P)
        10^2 - 10^3 : Not so popular (NSP)
        10^1 - 10^2 : Unpopular (U)
        < 10^1      : Very Unpopular (VU)
    """
    
    def __init__(self):
        self.VP = Data()
        self.P  = Data()
        self.NSP= Data()
        self.U  = Data()
        self.VU = Data()
        self.mRSE = 0


    def classification(self, refer_d, target_d):
        all = Data()
        all.load(open(FILE, "r"), refer_d, target_d)
        """
            Classify test data
        """
        for i in range(all.num_of_test):
            if sum(all.testx[i]) > 10 ** 4:
                self.VP.addTestData(all.testx[i], all.testy[i])
            elif sum(all.testx[i]) > 10 ** 3:
                self.P.addTestData(all.testx[i], all.testy[i])
            elif sum(all.testx[i]) > 10 ** 2:
                self.NSP.addTestData(all.testx[i], all.testy[i])
            elif sum(all.testx[i]) > 10 ** 1:
                self.U.addTestData(all.testx[i], all.testy[i])
            else:
                self.VU.addTestData(all.testx[i], all.testy[i])
        """
            Classify train data
        """
        for i in range(all.num_of_train):
            if sum(all.trainx[i]) > 10 ** 4:
                self.VP.addTrainData(all.trainx[i], all.trainy[i])
            elif sum(all.trainx[i]) > 10 ** 3:
                self.P.addTrainData(all.trainx[i], all.trainy[i])
            elif sum(all.trainx[i]) > 10 ** 2:
                self.NSP.addTrainData(all.trainx[i], all.trainy[i])
            elif sum(all.trainx[i]) > 10 ** 1:
                self.U.addTrainData(all.trainx[i], all.trainy[i])
            else:
                self.VU.addTrainData(all.trainx[i], all.trainy[i])
        del all


    def popMLModel(self, refer_d, target_d):
        self.classification(refer_d, target_d)
        count = 0
        RSE = 0
        predict = ClassicPredictModel()
        
        if self.VP.num_of_test > max(2, refer_d):
            predict.data = self.VP
            predict.MLModel()
            count += self.VP.num_of_test
            RSE += self.VP.num_of_test * predict.mRSE
        
        if self.P.num_of_test > max(2, refer_d):
            predict.data = self.P
            predict.MLModel()
            count += self.P.num_of_test
            RSE += self.P.num_of_test * predict.mRSE
        
        if self.NSP.num_of_test > max(2, refer_d):
            predict.data = self.NSP
            predict.MLModel()
            count += self.NSP.num_of_test
            RSE += self.NSP.num_of_test * predict.mRSE

        if self.U.num_of_test > max(2, refer_d):
            predict.data = self.U
            predict.MLModel()
            count += self.U.num_of_test
            RSE += self.U.num_of_test * predict.mRSE
        
        if self.VU.num_of_test > max(2, refer_d):
            predict.data = self.VU
            predict.MLModel()
            count += self.VU.num_of_test
            RSE += self.VU.num_of_test * predict.mRSE
        
        self.mRSE = RSE / count




class AltimanModel():
    """
        Classify video into 7 classes according to time series
    """

    def __init__(self):
        """
            @p: num of parameter
            @u: list of parameter
        """
        self.Gof = 0.0
        self.MSC = 0.0
        self.MER = 0.0
        self.p = 0
        self.u = []
        self.data = Data()
        
    
    def load(self, refer_d, target_d):
        self.data.load(open(FILE, "r"), refer_d, target_d)
        self.data.turnCDF()


    def LModel(self, x):
        """
            Linear Model: 
            y = y0 + lamb * x
        """
        y0, lamb = self.u
        if type(x) == int:
            return y0 + lamb * x
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append(y0 + lamb * x[i])
            return y

    def EModel(self, x):
        """
            Exponential Model: 
            y = y0 + (M - y0)(1 - e^(-lamb * x))
        """
        y0, lamb, M = self.u
        if type(x) == int:
            return y0 + (M - y0) * (1 - np.exp((-1) * lamb * x))
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append(y0 + (M - y0) * (1 - np.exp((-1) * lamb * x[i])))
            return y


    def GModel(self, x):
        """
            Gompertz Model: 
            y = M * e^(-log(M/y0) * exp(-lamb * x))
        """
        y0, lamb, M = self.u
        if type(x) == int:
            return M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x))
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append(M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x[i])))
            return y


    def SModel(self, x):
        """
            Sigmoid Model: 
            y = M / (1 + (M/y0 - 1) * exp(-lamb * x * M))
        """
        y0, lamb, M = self.u
        if type(x) == int:
            return (M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x * M))
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append((M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x[i] * M)))
            return y


    def MEModel(self, x):
        """
            Modified Modified Exponential Model: 
            y = y0 + (M - y0)(1 - e^(-lamb * x)) + kx
        """
        y0, lamb, M, k = self.u
        if type(x) == int:
            return y0 + (M - y0) * (1 - np.exp((-1) * lamb * x)) + k * x
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append(y0 + (M - y0) * (1 - np.exp((-1) * lamb * x[i])) + k * x[i])
            return y


    def MGModel(self, x):
        """
            Modified Gompertz Model: 
            y = M * e^(-log(M/y0) * exp(-lamb * x)) + kx
        """
        y0, lamb, M, k = self.u
        if type(x) == int:
            return M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x)) + k * x
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append(M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x[i])) + k * x[i])
            return y


    def MSModel(self, x):
        """
            Modified Sigmoid Model: 
            y = M / (1 + (M/y0 - 1) * exp(-lamb * x * M)) + kx
        """
        y0, lamb, M, k = self.u
        if type(x) == int:
            return (M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x * M)) + k * x
        if type(x) == list:
            y = []
            for i in range(len(x)):
                y.append((M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x[i] * M)) + k * x[i])
            return y


    """
       xError(): error func, used to fit model
    """

    def LModelError(self, u, x, y):
        y0, lamb = u
        return (y0 + lamb * x) - y
    
    
    def EModelError(self, u, x, y):
        y0, lamb, M = u
        return (y0 + (M - y0) * (1 - np.exp((-1) * lamb * x))) - y

    
    def GModelError(self, u, x, y):
        y0, lamb, M = u
        return (M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x))) - y


    def SModelError(self, u, x, y):
        y0, lamb, M = u
        return (M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x * M)) - y


    def MEModelError(self, u, x, y):
        y0, lamb, M, k = u
        return (y0 + (M - y0) * (1 - np.exp((-1) * lamb * x)) + k * x) - y
    
    
    def MGModelError(self, u, x, y):
        y0, lamb, M, k = u
        return (M * np.exp((-1) * np.log((M + 0.0)/ y0) * np.exp((-1) * lamb * x)) + k * x) - y


    def MSModelError(self, u, x, y):
        y0, lamb, M, k = u
        return (M + 0.0) / (1 + ((M + 0.0)/ y0 - 1) * np.exp((-1) * lamb * x * M) + k * x) - y
    
    

    def LModelFit(self, x):
        """
            let self.u fit Linear Model
        """
        self.p = 2
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.LModelError, u, args = (x, y))[0]


    def EModelFit(self, x):
        """
            let self.u fit Exponential Model
        """
        self.p = 3
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.EModelError, u, args = (x, y))[0]


    def GModelFit(self, x):
        """
            let self.u fit Gompertz Model
        """
        self.p = 3
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.GModelError, u, args = (x, y))[0]


    def SModelFit(self, x):
        """
            let self.u fit Sigmoid Model
        """
        self.p = 3
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.SModelError, u, args = (x, y))[0]


    def MEModelFit(self, x):
        """
            let self.u fit Modified Exponential Model
        """
        self.p = 4
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.MEModelError, u, args = (x, y))[0]


    def MGModelFit(self, x):
        """
            let self.u fit Modified Gompertz Model
        """
        self.p = 4
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.MGModelError, u, args = (x, y))[0]


    def MSModelFit(self, x):
        """
            let self.u fit Modified Sigmoid Model
        """
        self.p = 4
        u = [2] * self.p
        x = np.array([i + 1 for i in range(len(x))])
        y = np.array(x)
        self.u = leastsq(self.MSModelError, u, args = (x, y))[0]


    """
        @y1: original series @y2: predicted series
    """
    def getMSC(self, y1, y2):
        msc = 0.0
        for i in range(len(y1)):
            msc += (y1[i] -y2[i]) ** 2
        """
            normalization
        """
        msc /= (y1[len(y1) - 1] + 0.0)
        return msc


    def getGof(self, y1, y2):
        return self.getMSC(y1, y2) / (len(y1) - self.p)


    def getMER(self, y1, y2):
        mer = 0.0
        for i in range(len(y1)):
            mer += abs(y1[i] - y2[i])
        mer /= len(y1) * (y1[len(y1) - 1] + 1)
        return mer



class Altman:

    def __init__(self):
        self.L  = Data()
        self.E  = Data()
        self.G  = Data()
        self.S  = Data()
        self.ME = Data()
        self.MG = Data()
        self.MS = Data()
        self.mRSE = 0


    def classification(self, refer_d, target_d):
        """
            calculate Gof of each model, select the min one as the fit model
            @y1: original series @y2: predicted series
        """
        all = AltimanModel()
        all.load(refer_d, target_d)
        all.data.turnCDF()
        print all.data.num_of_test
        """
            Classify test data
        """
        for i in range(all.data.num_of_test):
            y1 = all.data.testxCDF[i]

            all.LModelFit(y1)
            y2 = all.LModel(y1)
            L_Gof = all.getGof(y1, y2)

            all.EModelFit(y1)
            y2 = all.EModel(y1)
            E_Gof = all.getGof(y1, y2)

            all.GModelFit(y1)
            y2 = all.GModel(y1)
            G_Gof = all.getGof(y1, y2)
            
            all.SModelFit(y1)
            y2 = all.SModel(y1)
            S_Gof = all.getGof(y1, y2)
            
            all.MEModelFit(y1)
            y2 = all.MEModel(y1)
            ME_Gof = all.getGof(y1, y2)
            
            all.MGModelFit(y1)
            y2 = all.MGModel(y1)
            MG_Gof = all.getGof(y1, y2)
            
            all.MSModelFit(y1)
            y2 = all.MSModel(y1)
            MS_Gof = all.getGof(y1, y2)
            
            #print [E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]

            if L_Gof    == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.L.addTestData(all.data.testx[i], all.data.testy[i])
            elif G_Gof  == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.G.addTestData(all.data.testx[i], all.data.testy[i])
            elif E_Gof  == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.E.addTestData(all.data.testx[i], all.data.testy[i])
            elif S_Gof  == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.S.addTestData(all.data.testx[i], all.data.testy[i])
            elif ME_Gof == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.ME.addTestData(all.data.testx[i], all.data.testy[i])
            elif MG_Gof == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.MG.addTestData(all.data.testx[i], all.data.testy[i])
            elif MS_Gof == min([L_Gof, E_Gof, G_Gof, S_Gof, ME_Gof, MG_Gof, MS_Gof]):
                self.MS.addTestData(all.data.testx[i], all.data.testy[i])


if __name__ == '__main__':
    
    p = ClassicPredictModel()
    p.load(17, 23)
    p.SHModel()
    print p.mRSE
    p.MLModel()
    print p.mRSE
    t = Pop()
    t.popMLModel(17, 23)
    print t.mRSE

    q = AltimanModel()
    q.load(17, 23)
    q.data.turnCDF()
    
    q.LModelFit(q.data.testxCDF[0])
    print q.u
    y2 = q.LModel(q.data.testxCDF[0])
    y1 = q.data.testxCDF[0]
    print q.p
    print q.getGof(y1, y2)
    print q.getMER(y1, y2)

    i = Altman()
    i.classification(17, 23)
    print i.L.num_of_test
    print i.E.num_of_test
    print i.G.num_of_test
    print i.S.num_of_test
    print i.ME.num_of_test
    print i.MG.num_of_test
    print i.MS.num_of_test







