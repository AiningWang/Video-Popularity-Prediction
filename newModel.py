# -*- coding: utf-8 -*-

"""
    Author: Aining Wang
"""

from loadInfo import AllVideoInfo
from testFeature import TestType
from predictPopularity import Data, ClassicPredictModel


class Type():
    """
        classify video according type, then use ML model...
    """
    
    def __init__(self):
        self.news   = AllVideoInfo()
        self.music  = AllVideoInfo()
        self.edu    = AllVideoInfo()
        self.movie  = AllVideoInfo()
        self.TV     = AllVideoInfo()
        self.sport  = AllVideoInfo()
        self.mRSE   = 0
    
    
    def classification(self):
        all = AllVideoInfo()
        all.load()
        for vid in all.video_info:
            if all.video_info[vid].type == "新闻":
                self.news.addVideo(all.video_info[vid])
            elif all.video_info[vid].type == "音乐":
                self.music.addVideo(all.video_info[vid])
            elif all.video_info[vid].type == "教育":
                self.edu.addVideo(all.video_info[vid])
            elif all.video_info[vid].type == "电影":
                self.movie.addVideo(all.video_info[vid])
            elif all.video_info[vid].type == "电视剧":
                self.TV.addVideo(all.video_info[vid])
            elif all.video_info[vid].type == "体育":
                self.sport.addVideo(all.video_info[vid])
        del all


    def TypeMLModel(self, refer_d, target_d):
        all = TestType()
        all.classification()
        RSE = 0
        all_num = 0
        
        one_type = ClassicPredictModel()
        one_type.data.load2(all.news.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE += one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test
        
        one_type = ClassicPredictModel()
        one_type.data.load2(all.music.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE += one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test

        one_type = ClassicPredictModel()
        one_type.data.load2(all.TV.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE += one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test
        
        one_type = ClassicPredictModel()
        one_type.data.load2(all.movie.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE = one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test

        one_type = ClassicPredictModel()
        one_type.data.load2(all.sport.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE = one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test

        one_type = ClassicPredictModel()
        one_type.data.load2(all.edu.video_info, refer_d, target_d)
        if one_type.data.num_of_train > max(2, refer_d):
            one_type.MLModel()
            all_num += one_type.data.num_of_test
            RSE = one_type.mRSE * one_type.data.num_of_test
            print one_type.mRSE, one_type.data.num_of_test

        self.mRSE = RSE / all_num



if __name__ == '__main__':
    all = AllVideoInfo()
    all.load()
    test = ClassicPredictModel()
    test.data.load2(all.video_info, 5, 23)
    test.MLModel()
    print test.mRSE
    
    test2 = TestType()
    test2.classification()
    test3 = ClassicPredictModel()
    test3.data.load2(test2.news.video_info, 5, 23)
    test3.MLModel()
    #print test3.mRSE

    p = Type()
    p.TypeMLModel(5, 23)
    print p.mRSE
    
