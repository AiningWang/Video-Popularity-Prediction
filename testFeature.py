# -*- coding: utf-8 -*-

from loadInfo import AllVideoInfo

import matplotlib.pyplot as plt


"""
    Test the influence of some factors(type, title, ...)
    if the factor works, implement the model in "newModel.py"
    
    Author: Aining Wang
    
"""

class TestType:
    """
        test the influence of type
    """
    
    def __init__(self):
        self.news   = AllVideoInfo()
        self.music  = AllVideoInfo()
        self.edu    = AllVideoInfo()
        self.movie  = AllVideoInfo()
        self.TV     = AllVideoInfo()
        self.sport  = AllVideoInfo()
    
    
    def classification(self):
        all = AllVideoInfo()
        all.load()
        for vid in all.video_info:
            v = all.video_info[vid]
            if v.type == "新闻":
                self.news.addVideo(v)
            elif v.type == "音乐":
                self.music.addVideo(v)
            elif v.type == "教育":
                self.edu.addVideo(v)
            elif v.type == "电影":
                self.movie.addVideo(v)
            elif v.type == "电视剧":
                self.TV.addVideo(v)
            elif v.type == "体育":
                self.sport.addVideo(v)
        del all


    def meanPop(self):
        self.news.meanPop()
        self.music.meanPop()
        self.edu.meanPop()
        self.movie.meanPop()
        self.TV.meanPop()
        self.sport.meanPop()


class TestNameEntity:
    """
        test the influence of type
    """
    
    def __init__(self):
        self.zero   = AllVideoInfo()
        self.one    = AllVideoInfo()
        self.two    = AllVideoInfo()
        self.three  = AllVideoInfo()
        self.four   = AllVideoInfo()


    def classification(self):
        all = AllVideoInfo()
        all.load()
        all.namedEntityCounter()
        
        for vid in all.video_info:
            v = all.video_info[vid]
            if v.type == "新闻" and v.title_named_entity == 0:
                self.zero.addVideo(v)
            if v.type == "新闻" and v.title_named_entity == 1:
                self.one.addVideo(v)
            if v.type == "新闻" and v.title_named_entity == 2:
                self.two.addVideo(v)
            if v.type == "新闻" and v.title_named_entity == 3:
                self.three.addVideo(v)
            if v.type == "新闻" and v.title_named_entity == 4:
                self.four.addVideo(v)
        del all
    
    
    def meanPop(self):
        self.zero.meanPop()
        self.one.meanPop()
        self.two.meanPop()
        self.three.meanPop()
        self.four.meanPop()



if __name__ == '__main__':

    all = AllVideoInfo()
    all.load()
    print all.num_of_video
    
    """
    test = TestType()
    test.classification()
    test.meanPop()
    print test.news.num_of_video
    plt.plot(test.sport.mean_pop)
    plt.title("Popularity change")
    plt.ylabel("mean popularity")
    plt.xlabel("day")
    plt.show()
    """
    """
    t = TestNameEntity()
    t.classification()
    t.meanPop()
    print t.one.num_of_video
    plt.plot(t.one.mean_pop)
    plt.title("Popularity change")
    plt.ylabel("mean popularity")
    plt.xlabel("day")
    plt.show()
    """
 



