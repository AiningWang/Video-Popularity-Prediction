# -*- coding: utf-8 -*-

from loadInfo import AllVideoInfo

import matplotlib.pyplot as plt


"""
    Test the influence of some factors(type, title, ...)
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


    def meanPop(self):
        self.news.meanPop()
        self.music.meanPop()
        self.edu.meanPop()
        self.movie.meanPop()
        self.TV.meanPop()
        self.sport.meanPop()



if __name__ == '__main__':

    all = AllVideoInfo()
    all.load()
    print all.num_of_video
    test = TestType()
    test.classification()
    test.meanPop()
    print test.music.num_of_video
    plt.plot(test.sport.mean_pop)
    plt.title("Popularity change")
    plt.ylabel("mean popularity")
    plt.xlabel("day")
    plt.show()




