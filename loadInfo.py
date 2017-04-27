# -*- coding: utf-8 -*-

"""
    Load dataset
    Author: Aining Wang
"""

FILE1 = "vtype_vdate_duration_title_vtotalview_sampling10_14.txt"
FILE2 = "new_upload_sampling10_14.txt"
POP_LEN = 23


from pyltp import Segmentor, Postagger, NamedEntityRecognizer


class OneVideoInfo():
    """
        including vid, popularity, title, duration, type
    """
   
    def __init__(self, vid):
        self.vid        = vid
        self.pop        = []
        self.title      = ""
        self.duration   = 0
        self.type       = ""
        self.title_named_entity = 0


    def namedEntityCounter(self, segmentor, postagger, recognizer):
        """
            count named entity in title
        """
        self.title_named_entity = 0
        try:
            words   = segmentor.segment(self.title)
            postags = postagger.postag(words)
            netags  = recognizer.recognize(words, postags)
        except:
            self.title_named_entity = 0
        else:
            for i in range(len(netags)):
                if netags[i] != "O":
                    self.title_named_entity += 1



class AllVideoInfo():
    """
        Load all video info
        @self.video_info[vid] = OneVideoInfo()
    """
    def __init__(self):
        self.num_of_video   = 0
        self.video_info     = {}
        self.mean_pop       = []


    def loadInfo(self, file):
        """
            load vid, title, duration, type
            
            the file is as follows:
                vid \t totalview \t type \t year \t month \t day \t duration \t title \n
        """
        for line in file:
            content = line.strip("\n").split("\t")
            vid     = content[0]
            type    = content[2]
            duration= content[6]
            title   = content[7]
            
            self.num_of_video += 1
            self.video_info[vid]            = OneVideoInfo(vid)
            self.video_info[vid].title      = title
            self.video_info[vid].duration   = duration
            self.video_info[vid].type       = type
        file.close()


    def loadPop(self, file):
        """
            load popularity
            if vid not in file, del AllVideoInfo[vid]
            
            the format of file is as follows:
                vid1 \t day1 \t day2 ... dayN \n
                vid2 ...
        """
        info = {}
        del_vid = []
        for line in file:
            line = line.strip("\n").split("\t")
            info[line[0]] = line[1:]
        for vid in self.video_info:
            if vid in info:
                self.video_info[vid].pop = map(int, info[vid])
            else:
                del_vid.append(vid)
        for i in range(len(del_vid)):
            del self.video_info[del_vid[i]]
            self.num_of_video -= 1
        del info
        del del_vid
        file.close()


    def load(self):
        self.loadInfo(open(FILE1, "r"))
        self.loadPop(open(FILE2, "r"))
    
    
    def typeCount(self):
        counter = {}
        for vid in self.video_info:
            if self.video_info[vid].type not in counter:
                counter[self.video_info[vid].type] = 1
            else:
                counter[self.video_info[vid].type] += 1
        for type in counter:
            print type, counter[type]


    def meanPop(self):
        """
            mean popularity of everyday
        """
        all_pop = [0] * POP_LEN
        for vid in self.video_info:
            all_pop = [all_pop[i] + self.video_info[vid].pop[i] for i in range(POP_LEN)]
        mean_pop = [(all_pop[i] + 0.0)/self.num_of_video for i in range(POP_LEN)]
        self.mean_pop = mean_pop


    def addVideo(self, info):
        """
            add a video into data set
        """
        self.num_of_video += 1
        self.video_info[info.vid] = info


    def namedEntityCounter(self):
        segmentor = Segmentor()
        postagger = Postagger()
        recognizer = NamedEntityRecognizer()
        segmentor.load('/Users/ainingwang/Documents/popularity/ltp_data/cws.model')
        postagger.load('/Users/ainingwang/Documents/popularity/ltp_data/pos.model')
        recognizer.load('/Users/ainingwang/Documents/popularity/ltp_data/ner.model')
        
        for vid in self.video_info:
            if self.video_info[vid].type == "新闻":
                self.video_info[vid].namedEntityCounter(segmentor, postagger, recognizer)
        #print str(self.video_info[vid].title_named_entity) + "\t" + str(sum(self.video_info[vid].pop[:5])) + "\t" + self.video_info[vid].title

        segmentor.release()
        postagger.release()
        recognizer.release()



if __name__ == '__main__':

    all = AllVideoInfo()
    all.loadInfo(open(FILE1, "r"))
    print all.num_of_video
    all.loadPop(open("new_upload_sampling10_14.txt", "r"))
    print all.num_of_video
    all.typeCount()
    all.meanPop()
    print all.mean_pop
    all.namedEntityCounter()


