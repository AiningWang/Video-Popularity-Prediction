# -*- coding: utf-8 -*-

"""
    Collect video infomation from Internet
    Author: Aining Wang
"""


import urllib2
import re
import time

def Get_Vid_List(file):
	vid_list = []
	for line in file:
		line = line.strip("\n")
		vid_list.append(line)
	return vid_list


def Collect_Date(vid, file1, file2):

    vid = str(vid)[0 : len(vid) - 1]

    my_url = "https://v.qq.com/x/page/" + vid + ".html"
    request = urllib2.Request(my_url)
    request.add_header('User-Agent','Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    try:
        response = urllib2.urlopen(request, timeout = 5).read()
    except:
        print "Error"
    else:
        date = re.findall("20(.*?)年(.*?)月(.*?)日发布", response, flags = 0)
        type = re.findall("VIDEOTYPE = '(.*?)'", response, flags = 0)
        title_duration = re.findall('VIDEO_INFO = {"title":"(.*?)","duration":(.*?),', response, flags = 0)
        auther_info = re.findall('VPP_INFO = {"auth_type":(.*?),"avatar":(.*?),"count1":(.*?),"count2":(.*?),"euin":(.*?),"foldercount":(.*?),"info":(.*?),"isvpp":(.*?),"nick":"(.*?)","playcount":(.*?),"totalvideo":(.*?),', response, flags = 0) # all 11
        vtotalview = re.findall('<meta itemprop="interactionCount" content="(.*?)" />', response, flags = 0)

        if (type != []) and (date != []) and (title_duration != []) and (vtotalview != []):
            type        = type[0]
            date        = date[0]
            title       = title_duration[0][0]
            duration    = title_duration[0][1]
            vtotalview  = vtotalview[0] # 这部视频到现在的浏览量
            
            if duration[0] == "\"":
                duration = duration[1 : len(duration)-1] # sometimes duration = "xxx"
            if date[0] == "14" and date[1] == "11":
                print "get"
            file1.write(vid + "\t" + vtotalview + "\t" + type + "\t" + date[0] + "\t" + date[1] + "\t" + date[2] + "\t" + duration + "\t" + title + "\n")
            
        if (auther_info != []) and (vtotalview != []):
            vtotalview  = vtotalview[0]
            nick        = auther_info[0][8] # 发布者昵称
            totalvideo  = auther_info[0][10] # 发布者一共发布的视频数
            totalview   = auther_info[0][3] # 发布者总共的浏览量
            foldercount = auther_info[0][5] # 发布者的专辑数
            fans        = auther_info[0][2] # 订阅了这个发布者的人数
            file2.write(vid + "\t" + vtotalview + "\t" + nick + "\t" + totalvideo + "\t" + totalview + "\t" + foldercount + "\t" + fans + "\n")
        

file1 = open("vid_list_sampling.txt", "r")
file2 = open("vtype_vdate_duration_title_vtotalview_sampling.txt", "w")
file3 = open("vpp_info.txt", "w")
vid_list = Get_Vid_List(file1)

i = 0
t1 = time.time()
for vid in vid_list:
	if i % 20 == 0:
		t2 = time.time()
		print "\n Collect %d , spend %d second " % (i, (t2-t1))
	Collect_Date(vid, file2, file3)
	i += 1

file1.close()
file2.close()
file3.close()
