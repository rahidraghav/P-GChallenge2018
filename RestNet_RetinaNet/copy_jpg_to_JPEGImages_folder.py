# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 13:18:07 2018

@author: vnk7kor
"""

import os, shutil, glob

src_fldr = "C:\innovation\hackaton\dataset_1\pharmacy"; ## Edit this

dst_fldr = "C:\innovation\hackaton\JPEGImages"; ## Edit this

try:
    os.makedirs(dst_fldr); ## it creates the destination folder
    for xml_file in glob.glob(src_fldr+"\\*.jpg"):
        shutil.copy2(xml_file, dst_fldr);
except IOError:
    print ("Error: in folder creation!")
else:
   print ("Written content in the file successfully")