# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:33:37 2018

@author: vnk7kor
"""

import csv
import tensorflow as tf
import random
import xml.etree.ElementTree as ET


        
        
def create_example(xml_file, filewriter):
        #process the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_path = xml_file.decode('utf-8')
#        image_filename = root.find('filename').text
#        local_path= "./Annotations/"+image_filename
        for member in root.findall('object'):
            class_name = member[0].text
            csv_entry=[]
            csv_entry.append(image_path)
            csv_entry.append(float(member[4][0].text))
            csv_entry.append(float(member[4][1].text))
            csv_entry.append(float(member[4][2].text))
            csv_entry.append(float(member[4][3].text))
            csv_entry.append(class_name)
            filewriter.writerow(csv_entry)

		
    
def main(_):
    
    #provide the path to annotation xml files directory
    filename_list=tf.train.match_filenames_once("./Annotations/*.xml")
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    list=sess.run(filename_list)
    random.shuffle(list)   #shuffle files list
    with open('consolidated_annotation.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for xml_file in list:
            create_example(xml_file, filewriter)
    print('Successfully created dataset')	
	
if __name__ == '__main__':
    tf.app.run()