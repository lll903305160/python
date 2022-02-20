# -*- coding: utf-8 -*-
from efficient_apriori import apriori
from lxml import etree
import time
from selenium import webdriver
import csv
driver = webdriver.Chrome()

#设置搜寻导演
director = u'宁浩'

#写CSV文件
file_name = './'+ director + '.csv'
base_url = 'https://movie.douban.com/subject_search?search_text='+director+'&cat=1002&start='
out = open(file_name,'w',newline='',encoding='utf-8-sig')
csv_write = csv.writer(out,dialect='excel')
flags = []

#下载指定页面数据
def download(request_url):
    driver.get(request_url)
    time.sleep(1)
    html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
