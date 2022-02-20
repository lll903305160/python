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
    html = etree.HTML(html)
    movie_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")
    name_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")
    #获取返回电影数据的个数
    num = len(movie_lists)
    if num>15:        #豆瓣搜索第一页会有16条数据，第一条是导演介绍需跳过
        movie_lists = movie_lists[1:]
        name_lists = name_lists[1:]
    for movie,name_list in zip(movie_lists,name_lists):
        if name_list.text is None:
            continue
        print(name_list.text)
        names = name_list.text.split('/')
        if names[0].strip() ==director and movie.text not in flags:
            names[0] = movie.text #保留疑问
            flags.append(movie.text)
            csv_write.writerow(names)
    print('OK')
    print(num)
    if num >= 14:
        return True
    else:
        return False
    

start = 0
while start < 10000:
    request_url = base_url+str(start)
    flag = download(request_url)
    if flag:
        start = start + 15
    else:
        break
        
out.close()
print('finished')
    
