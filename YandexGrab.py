from selenium import webdriver
import requests
import time
from tqdm import tqdm
import numpy as np
import shutil
import pandas as pd
import os

print('import all libs')


def init_wd():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    wd = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
    return wd


def get_image(url, savename):
    try:
        response = requests.get(url, timeout=5)
        if not response.ok:
            print(response)
        with open(savename, 'wb') as handle:
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
    except:
        print('error response', url)


try:
    os.mkdir('data')
except FileExistsError:
    pass
wd = init_wd()  # init first time WebDriver
print('init wd')
#########################################################################################
# this url change with yours url 
url = 'https://yandex.ru/images/search?text=кулак%20человека%20фото&from=tabbar&pos=0&img_url=https%3A%2F%2Fst.depositphotos.com%2F1378829%2F2779%2Fi%2F950%2Fdepositphotos_27799037-stock-photo-fist.jpg&rpt=simage'
wd.get(url)
#########################################################################################
print('get url:', url)
r = wd.get_screenshot_as_png()
with open('wd_init.png', 'wb') as f:
    f.write(r)

limit = 100
last_height = wd.execute_script("return document.body.scrollHeight")
last_len = 0
print('start parse imgs')
while (True):
    imgs = wd.find_elements_by_class_name('serp-item__thumb')
    print(len(imgs))
    b = wd.find_element_by_class_name('more_direction_next')
    try:
        b.click()
    except:
        pass
    time.sleep(1)
    new_height = wd.execute_script("return document.body.scrollHeight")
    imgs = wd.find_elements_by_class_name('serp-item__link')
    if (last_len == len(imgs)):
        break;
    if (limit != 0):
        if (len(imgs) > limit):
            break;
    last_height = new_height
    last_len = len(imgs)

print('end parse imgs')

list_data = []
for im in imgs:
    url = im.get_attribute('href')
    # print(url)
    d = url.split('&')
    for i in range(len(d)):
        if ('img_url' in d[i]):
            d2 = d[i].split('=')[1]
            url = d2.split('%3A')
            url = ':'.join(url)
            url = url.split('%2F')
            url = '/'.join(url)
            list_data.append(url)
            break;
    # print(list_data[0])
result_url = list_data

"""
print('len list data=',len(list_data))
print(list_data[0])
result_url=[]
#print(filename)
for i in tqdm(range(len(list_data))):
    #print(i)
    wd.get(list_data[i])
    #stime=np.random.randint(100,300)/100
    #time.sleep(stime)
    #if(i==0):
    #r=wd.get_screenshot_as_png()
    #with open('wd.png', 'wb') as f:
     #           f.write(r)
    
    for el in wd.find_elements_by_tag_name('a'):
        try: class_name=el.get_attribute('class')
        except: break
        if('sizes__download' in class_name):
            #print(el.get_attribute('class'),el.get_attribute('href'))
           
            url=el.get_attribute('href')
            result_url.append(url)
            #print(i,url)
            #try: wget.download(url,'data',bar=None)
            #except: pass
            #urllib.request.urlretrieve(url,'test.png')
            #image=urllib.URLopener()
            #image.retrieve(url,'test.png')
            #print(url)
            #get_image(url,'test.png')
            break;
         
    if(len(result_url)==0 and i>5):
            break;        
    #r=wd.get_screenshot_as_png()
    #with open('wd.png', 'wb') as f:
     #   f.write(r)
print('save the url list')"""

pdata = pd.DataFrame(result_url, columns=['url'])
pdata.to_csv('url_list.csv')

print('start grab the images')

for i in tqdm(range(len(result_url))):
    get_image(result_url[i], f'data/{i}.jpg')
# print(result_url)
