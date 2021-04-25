import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time
import json
from tqdm import tqdm
import re # 정규식을 활용하여 괄호 안 숫자 추출
import warnings
warnings.filterwarnings("ignore")

class crawling:
    def __init__(self, data_path,save_path):
        self.save_path = save_path
        self.data = pd.read_csv(os.path.join(data_path,"Final Res.csv"))
        self.data_pre = self.preprocess()
        self.run()
    
    def preprocess(self):
        data = self.data
        data = data[data["상권업종중분류명"] == "커피점/카페"]
        data = data[["상호명","상권업종중분류명","상권업종소분류명","도로명주소","역명","거리"]]
        data["역"] = data["역명"].apply(lambda x : x.split(" ")[1])
        data["역"] = data["역"].apply(lambda x : x.split("(")[0])
        data["검색어"] = data.apply(lambda x : x["역"] + "역 "+str(x["상호명"]) ,axis=1)
        return data
    
    def run(self):
        items = self.data_pre["검색어"]
        rating_df = pd.DataFrame(columns = ["UserID","ItemID","Rating","Timestamp"])
        user_review_id = {}
        review_json = {}
        image_json = {}
        item_meta = {}

        options = Options()
        options.add_argument('--headless')
        driver = webdriver.Chrome('/home/zhqhddl13/chromedriver.exe',chrome_options=options)
        driver.maximize_window() # 창 크기 최대화
        for item in tqdm(items): 
            driver.get("https://www.google.com/maps/") # 구글 지도 접속하기
            time.sleep(1)
            searchbox = driver.find_element_by_css_selector("input#searchboxinput") # 검색창에 입력하기
            searchbox.send_keys(item)
            searchbutton = driver.find_element_by_class_name("searchbox-searchbutton") # 검색버튼 누르기
            searchbutton.click()
            time.sleep(5)

            # 바로 식당 페이지가 나오는 경우
            if len(driver.find_elements_by_xpath("//button[@jsaction='pane.rating.moreReviews']")) != 0:
                reviewbutton = driver.find_element_by_xpath("//button[@jsaction='pane.rating.moreReviews']")
                res_name = driver.find_element_by_xpath("//div[@class='section-layout section-layout-root']").get_attribute("aria-label")
                if res_name == None:
                    res_name = driver.find_elements_by_xpath("//div[@class='section-layout section-layout-root']")[1].get_attribute("aria-label")
                reviewbutton.click()

            # 식당 검색결과가 목록으로 나오는 경우 (여러 식당)
            elif len(driver.find_elements_by_xpath("//a[@class='place-result-container-place-link']")) > 0:
                search_list =  driver.find_elements_by_class_name('place-result-container-place-link')
                search_cnt = len(search_list) # 식당 목록 중 최대 상위 6개 

                flag_res = False
                flag_rev = False
                for i in range(search_cnt): # 식당 목록 탐색
                    # 구글 Map상의 식당이름에 원하는 식당 이름이 포함된 경우 
                    if item.split()[1] in driver.find_elements_by_xpath("//a[@class='place-result-container-place-link']")[i].get_attribute("aria-label"):
                        res_name = driver.find_elements_by_xpath("//a[@class='place-result-container-place-link']")[i].get_attribute("aria-label")
                        driver.get(driver.find_elements_by_class_name('place-result-container-place-link')[i].get_attribute('href')) # 해당 식당으로 접속
                        time.sleep(2)
                        flag_res = True
                        if len( driver.find_elements_by_xpath("//button[@jsaction='pane.rating.moreReviews']")) != 0:
                            flag_rev=True
                            reviewbutton = driver.find_element_by_xpath("//button[@jsaction='pane.rating.moreReviews']")
                            reviewbutton.click()

                        break
                if flag_res == False:
                    continue
                elif flag_rev == False:
                    continue

            else:
                continue

            time.sleep(2)
            # 리뷰 전체 뽑기 위해 스크롤 다운
            for num in range(30):
                try:
                    driver.find_element_by_class_name('section-loading').click()
                    time.sleep(2)
                except:
                    time.sleep(2)
                    break

            spread_review = driver.find_elements_by_xpath("//button[@jsaction='pane.review.expandReview']") 


            # 시간 지연
            time.sleep(2)
            user = driver.find_elements_by_xpath("//a[@class='section-review-reviewer-link']")
            user_review = driver.find_elements_by_xpath("//button[@class='section-review-action-menu section-review-action-menu-with-title']")
            date = driver.find_elements_by_xpath("//span[@class='section-review-publish-date']")
            rating = driver.find_elements_by_xpath("//span[@class='section-review-stars']")
            review = driver.find_elements_by_xpath("//span[@class='section-review-text']")


            # Image
            image = driver.find_elements_by_xpath("//button[@class='section-review-photo']")
            review_json[item] = {}
            user_review_id[item] = {}
            image_json[item] = {}
            item_meta[item] = {"Name" : res_name}

            for i in range(len(date)):
                if review[i].text is not "":
                    user_elem = re.findall("\d+",user[i].get_attribute("href"))[0]
                    item_elem = item
                    rating_elem = re.findall("\d+",rating[i].get_attribute('aria-label'))[0]
                    date_elem = date[i].text
                    row = {"UserID":user_elem,"ItemID":item_elem,"Rating":rating_elem,"Timestamp":date_elem}
                    row = pd.DataFrame(row, index=[i])
                    rating_df = rating_df.append(row,ignore_index=True)
                    user_review_id[item][user_elem] = user_review[i].get_attribute("data-review-id")
                    review_json[item][user_review[i].get_attribute("data-review-id")] = review[i].text

            for i in range(len(image)):
                image_json[item][image[i].get_attribute("data-review-id")] = re.findall(r'\"(.+?)\"',image[i].get_attribute('style'))[0]
            
        with open(os.path.join(self.save_path,"item_meta.json"), 'w') as outfile:
            json.dump(item_meta, outfile)
        with open(os.path.join(self.save_path,"review.json"), 'w') as outfile:
            json.dump(review_json, outfile)
        with open(os.path.join(self.save_path,"user_review.json"), 'w') as outfile:
            json.dump(user_review_id, outfile)
        with open(os.path.join(self.save_path,"image.json"), 'w') as outfile:
            json.dump(image_json, outfile)
        rating_df.to_csv(os.path.join(self.save_path,"rating.csv"),index=False)
        driver.close()
        
        
if __name__ == "__main__":
    data_path = "./"
    save_path = "./"
    crawling(data_path,save_path)