# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:50:36 2019

@author: Wei Qin
"""

"""
Xpath
//div/a

//div/a/text()

//div/a/@href

//p[@id = 'xxx']

//p[@class = 'xxx']

"""

import scrapy
from datetime import date
from datetime import timedelta
from pricecrawler.items import GoldItem
from scrapy.loader import ItemLoader

#To start a new scrapy project, use anaconda prompt, go to the desired folder or location, type 'scrapy startproject projectname'
#It will then create a default folder structure and files required.
#tutorial from http://mroseman.com/scraping-dynamic-pages/ and also https://www.youtube.com/watch?v=Wp6LRijW9wg
#Make sure chrome driver is downloaded
#Chrome driver version is based on the browser, update the google chrom browser to version 77 (latest as of 22 Sep 2019)
#Download chrome driver for version 77.Put the chromedriver.exe in the directory of the project folder.

class goldSpider(scrapy.Spider):
    name = 'gold'
    lastYear  = (date.today() - timedelta(days = 365)).strftime('%d.%m.%y')
    todayDate = date.today().strftime('%d.%m.%y')
    my_url = 'https://markets.businessinsider.com/commodities/historical-prices/gold-price/usd/'  + lastYear + '_' + todayDate
    start_urls = [my_url]
    
    def parse(self, response):
        for row in response.xpath("//div[@class='table-responsive']/table/tbody/tr"):
            l = ItemLoader(item=GoldItem(), selector = row)
            l.add_xpath("date_text", 'td[1]//text()')
            l.add_xpath("closing_price", 'td[2]//text()')
            l.add_xpath("open_price", 'td[3]//text()')
            l.add_xpath("daily_high", 'td[4]//text()')
            l.add_xpath("daily_low", 'td[5]//text()')
            
            yield l.load_item()
            

# -*- coding: utf-8 -*-

