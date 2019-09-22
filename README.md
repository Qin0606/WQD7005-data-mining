# WQD7005-data-mining
Repo for codes and data used in WQD7005 data mining course assignment

List of assignment completed:
  1. Gold price and news web crawler using Python Scrapy and Selenium.

To start a new scrapy project, use anaconda prompt, go to the desired folder or location, type 'scrapy startproject projectname'

It will then create a default folder structure and files required.

Tutorial from http://mroseman.com/scraping-dynamic-pages/ and also https://www.youtube.com/watch?v=Wp6LRijW9wg

Why Selenium is required? This is because some HTML elements involved JavaScript and can only be executed in the internet browser, and that JavaScript will grab data for a webpage. Scrapy does not have the ability to execute this JavaScript.

Make sure chrome driver is downloaded, required by Selenium. Can be downloaded from https://sites.google.com/a/chromium.org/chromedriver/

Chrome driver version is based on the browser, update the google chrom browser to version 77 (latest as of 22 Sep 2019)

Download chrome driver for version 77.Put the chromedriver.exe in the directory of the project folder.
