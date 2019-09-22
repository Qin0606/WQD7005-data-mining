# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import MapCompose, TakeFirst
from w3lib.html import remove_tags

def remove_whitespace(value):
    return value.strip()

class GoldItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    
    date_text = scrapy.Field(
            input_processor = MapCompose(remove_tags,remove_whitespace),
            output_processor = TakeFirst()
            )
    
    closing_price = scrapy.Field(
            input_processor = MapCompose(remove_tags,remove_whitespace),
            output_processor = TakeFirst()
            )
    
    open_price = scrapy.Field(
            input_processor = MapCompose(remove_tags,remove_whitespace),
            output_processor = TakeFirst()
            )
    
    daily_high = scrapy.Field(
            input_processor = MapCompose(remove_tags,remove_whitespace),
            output_processor = TakeFirst()
            )
    
    daily_low = scrapy.Field(
            input_processor = MapCompose(remove_tags,remove_whitespace),
            output_processor = TakeFirst()
            )