import json

import scrapy
from pydispatch import dispatcher
from scrapy import signals

# to run: /usr/local/bin/scrapy crawl wiki_cats

class WikiCats(scrapy.Spider):
    name = 'wiki_cats'
    allowed_domains = ['www.wikidata.org']
    base_domain_url = 'https://www.wikidata.org'
    base_url = 'https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all'
    start_urls = [base_url]
    # Note: As of Oct 19 2021 the following languages have "list of properties" page: 'en', 'bn', 'cs', 'de', 'el', 'eo', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'it', 'nl', 'nb', 'pt', 'ru', 'sv'

    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.lang_map = {}
        dispatcher.connect(self.spider_closed, signals.spider_closed)

    def spider_closed(self, spider):
        for lang in self.lang_map:
            # with open(f'pid_names_{lang}_ext.json', 'wt', encoding='utf8') as file:
            #     json.dump(self.lang_map[lang], file, ensure_ascii=False, indent=4, sort_keys=True)
            with open(f'pid_names_{lang}.json', 'wt', encoding='utf8') as file:
                pid_names = {}
                for item in self.lang_map[lang]:
                    pid_names[item['pid']] = item['name']
                json.dump(pid_names, file, ensure_ascii=False, indent=4, sort_keys=True)

    def parse(self, response, **kwargs):
        try:
            if response.url == self.base_url:
                lang = 'en'
                lang_lis = response.css('#mw-content-text > div.mw-parser-output > div > div > ul > li')
                for li in lang_lis:
                    lang_local_link = li.xpath("bdi/a/@href").extract_first()
                    if lang_local_link:
                        lang_link = self.base_domain_url + lang_local_link
                        if self.base_url in lang_link:
                            yield scrapy.Request(lang_link, self.parse)
            else:
                lang = response.url.rpartition('/')[-1]
            cat_rows = response.css('#mw-content-text > div.mw-parser-output > table > tbody > tr')
            if cat_rows:
                for row in cat_rows[1:]:
                    td1 = row.xpath('td[1]/a[1]/text()').extract_first()
                    td2 = row.xpath('td[2]/text()').extract_first()
                    td3 = row.xpath('td[4]/text()').extract_first()
                    if td1 and td2:
                        if lang not in self.lang_map:
                            self.lang_map[lang] = []
                        pid = td1.strip()
                        name = td2.strip()
                        aliases = td3.strip() if td3 else ''
                        aliases = tuple([alias.strip() for alias in aliases.split(',')])
                        self.lang_map[lang].append({'pid': pid, 'name': name, 'aliases': aliases})
            return []
        except BaseException as e:
            print(e)
            raise e
