# The AIDA dataset is from the early 2000s, so Wikipedia pages mentioned in the original data now have different titles. 
# Example: http://en.wikipedia.org/wiki/San_Diego_Chargers (referenced in the original) now redirects to https://en.wikipedia.org/wiki/History_of_the_San_Diego_Chargers
# We need to process every title and update it with its new version 



import argparse, os, requests, urllib

def get_page_title_from_url(url):
    
    url = url.replace("http://en.wikipedia.org/wiki/", "")
    url = url.replace("_", " ")
    return url

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotated_file', type=str, default='data/aida/AIDA-YAGO2-dataset.tsv', help='Path to AIDA file')
    parser.add_argument('--out_dir', type=str, default='data/aida/', help='Path to AIDA file')
    args = parser.parse_args()
    return args

def get_updated_title(original_url):

    original_title = urllib.parse.quote(get_page_title_from_url(original_url))
    query = f"https://en.wikipedia.org/w/api.php?action=query&titles={original_title}&&redirects&format=json"
    r = requests.get(query)
    obj = r.json()
    if 'warnings' in obj:
        print("ERROR:", obj)
    obj = obj['query']['pages']
    # response looks something like this: {'batchcomplete': '', 'query': {'redirects': [{'from': 'Halab', 'to': 'Aleppo'}], 'pages': {'159244': {'pageid': 159244, 'ns': 0, 'title': 'Aleppo'}}}}    
    key = list(obj.keys())[0]
    return obj[key]['title']

def load_sentences(args):
    ''' Load sentences from fp ''' 

    original_urls = set()
    with open(args.annotated_file, 'r', encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()         
            if len(line.split("\t")) > 4:
                items = line.split("\t")
                wiki = items[4]
                original_urls.add(wiki)
    print(f"{len(original_urls)} distinct urls.")
    url_to_title = {}
    for i, url in enumerate(list(original_urls)):
        title = get_updated_title(url)
        url_to_title[url] = title
        same_title = title == get_page_title_from_url(url)
        if same_title:
            print(f"{i}/{len(original_urls)}. {url} --> {title}")
        else:
            print(f"{i}/{len(original_urls)}. {url} --> {title} CHANGED")

    with open(os.path.join(args.out_dir, 'url_to_title.tsv'), 'w', encoding='utf8') as out_file:
        for url, title in url_to_title.items():
            out_file.write(f"{url}\t{title}\n")




def main():
    args = parse_args()
    load_sentences(args)

if __name__ == '__main__':
    main()