import http.client
import urllib.parse
import json
import time
from newspaper import Article
from newspaper import Config
from tqdm import tqdm
import os
import random
from nltk.tokenize import word_tokenize
import pandas as pd
from dotenv import load_dotenv
import sys

load_dotenv()
ACCESS_KEY = os.getenv('MEDIASTACK_API_KEY', '')  # mediastack API key
if not ACCESS_KEY:
    raise RuntimeError("MEDIASTACK_API_KEY is not set. Add it to your .env.")

# Safer to create a new connection inside the loop
def create_connection():
    return http.client.HTTPConnection('api.mediastack.com')

category = os.getenv("FOLDER_PATH", "demo")

# Create category folder if it does not exist
if not os.path.exists(category):
    os.makedirs(category)

# Default API request parameters
base_params = {
    'access_key': ACCESS_KEY,
    'countries': os.getenv('MEDIASTACK_COUNTRIES', 'us'),
    'languages': os.getenv('MEDIASTACK_LANGUAGES', 'en'),
    'limit': int(os.getenv('MEDIASTACK_LIMIT', '10')),
}
media_category = os.getenv('MEDIASTACK_CATEGORY', '')
if media_category:
    base_params['categories'] = media_category
media_date = os.getenv('MEDIASTACK_DATE', '')
if media_date:
    base_params['date'] = media_date
media_sources = os.getenv('MEDIASTACK_SOURCES', '')
if media_sources:
    base_params['sources'] = media_sources

offset = int(os.getenv('MEDIASTACK_OFFSET', '0'))
batch_number = 1
total_count = 0
total = 0
article_count = int(os.getenv('CRAWL_MAX_ARTICLES', '20'))
dfs = []
while total <= article_count:
    conn = create_connection()

    params = base_params.copy()
    params['offset'] = offset

    query = urllib.parse.urlencode(params)
    conn.request('GET', f'/v1/news?{query}')

    res = conn.getresponse()
    data = res.read()
    conn.close()

    decoded_data = data.decode('utf-8')
    decoded_data = json.loads(decoded_data)

    if 'data' not in decoded_data or not decoded_data['data']:
        print("No more news available from API.")
        break

    batch_data = decoded_data['data']

    # Save each batch separately
    filename = f'{category}/news_metadata_{batch_number}.json'
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump({'data': batch_data}, json_file, ensure_ascii=False, indent=4)

    print(f"Saved {len(batch_data)} news: {filename} (cumulative {total_count + len(batch_data)})")

    total_count += len(batch_data)
    offset += 100

    time.sleep(1)  # wait 1s to avoid API rate limit
    
    # Load saved API metadata
    with open(f'{category}/news_metadata_{batch_number}.json', 'r') as file:
        data = json.load(file)['data']

    # Path to save crawled news data
    save_path = f'{category}/news_data_{batch_number}.json'

    # Helper: append dict to JSON list file
    def wr_dict(filename, dic):
        if not os.path.isfile(filename):
            data = []
            data.append(dic)
            with open(filename, 'w') as f:
                json.dump(data, f)
        else:      
            with open(filename, 'r') as f:
                data = json.load(f)
                data.append(dic)
            with open(filename, 'w') as f:
                json.dump(data, f)
                
    # Helper: remove file
    def rm_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    # Load already crawled data (empty list if file does not exist)
    if os.path.exists(save_path):
        with open(save_path, 'r') as file:
            have = json.load(file)
    else:
        have = []  # initialize empty list

    # Configure user agent
    USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

    config = Config()
    config.headers = {'Cookie': "cookie1=xxx;cookie2=zzzz"}  # add cookies if needed
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 5

    RETRY_ATTEMPTS = 1  # retry attempts
    count = 0

    # Parse article body text
    def parse_article(url):
        for attempt in range(RETRY_ATTEMPTS):
            try:
                article = Article(url, config=config)
                article.download()
                article.parse()
                return article.text
            except Exception as e:
                print(f"Error retrieving article from URL '{url}': {e}")
                return None
        return None

    # Crawl and persist news articles
    for idx, d in enumerate(tqdm(data)):
        # Skip already crawled entries
        if idx < len(have):
            continue
        url = d['url']
        maintext = parse_article(url.strip())
        
        if maintext is None:
            print(f"Failed to fetch article body: {url}")
            continue
        
        # Add body field
        d['body'] = maintext
        
        # Save crawled record
        wr_dict(save_path, d)
        
        # Increment processed count
        count += 1

        # Random delay between requests (1-3s)
        time.sleep(random.uniform(1, 3))

    with open(save_path, 'r') as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        dfs.append(df)
        total += len(df)
    
    batch_number += 1

    print(f"Crawled {total}/{article_count} articles so far...")

# Merge into a single DataFrame
if not dfs:
    print("No news retrieved. Exiting without creating artifacts.")
    sys.exit(0)
df = pd.concat(dfs, ignore_index=True)
df = df[df['body'].apply(lambda x: x!='')]
print(len(df))
df = df.drop_duplicates(subset=['body'])
print(len(df))
df['token_count'] = df['body'].apply(lambda x: len(word_tokenize(x)))
df = df[(df['token_count']>=512) & (df['token_count']<=8192)]
print(len(df))
df.to_csv(f"{category}/chunk.csv", index=False)

###################################################################
# Split into sentences and save as sentence.csv
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

chunk_df = pd.read_csv(f"{category}/chunk.csv")

# Split into sentences and build a DataFrame
rows = []
for idx, row in chunk_df.reset_index(drop=True).iterrows():
    text = row['body']
    chunk_id = idx  # use DataFrame index as chunk_id
    sentences = sent_tokenize(text)
    for sentence in sentences:
        rows.append({'chunk_id': chunk_id, 'text': sentence})

# Create sentences DataFrame
df_sentences = pd.DataFrame(rows)

# Save
df_sentences.to_csv(f"{category}/sentence.csv", index=False)

print("Sentence splitting complete. Total sentences:", len(df_sentences))