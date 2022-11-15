import requests
from datetime import datetime
import time
import json
import sys
import pandas

username = ""  # put the username you want to download in the quotes
subreddit = "relationships"  # put the subreddit you want to download in the quotes
# leave either one blank to download an entire user's or subreddit's history
# or fill in both to download a specific users history from a specific subreddit

filter_string = None

if username == "" and subreddit == "":
    print("Fill in either username or subreddit")
    sys.exit(0)
elif username == "" and subreddit != "":
    filter_string = f"subreddit={subreddit}"
elif username != "" and subreddit == "":
    filter_string = f"author={username}"
else:
    filter_string = f"author={username}&subreddit={subreddit}"

url = "https://api.pushshift.io/reddit/{object_type}/search?limit=1000&sort=desc&{filter_string}&before="
start_time = datetime.utcnow()


def download_from_url(filename, object_type):
    print(f"Saving {object_type}s to {filename}")

    count = 0
    handle = open(filename, 'w')
    previous_epoch = int(start_time.timestamp())

    while True:
        new_url = url.format(object_type=object_type, filter_string=filter_string) + str(previous_epoch)
        json_text = requests.get(new_url, headers={'User-Agent': "Post downloader by /u/Watchful1"})
        time.sleep(1)  # pushshift has a rate limit, if we send requests too fast it will start returning error messages

        try:
            json_data = json_text.json()
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            continue

        if 'data' not in json_data:
            break

        objects = json_data['data']
        if len(objects) == 0:
            break

        for object in objects:
            previous_epoch = object['created_utc'] - 1
            count += 1

        # initialize temp dataframe for batch of data in response
        df = pandas.DataFrame()

        # loop through each post pulled from res and append to df
        for post in objects:

            df = df.append({
                'subreddit': post.get('comment', 'relationships'),
                'title': post.get('title', '$$'),
                'created_utc': datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'selftext': post.get('selftext', '$$'),
                'upvote_ratio': post.get('upvote_ratio', '$$'),
                'score': post.get('score', 0),
                'permalink': post.get('permalink', '$$'),
                'id': post.get('id', '$$')
            }, ignore_index=True)
            print(df.head)
        df.to_csv(filename, mode='a')

    handle.close()


download_from_url("allcomments.csv", "comment")
# download_from_url("comments.txt", "comment")
