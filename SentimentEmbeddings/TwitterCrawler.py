import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import re

consumer_key = 'eu29hnoeY8pv8Iw4FGZl5FvGR'
consumer_secret = 'IFBMnrWlyF1QjPmf9eEcC81xbSLP0WG2YanfYa6X6CjNK02jrO'
access_token = '2671955514-sVYOa4hJy9G8FAqKFjHI0jQjGqKQZ1710R4jetS'
access_secret = 'Y3GVVuVB0AQhvcpzSxPkLhuP6wx41Rhbi73WZGep4YLdD'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth,wait_on_rate_limit = True, wait_on_rate_limit_notify =True)

def break_http(tweet):
    tweet=tweet.split('https')[0]
    if('#' in tweet):
       tokens=tweet.split()
       for tok in tokens:
           if(tok.startswith('#')): 
              print(tok)
    tweet=re.sub(r'[-#@._\']','',tweet)
    return tweet

def process_or_store(tweet):
    print(json.dumps(tweet))

for status in tweepy.Cursor(api.home_timeline).items(5):
    print(break_http(status.text),end='\n\n')

class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('twitter.txt', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#MachineLearning','#ComputerVision',\
                '#DeepLearning','#DonaldTrump','#SarahHuckabee'])
#this will give the list of friends' urls shared
'''
ids= api.friends_ids()
urls=[]

for friend in ids:
    statuses = api.user_timeline(id=friend, count=200)
    for status in statuses:
        if status.entities and status.entities['urls']:
           for url in status.entities['urls']:
               urls.append((url['expanded_url'], status.author.screen_name))
               print((url['expanded_url'], status.author.screen_name))
'''
