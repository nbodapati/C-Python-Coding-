import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import re
import json
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

'''
query='1F602'
for status in tweepy.Cursor(api.search,q=query).items(50):
    print(status.text,end='\n\n')
'''
class MyListener(StreamListener):
    def on_data(self, data):
        try:

           with open('emojis.txt','a') as fp:
                fp.write(data)
           fp.close()
  
           d=json.loads(data)
           print(d['text'])
           return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[u"\U0001F602",u'\U0001F601',u'\U0001F603',u'\U0001F604',u'\U0001F605',\
                             u'\U0001F606',u'\U0001F609',u'\U0001F60A',u'\U0001F60B',u'\U0001F60D', 
                             u'\U0001F60C',u'\U0001F60F',u'\U0001F612',\
                             u'\U0001F612',u'\U0001F612',u'\U0001F62A',u'\U0001F622',u'\U0001F62D',u'\U0001F63F'])
