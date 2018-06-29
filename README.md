
# News Mood

## Analysis

### Observed trend 1
Sentiment analysis of tweets by BBC, CBS, CNN, Fox News and the New York Times shows that most of the tweets are generally neutral in sentiment

### Observed trend 2
Aggregate sentiment for BBC, CNN and Fox News is very similar, at about -0.075. CBS is an outlier with the aggregate negative sentiment score exceeding -0.150 

### Observed trend 3
Out of the five media outlets analyzed, only tweets by the New York Times do not have negative aggregate sentiment


```python
# Dependencies

import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```

# Perform API Calls and Calculate Sentiments


```python
# Target User Accounts
target_user = ("@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes")

# Variables for holding sentiments and target user tweets
source_account_list = []
text_list = []
date_list = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
tweets_ago_list = []


# Variable for max_id
oldest_tweet = None

# Loop through each user
for user in target_user:
    
    # Counter
    counter = 1
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, max_id = oldest_tweet)

        # Loop through all tweets 
        for tweet in public_tweets:

            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            tweets_ago = counter
            
            # Add each value to the appropriate list
            source_account_list.append(user)
            text_list.append(tweet['text'])
            date_list.append(tweet['created_at'])
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            tweets_ago_list.append(counter)
        
            # Get Tweet ID, subtract 1, and assign to oldest_tweet
            oldest_tweet = tweet['id'] - 1
            
            
            # Add to counter 
            counter += 1
```


```python
# Create a dictionaty of results
sentiments_dict = {
                "Source Account": source_account_list,
                "Text": text_list,
                "Date": date_list,
                "Compound Score": compound_list,
                "Positive Score": positive_list,
                "Neutral Score": neutral_list,
                "Negative Score": negative_list,
                "Tweets Ago": tweets_ago_list
                }
        

# Create and print data frame
sentiments_pd = pd.DataFrame(sentiments_dict, columns = ['Source Account', 'Text', 'Date', 'Compound Score', 'Positive Score',
                                               'Neutral Score', 'Negative Score', 'Tweets Ago'])
sentiments_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source Account</th>
      <th>Text</th>
      <th>Date</th>
      <th>Compound Score</th>
      <th>Positive Score</th>
      <th>Neutral Score</th>
      <th>Negative Score</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCWorld</td>
      <td>'Treated like dogs': Italy's Roma minority on ...</td>
      <td>Fri Jun 29 00:40:19 +0000 2018</td>
      <td>0.3612</td>
      <td>0.217</td>
      <td>0.783</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCWorld</td>
      <td>Terrace House: Japan's nice, calm Love Island ...</td>
      <td>Fri Jun 29 00:27:10 +0000 2018</td>
      <td>0.8519</td>
      <td>0.608</td>
      <td>0.392</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCWorld</td>
      <td>Africa's week in pictures: 22-28 June 2018 htt...</td>
      <td>Fri Jun 29 00:10:52 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCWorld</td>
      <td>Annapolis shooting: How journalists tweeted th...</td>
      <td>Thu Jun 28 22:49:29 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCWorld</td>
      <td>RT @BBCBreaking: Several people dead in shooti...</td>
      <td>Thu Jun 28 20:44:06 +0000 2018</td>
      <td>-0.6486</td>
      <td>0.000</td>
      <td>0.815</td>
      <td>0.185</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save csv file
sentiments_pd.to_csv("News_Mood.csv", index=False)
```

# Sentiment Analysis of Media Tweets


```python
# Split up data into groups based on Source Account

bbc = sentiments_pd.loc[sentiments_pd["Source Account"] == "@BBCWorld"]
cbs = sentiments_pd.loc[sentiments_pd["Source Account"] == "@CBSNews"]
cnn = sentiments_pd.loc[sentiments_pd["Source Account"] == "@CNN"]
fox = sentiments_pd.loc[sentiments_pd["Source Account"] == "@FoxNews"]
nyt = sentiments_pd.loc[sentiments_pd["Source Account"] == "@nytimes"]

```


```python
# Create scatter plot for each Source Account
plt.scatter(bbc["Tweets Ago"], bbc["Compound Score"], marker="o", facecolor = "lightskyblue", edgecolors="black", label = "BBC", alpha = 0.5)
plt.scatter(cbs["Tweets Ago"], cbs["Compound Score"], marker="o", facecolor = "green", edgecolors ="black", label = "CBS", alpha = 0.5)
plt.scatter(cnn["Tweets Ago"], cnn["Compound Score"], marker="o", facecolor = "red", edgecolors = "black", label = "CNN", alpha = 0.5)
plt.scatter(fox["Tweets Ago"], fox["Compound Score"], marker="o", facecolor = "navy", edgecolors = "black", label = "FOX", alpha = 0.5)
plt.scatter(nyt["Tweets Ago"], nyt["Compound Score"], marker="o", facecolor = "gold", edgecolors = "black", label = "NYT", alpha = 0.5)


# Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%Y-%m-%d")

# Create Title with current date
plt.title(f"Sentiment Analysis of Media Tweets ({now})")

# Label and limit axes
plt.ylabel("Tweet Polarity")
plt.ylim(-1,1)
plt.xlabel("Tweets Ago")
plt.xlim(110, -10)

# Add legend
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, title="Media Sources")

# Display grid
plt.grid(True)

# Save an image of our chart and print the final product to the screen
plt.savefig("Images/Sentiment_Plot.png", bbox_inches="tight")
plt.show()

```


![png](output_10_0.png)


# Overall Media Sentiment Based on Twitter


```python
# Create bar chart for each Source Account
plt.bar(0, bbc["Compound Score"].mean(), color = "lightskyblue", edgecolor="black", width=1)
plt.bar(1, cbs["Compound Score"].mean(), color = "green", edgecolor ="black", width=1)
plt.bar(2, cnn["Compound Score"].mean(), color = "red", edgecolor="black", width=1)
plt.bar(3, fox["Compound Score"].mean(), color = "navy", edgecolor="black", width=1)
plt.bar(4, nyt["Compound Score"].mean(), color = "gold", edgecolor="black", width=1)

# Define x axis
media = ["BBC", "CBS", "CNN", "FOX", "NYT"]
x_axis = np.arange(len(media))

# Create the ticks for our bar chart's x axis
tick_locations = [value for value in x_axis]
plt.xticks(tick_locations, media)

# Give our chart some labels and a tile
plt.title(f"Overall Media Sentiment Based on Twitter ({now})")
plt.xlabel("Media Sources")
plt.ylabel("Tweet Polarity")

# Save an image of our chart and print the final product to the screen
plt.savefig("Images/Sentiment_Bar.png")
plt.show()
```


![png](output_12_0.png)

