#!/usr/bin/env python
# coding: utf-8

# # PYTHON MARKDOWN - YOUTUBE TRENDING VIDEOS DATA ANALYSIS
# 

# K HARINI REDDY - 160120771004
# NITYA NARLA - 160120771011

# # 1. Importing dataset and data preprocessing

# 
# 
# # 1.1. Importing essential libraries

# First, we import some Python packages that will help us analyzing the data, especially Pandas and Numpy for data analysis and matplotlib , Seaborn and Plotly for visualization.

# In[2]:



get_ipython().system('pip install plotly')


# In[3]:


import sys 
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

print('Python: {}'.format(sys.version[0:5]))
# for Statistical analysis
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
# for Data Visualization 
print('Seaborn: {}'.format(sns.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Plotly: {}'.format(plotly.__version__))


# # 1.2. Importing dataset

# In[4]:


in_videos = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')
in_videos_categories = pd.read_json(r'C:\Users\Raju\Downloads\IN_category_id.json')


# # 1.3. Let's Observe the dataset

# In[5]:


in_videos.head()


# In[6]:


f"The data set has {in_videos.shape[0]} rows of data with {in_videos.shape[1]} features."


# In[7]:


in_videos.columns


# In[8]:


in_videos.info()


# Now, from the above we can see that there are 20997 entries in the dataset. We can see also that all columns in the dataset are complete except description column which has some null values; it only has 20997 non-null values.

# # 1.4. Unique values of the dataset

# In[9]:


t = pd.DataFrame([[i,in_videos[i].dtype,in_videos[i].unique()] for i in in_videos.columns])
t.columns = ['name','dtype','unique']
t


# # 1.5. Data preprocessing and feature engineering

# # 1.5.1. Dataset collection years

# In[10]:


cdf = in_videos["trending_date"].apply(lambda x: '20' + x[:2]).value_counts()             .to_frame().reset_index()             .rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data=cdf, 
                palette=sns.color_palette(['#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")


# # 1.5.2. Removing Column 'Description'
# Summary and Titles are creating some issues with shifting the whole row into some next cell in the final csv output file. We are ignoring rows of data of that nature.

# In[11]:


in_videos["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)


# In[12]:


in_videos = in_videos.drop(['description'], axis = 1)


# # 1.5.3. Datetime format of Trending date and Publish time
# 
# Firstly we will transform trending_date as well as publish_time from string to datetime format. This will allow us to easily perform arithmetic operations and compare these values. publish_time column will be divided into three separate ones publish_date, publish_time and publish_hour 

# In[13]:


################################### Use only once (Fails after 1st Attempt) ##################################
# Transforming Trending date column to datetime format
in_videos['trending_date'] = pd.to_datetime(in_videos['trending_date'], format='%y.%d.%m').dt.date

# Transforming Trending date column to datetime format and splitting into two separate ones
publish_time = pd.to_datetime(in_videos['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
in_videos['publish_date'] = publish_time.dt.date
in_videos['publish_time'] = publish_time.dt.time
in_videos['publish_hour'] = publish_time.dt.hour


# In[14]:


in_videos.head()


# # 1.5.4. Addition of column 'category'
# Next we connect the category with the category_id they belong to. We will associate the information in two files: INvideos.csv and IN_category_id.json .

# In[15]:


# We'll use a very nice python featur - dictionary comprehension, to extract most important data from IN_category_id.json
categories = {category['id']: category['snippet']['title'] for category in in_videos_categories['items']}

# Now we will create new column that will represent name of category
in_videos.insert(4, 'category', in_videos['category_id'].astype(str).map(categories))
in_videos.tail(3)


# In[16]:


# dataset summary statistics - Categorical variables.
# T means Transpose.
in_videos.describe(include = ['O']).T


# # 1.5.5. Addition of column 'non-reactors'
# Addeing a new column of non reactors (i.e. who just viewed the video but not either liked or disliked it.)

# In[18]:


in_videos['non-reactors'] = in_videos.apply(lambda row: row.views - (row.likes + row.dislikes) , axis=1)

in_videos.head()


# # 1.5.6. Addition of column 'Days before trend'
# Now we create new feature days_before_trend representing the time (in days) between publication and the day when it became trending.

# Added a column days_befor_trend for which the data is obtained by substracting/deducting the publish_date from the trending_date and the result obtained will be added to the column days_befor_trend. Basically, it is a count of days the video took to get trending.

# In[19]:


in_videos["days_before_trend"] = (in_videos.trending_date - in_videos.publish_date) / np.timedelta64(1, 'D')
in_videos["days_before_trend"] = in_videos["days_before_trend"].astype(int)
in_videos.tail(3)


# # 1.5.7. Addition of column 'Views per day'

# In[20]:


in_videos.loc[(in_videos['days_before_trend'] < 1), 'days_before_trend'] = 1
in_videos["views_per_day"] = in_videos["views"].astype(int) / in_videos["days_before_trend"]
in_videos["views_per_day"] = in_videos["views_per_day"]
in_videos.tail(3)


# From the table above, we can see that there are 194 unique dates, which means that our dataset contains collected data about trending videos is vast.
# 
# From video_id count, we can see that there are 1,00,000 videos (which is expected because our dataset contains 100000 entries), but we can see also that there are only 16237 unique videos which means that some videos appeared on the trending videos list on more than one day.
# 
# The table also tells us that the top frequent title is पीरियड्स के समय, पेट पर पति करता ऐसा, देखकर दं... and that it appeared 91 times on the trending videos list.

# # 1.5.9. Number of words with all upper case in title (For Visualization Section)

# In[22]:


# Helper function
in_videos_first = in_videos.copy()

def numberOfUpper(string):
    i = 0
    for word in string.split():
        if word.isupper():
            i += 1
    return(i)

in_videos_first["all_upper_in_title"] = in_videos["title"].apply(numberOfUpper)
print(in_videos_first["all_upper_in_title"].tail(10))


# # 1.5.10. Distribution of basic parameters (For Visualization Section)

# In[23]:


in_videos_first['likes_log'] = np.log(in_videos['likes'] + 1)
in_videos_first['views_log'] = np.log(in_videos['views'] + 1)
in_videos_first['dislikes_log'] = np.log(in_videos['dislikes'] + 1)
in_videos_first['comment_log'] = np.log(in_videos['comment_count'] + 1)

plt.figure(figsize = (12,6))

plt.subplot(221)
g1 = sns.distplot(in_videos_first['views_log'],color='blue')
g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)

plt.subplot(222)
g4 = sns.distplot(in_videos_first['comment_log'],color='black')
g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)

plt.subplot(223)
g3 = sns.distplot(in_videos_first['dislikes_log'], color='r')
g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)

plt.subplot(224)
g2 = sns.distplot(in_videos_first['likes_log'],color='green')
g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()


# # 1.5.11. Duplicates (For Visualization Section)
# 
# Because many of the films have been trending you several times, we will create a separate datasets in which we will get rid of repetitions. Still, we leave the original dataset, because there is a lot of interesting information in it.

# In[24]:


in_videos_last = in_videos.drop_duplicates(subset=['video_id'], keep='last', inplace=False)
in_videos_first = in_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=False)
print(in_videos_last.head(2))


# As a part of Data Preprocessing and Data Cleaning, here we have deleted the duplicate data. So, now we will be having relevant data which would further help in better analysis.

# In[25]:


print("in_videos dataset contains {} videos".format(in_videos.shape[0]))
print("in_videos_first dataset contains {} videos".format(in_videos_first.shape[0]))
print("in_videos_last dataset contains {} videos".format(in_videos_last.shape[0]))


# # 1.5.12. Missing Value for Category Columns

# In[26]:


in_videos.isnull().sum()


# In[27]:


null_data = in_videos[in_videos["category"].isnull()]
null_data.head()


# In[28]:


in_videos["category"].fillna("Nonprofits & Activism", inplace = True) 
in_videos[in_videos["category_id"]  == 29]
in_videos[in_videos["category_id"]  == 29].tail(2)


# In[29]:


in_videos.isnull().sum()


# # 1.5.13. Outputing the file in CSV Format

# In[30]:


in_videos.to_csv('preprocessedIndia.csv',index=False)


# # 2. Data Visualization and Analysis

# # 2.1. Best time to publish video

# In[31]:


# Initialization of the list storing counters for subsequent publication hours
publish_h = [0] * 24

for index, row in in_videos_first.iterrows():
    publish_h[row["publish_hour"]] += 1
    
values = publish_h
ind = np.arange(len(values))


# Creating new plot
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.yaxis.grid()
ax.xaxis.grid()
bars = ax.bar(ind, values)

# Sampling of Colormap
for i, b in enumerate(bars):
    b.set_color(plt.cm.viridis((values[i] - min(values))/(max(values)- min(values))))
    
plt.ylabel('Number of videos that got trending', fontsize=20)
plt.xlabel('Time of publishing', fontsize=20)
plt.title('Best time to publish video', fontsize=35, fontweight='bold')
plt.xticks(np.arange(0, len(ind), len(ind)/6), [0, 4, 8, 12, 16, 20])

plt.show()


# In[32]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
in_videos.describe()


# The average number of views of a trending video is 10,60,582
# 
# The median value for the number of views is 2,97,434, which means that half the trending videos have views that are less than that number, and the other half have views larger than that number
# 
# The average number of likes of a trending video is 26,425, while the average number of dislikes is 1,506
# 
# The Average comment count is 2,552 while the median is 304

# # 2.3. Trending video titles contain capitalized word
# 
# Now we want to see how many trending video titles contain at least a capitalized word (e.g. HOW). To do that, we will add a new variable (column) to the dataset whose value is True if the video title has at least a capitalized word in it, and False otherwise

# In[33]:


# Helper function
def numberOfUpper(string):
    for word in string.split():
        if word.isupper():
            return True
    return False

in_videos["contains_capitalized"] = in_videos["title"].apply(numberOfUpper)

value_counts = in_videos["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized Word?')


# In[34]:


in_videos["contains_capitalized"].value_counts(normalize=True)


# We can see that 39% of trending video titles contain at least a capitalized word. 

# # 2.4. Trending video titles Lengths
# 
# Now we ddd another column to our dataset to represent the length of each video title, then plot the histogram of title length to get an idea about the lengths of trending video titles

# In[35]:


in_videos["title_length"] = in_videos["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(in_videos["title_length"], kde=False, rug=False, 
                  hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))


# We can see that title-length distribution resembles a normal distribution, where most videos have title lengths between 90 and 100 character approximately.
# 
# Now let's draw a scatter plot between title length and number of views to see the relationship between these two variables

# In[36]:


fig, ax = plt.subplots()
_ = ax.scatter(x=in_videos['views'], y=in_videos['title_length'], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Title Length")


# By looking at the scatter plot, we can say that there is no relationship between the title length and the number of views. However, we notice an interesting thing: videos that have 40,00,000 views and more have title length between 55 and 60 characters approximately.

# # 2.5. Correlation between dataset variables
# Now let's see how the dataset variables are correlated with each other: for example, we would like to see how views and likes are correlated, meaning do views and likes increase and decrease together (positive correlation)? Does one of them increase when the other decrease and vice versa (negative correlation)? Or are they not correlated?
# 
# Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation.
# 
# Now checking the correlation table between our dataset variables (numerical and boolean variables only)

# In[37]:


in_videos.corr()


# We see for example that views and likes are highly positively correlated with a correlation value of 0.85; we see also a high positive correlation (0.81) between likes and comment count, and between dislikes and comment count (0.81).
# 
# There is some positive correlation between views and dislikes, between views and comment count, between likes and dislikes.
# 
# Now let's visualize the correlation table above using a heatmap

# In[38]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(in_videos.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(in_videos.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# The correlation map and correlation table above say that views and likes are highly positively correlated. Now we can verify that by plotting a scatter plot between views and likes to visualize the relationship between these variables

# In[39]:


fig, ax = plt.subplots()
_ = plt.scatter(x=in_videos['views'], y=in_videos['likes'],  edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Likes")


# We see that views and likes are truly positively correlated: as one increases, the other increases too—mostly.

# # 2.9. Average time interval

# The average time interval for each category describes on average how fast a video can show up on the trending board. This is also a important criterion that which need to be cared about, because the longer time interval is, the larger the time cost will be.

# In[40]:


# Average time interval between published and trending
in_videos['interval'] = (pd.to_datetime(in_videos['trending_date']).dt.date - pd.to_datetime(in_videos['publish_date']).dt.date).astype('timedelta64[D]')
df_t = pd.DataFrame(in_videos['interval'].groupby(in_videos['category']).mean())
plt.figure(figsize = (42,12))
plt.plot(df_t, color='blue', linewidth=2)
plt.title("Average Days to be trending video", fontsize=28)
plt.xlabel('Category',fontsize=25)
plt.ylabel('Average Time Interval',fontsize=25)
plt.tick_params(labelsize=14)
plt.show();


# In[41]:


print(type(in_videos["video_id"]))


# # 2.11. Non-Reactor Viewers

# In[42]:


in_videos_first['likes_log'] = np.log(in_videos['likes'] + 1)
in_videos_first['dislikes_log'] = np.log(in_videos['dislikes'] + 1)
in_videos_first['comment_log'] = np.log(in_videos['comment_count'] + 1)
in_videos_first['non-reactors'] = np.log(in_videos['non-reactors'] + 1)

likes = in_videos_first['likes_log'].sum() / in_videos_first['likes_log'].count()
#print(likes)
dislikes = in_videos_first['dislikes_log'].sum() / in_videos_first['dislikes_log'].count()
#print(dislikes)
comments = in_videos_first['comment_log'].sum() / in_videos_first['comment_log'].count()
#print(comments)
non_reactors = in_videos_first['non-reactors'].sum() / in_videos_first['non-reactors'].count()
#print(non_reactors)

x = 'Likes', 'Dislikes', 'Comments' , 'Non-Reactors'
y = [likes,dislikes,comments,non_reactors]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
plt.pie(y, labels=x,colors=colors,autopct='%1.1f%%', startangle=140)
plt.show()


# Trending videos that have 41.0% Non-Reactors Viewers, 24.9% Likes, 17.1% Dislikes, 17.0% Comments

# # 2.12. Tags wordcloud
# This section dedicated to tags that support the videos to reach the trending list.

# In[44]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import urllib
import requests
import numpy as np
import matplotlib.pyplot as plt


mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))

# This function takes in your text and your mask and generates a wordcloud. 
def generate_wordcloud(mask):
    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(str(in_videos["tags"]))
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
#Run the following to generate your wordcloud
generate_wordcloud(mask)


# # 2.13. Videos have both comments and ratings disabled

# In[45]:


len(in_videos[(in_videos["comments_disabled"] == True) & (in_videos["ratings_disabled"] == True)].index)


# # 2.14 Showing the relation between the location and the trend

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
in_videos = pd.read_csv(r'C:\Users\Raju\Downloads\adsr.csv')

# Count the total number of entries in the dataframe
total_entries = len(in_videos)

# Extract the feature you want to plot
feature = in_videos['Location']

# Compute the frequency of each category in the feature
category_counts = feature.value_counts()

# Specify the width of each bar and the positions of the left edges of the bars
bar_width = 0.8
bar_positions = range(len(category_counts))

# Create a bar plot of the category frequencies with spaces between the bars
plt.bar(bar_positions, category_counts.values, width=bar_width, align='center')

# Set the x-axis labels to the category names
plt.xticks(bar_positions, category_counts.index)

# Set the x-axis label
plt.xlabel('Locations')

# Set the y-axis label
plt.ylabel('Videos')

# Show the plot
plt.show()


# # 2.15 Relation between the comment count and the trending videos

# In[47]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas dataframe

in_videos = pd.read_csv(r'C:\Users\Raju\Downloads\adsr.csv')

# Count the total number of entries in the dataframe
total_entries = len(in_videos)

# Extract the feature you want to plot
feature = in_videos['comment_count']

# Plot the feature with respect to the total number of entries
plt.plot(feature, range(total_entries))

# Set the x-axis label
plt.xlabel('Comments Count')

# Set the y-axis label
plt.ylabel('Videos')

# Show the plot
plt.show()


# # 2.16 If Sheild ( creator award ) is affecting the trend

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
in_videos = pd.read_csv(r'C:\Users\Raju\Downloads\adsr.csv')

# Count the total number of entries in the dataframe
total_entries = len(in_videos)

# Extract the feature you want to plot
feature = in_videos['sheild']

# Compute the frequency of each category in the feature
category_counts = feature.value_counts()

# Specify the width of each bar and the positions of the left edges of the bars
bar_width = 0.8
bar_positions = range(len(category_counts))

# Create a bar plot of the category frequencies with spaces between the bars
plt.bar(bar_positions, category_counts.values, width=bar_width, align='center')

# Set the x-axis labels to the category names
plt.xticks(bar_positions, category_counts.index)

# Set the x-axis label
plt.xlabel('Sheild(Creator Award)')

# Set the y-axis label
plt.ylabel('Videos')

# Show the plot
plt.show()


# # 2.16 Displaying the various channel names and the number of trending videos that were there

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
in_videos = pd.read_csv(r'C:\Users\Raju\Downloads\adsr.csv')

# Count the total number of entries in the dataframe
total_entries = len(in_videos)

# Extract the feature you want to plot
feature = in_videos['channel_title']

# Compute the frequency of each category in the feature
category_counts = feature.value_counts()

# Specify the width of each bar and the positions of the left edges of the bars
bar_width = 0.8
bar_positions = range(len(category_counts))

# Create a bar plot of the category frequencies with spaces between the bars
plt.bar(bar_positions, category_counts.values, width=bar_width, align='center')

# Set the x-axis labels to the category names
plt.xticks(bar_positions, category_counts.index)

# Set the x-axis label
plt.xlabel('Channel Name')

# Set the y-axis label
plt.ylabel('Videos')

# Show the plot
plt.show()


# # 2.16.1 Top 5 channels

# In[50]:


import pandas as pd
import numpy as np
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
top_5 = in_videos.channel_title.value_counts().iloc[:5]
sns.countplot(x = "channel_title", data = in_videos, order=top_5.index);


# # 2.17 Top 6 Locations with the highest trending videos

# In[51]:


top_6 = in_videos.Location.value_counts().iloc[:6]
sns.countplot(x = "Location", data = in_videos, order=top_6.index);


# In[52]:


top_13 = in_videos.Location.value_counts().iloc[7:13]
sns.countplot(x = "Location", data = in_videos, order=top_13.index);


# # 2.18 Top 5 Categories

# In[55]:


# We'll use a very nice python featur - dictionary comprehension, to extract most important data from IN_category_id.json
categories = {category['id']: category['snippet']['title'] for category in in_videos_categories['items']}

# Now we will create new column that will represent name of category
in_videos.insert(4, 'category', in_videos['category_id'].astype(str).map(categories))
in_videos.tail(3)


# In[56]:


top_5 = in_videos.category.value_counts().iloc[0:5]
sns.countplot(x = "category", data = in_videos, order=top_5.index);


# # 2.18 Relation between comments disabled count, rating disabled count, sheild, video removed count, category id's 

# In[57]:


for x in (['comments_disabled','ratings_disabled','sheild','video_error_or_removed','category_id']):
    count=in_videos[x].value_counts()
    print(count)
    plt.figure(figsize=(7,7))
    sns.barplot(count.index, count.values, alpha=0.8)
    plt.title('{} vs No of video'.format(x))
    plt.ylabel('No of video')
    plt.xlabel('{}'.format(x))
    plt.show()


# # 3.Conclusions

# Here are the some of the results we extracted from the analysis:
# 
# -We analyzed a dataset that contains 1,00,000 videos entry.
# -Trending videos that have 41.0% Non-Reactors Viewers, 24.9% Likes, 17.1% Dislikes, 17.0% Comments
# -Some videos may appear on the trending videos list on more than one day. 
# -The words 'New', 'Song', 'Wedding', 'Movie', and 'Sharry' were common also in trending video tags.
# -There is a strong positive correlation between the number of views and the number of likes of trending videos: As one of them increases, the other increases, and vice versa.
# -There is a strong positive correlation also between the number of likes and the number of comments, and a slightly weaker one between the number of dislikes and the number of comments.
# -The category that has the largest number of trending videos is 'Entertainment' with 87.712k videos, followed by 'News & Politics' category with 3,240 videos, followed by 'Music' category with 2019 videos.
# -On the opposite side, the category that has the smallest number of trending videos is 'Nonprofits & Activisim' with 155 videos, followed by 'Animals' with 105 videos

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\preprocessedIndia.csv')

# Select the features and target variable
X = data[['views', 'likes', 'dislikes', 'comment_count']]
y = data['category_id']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier and define the hyperparameter grid
clf = DecisionTreeClassifier()
param_grid = {'max_depth': [2, 4, 6, 8, 10]}

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Use the best hyperparameters to create a new decision tree classifier
clf = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'])
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[14]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the YouTube trending videos dataset
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Select the relevant features
X = df[['category_id', 'views']]

# Scale the features using Min-Max normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering with k=5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# Print the mean values of each feature for each cluster
print(df.groupby('cluster').mean())


# In[16]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the YouTube trending videos dataset
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Select the relevant features
X = df[['category_id', 'views']]

# Scale the features using Min-Max normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering with k=5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# Define marker shapes and sizes for each cluster
markers = ['o', 's', '^', 'D', 'X']
sizes = [30, 70, 100, 130, 160]

# Create a scatter plot for each cluster
for i in range(5):
    plt.scatter(df.loc[df['cluster']==i, 'category_id'], 
                df.loc[df['cluster']==i, 'views'], 
                marker=markers[i], s=sizes[i], 
                label='Cluster {}'.format(i+1))

# Add axis labels and a title to the plot
plt.xlabel('Category ID')
plt.ylabel('Views')
plt.title('K-Means Clustering of YouTube Trending Videos')

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv' , nrows=10000)

# Extract the 'views' and 'category_id' features from the dataset
X = data[['views', 'category_id']].values

# Initialize and fit the DBSCAN clustering model
dbscan = DBSCAN(eps=50000, min_samples=10)
clusters = dbscan.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
plt.xlabel('Views')
plt.ylabel('Category ID')
plt.title('DBSCAN Clustering of YouTube Trending Videos')
plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Load a subset of the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv', nrows=10000)

# Extract the 'views' and 'category_id' features from the dataset
X = data[['views', 'category_id']].values

# Initialize and fit the Agglomerative Clustering model
agg_clustering = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clusters = agg_clustering.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
plt.xlabel('Views')
plt.ylabel('Category ID')
plt.title('Agglomerative Clustering of YouTube Trending Videos')
plt.show()


# In[17]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the YouTube trending videos dataset
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Select the feature to use as the predictor variable (X) and the target variable (y)
X = df[['likes']]
y = df['views']

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X, y)

# Print the coefficients and intercept of the linear regression line
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)


# In[19]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the YouTube trending videos dataset
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Select the feature to use as the predictor variable (X) and the target variable (y)
X = df[['likes']]
y = df['views']

# Perform linear regression
regressor = LinearRegression()
regressor.fit(X, y)

# Create a scatter plot of the data points
plt.scatter(X, y, color='blue', alpha=0.5, label='Data Points')

# Create the predicted values for the regression line
y_pred = regressor.predict(X)

# Plot the regression line
plt.plot(X, y_pred, color='red', label='Linear Regression Line')

# Add axis labels and a title to the plot
plt.xlabel('Likes')
plt.ylabel('Views')
plt.title('Linear Regression of YouTube Trending Videos')

# Add a legend to the plot
plt.legend(loc='upper left')

# Display the plot
plt.show()


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Split the data into training and testing sets
X = df[['views']].values
y = df['likes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform polynomial regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

# Predict likes for test data
y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

# Visualize the results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg.predict(poly_reg.fit_transform(X_train)), color='blue')
plt.title('Likes vs Views (Polynomial Regression)')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.show()


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Preprocess the data
X = data['views'].values.reshape(-1, 1)
y = data['likes'].values.reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the ridge regression model
model = Ridge(alpha=0.1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the coefficient of determination (R^2) on the test data
r_squared = model.score(X_test, y_test)
print(f'R^2: {r_squared:.2f}')

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('views')
plt.ylabel('likes')
plt.title('Ridge Regression')
plt.show()


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Preprocess the data
X = data['views'].values.reshape(-1, 1)
y = data['likes'].values.reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the lasso regression model
model = Lasso(alpha=0.1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the coefficient of determination (R^2) on the test data
r_squared = model.score(X_test, y_test)
print(f'R^2: {r_squared:.2f}')

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('views')
plt.ylabel('likes')
plt.title('Lasso Regression')
plt.show()


# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Extract the 'views' and 'likes' features from the dataset
X = data[['likes', 'dislikes', 'comment_count']].values
y = data['views'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree Regression model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the views for the testing set
y_pred = model.predict(X_test)

# Evaluate the model using R^2 score
r2_score = model.score(X_test, y_test)
print('R^2 Score:', r2_score)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=model.feature_importances_, y=['Likes', 'Dislikes', 'Comment Count'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Decision Tree Regression of YouTube Trending Videos')
plt.show()


# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Extract the 'views', 'likes', 'dislikes', and 'comment_count' features from the dataset
X = data[['likes', 'dislikes', 'comment_count']].values
y = data['views'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the views for the testing set
y_pred = model.predict(X_test)

# Evaluate the model using R^2 score
r2_score = model.score(X_test, y_test)
print('R^2 Score:', r2_score)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=model.feature_importances_, y=['Likes', 'Dislikes', 'Comment Count'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Regression of YouTube Trending Videos')
plt.show()


# In[31]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Extract the 'category_id' feature from the dataset
X = data[['category_id']].values
y = data['title'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict the categories for the testing set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score
accuracy_score = model.score(X_test, y_test)
print('Accuracy Score:', accuracy_score)

# Plot the distribution of categories
plt.figure(figsize=(10, 8))
sns.histplot(data, x='category_id', hue='title')
plt.xlabel('Category ID')
plt.ylabel('Count')
plt.title('K-Nearest Neighbors of YouTube Trending Videos')
plt.show()


# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each video title
data['sentiment'] = data['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Visualize the distribution of sentiment scores
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='sentiment', bins=20)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores for YouTube Video Titles')
plt.show()


# In[ ]:


#Clustering of videos based on likes, dislikes, views, and comment count using K-means:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the YouTube trending videos dataset
data = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')

# Extract the 'category_id' feature from the dataset
X = data[['category_id']].values
y = data['title'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict the categories for the testing set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score
accuracy_score = model.score(X_test, y_test)
print('Accuracy Score:', accuracy_score)

# Plot the distribution of categories
plt.figure(figsize=(10, 8))
sns.countplot(data=data, x='category_id', hue='title')
plt.xlabel('Category ID')
plt.ylabel('Count')
plt.title('Distribution of Video Categories in the YouTube Trending Videos Dataset')
plt.show()


# In[23]:


import pandas as pd
get_ipython().system('pip install lifelines')
import lifelines
from lifelines import CoxPHFitter

# Load your YouTube trending video dataset (assuming it's in a CSV file)
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')
# Feature selection and engineering
selected_features = ['likes', 'dislikes', 'sheild', 'Location', 'views', 'comment_count', 'comments_disabled']

# Select only the relevant columns
df = df[selected_features]

# Convert the 'sheild' column to numerical values (1 for 'Yes', 0 for 'No')
df['sheild'] = df['sheild'].map({'Yes': 1, 'No': 0})

# Assuming 'Location' is a categorical feature, you can one-hot encode it
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Create a dummy duration column with a constant value
df['duration'] = 1

# Create a DataFrame with 'event' as a binary variable (trending or not)
# Assuming you have a 'trending_date' column representing the date when the video became trending
df['event'] = 1  # Videos that became trending are considered as events

# Fit a Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event')

# Take input from the user
likes = int(input("Enter the number of likes: "))
dislikes = int(input("Enter the number of dislikes: "))
sheild = int(input("Is the video sheilded? (1 for yes, 0 for no): "))
Location = input("Enter the video location (e.g., 'California'): ")
views = int(input("Enter the number of views: "))
comment_count = int(input("Enter the comment count: "))
comments_disabled = int(input("Are comments disabled? (1 for yes, 0 for no): "))

# Create a DataFrame with user input
input_features = pd.DataFrame({
    'likes': [likes],
    'dislikes': [dislikes],
    'sheild': [sheild],
    'duration': [1]  # Dummy duration
})

# Check if the one-hot encoded location column exists before adding it
location_column = 'Location_' + Location
if location_column in df.columns:
    input_features[location_column] = 1
else:
    # If the location doesn't exist in the dataset, assume it's not a significant factor
    print(f"Warning: Location '{Location}' not found in the dataset. Location will not be considered in prediction.")

# Predict the likelihood of trending for the user input
predicted_likelihood = cph.predict_expectation(input_features)

print(f'Predicted Likelihood of Trending: {predicted_likelihood.values[0]:.2f}')


# In[3]:


import pandas as pd
from lifelines import CoxPHFitter

df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')


# Convert the 'trending_date' and 'publish_time' columns to datetime objects without timezones
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m').dt.tz_localize(None)
df['publish_time'] = pd.to_datetime(df['publish_time']).dt.tz_localize(None)

# Calculate the duration to trending in days
df['duration'] = (df['trending_date'] - df['publish_time']).dt.days

# Feature selection and engineering
selected_features = ['likes', 'dislikes', 'ratings_disabled', 'duration']

# Create a new DataFrame with selected features
df_selected = df[selected_features]

# Fit a Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_selected, duration_col='duration')

# Take input from the user
likes = int(input("Enter the number of likes: "))
dislikes = int(input("Enter the number of dislikes: "))
ratings_disabled = input("Is ratings disabled? (True/False): ").capitalize() == 'True'
duration = int(input("Enter the duration (in days): "))

input_features = {
    'likes': likes,
    'dislikes': dislikes,
    'ratings_disabled': ratings_disabled,
    'duration': duration
}

# Predict the likelihood of trending for the user input
predicted_likelihood = cph.predict_expectation(pd.DataFrame([input_features]))

print(f'Predicted Likelihood of Trending: {predicted_likelihood.values[0]:.2f}')


# In[ ]:





# In[14]:


import pandas as pd
from lifelines import CoxPHFitter

df = pd.read_csv(r'C:\Users\Raju\Downloads\dsrmp.csv')
selected_features = ['likes', 'dislikes', 'sheild', 'views', 'comment_count', 'comments_disabled']
df = df[selected_features]

df['sheild'] = df['sheild'].map({'Yes': 1, 'No': 0})

df['duration'] = 1
df['event'] = 1  

cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event')

likes = int(input("Enter the number of likes: "))
dislikes = int(input("Enter the number of dislikes: "))
sheild = int(input("Is the video shielded? (1 for yes, 0 for no): "))
views = int(input("Enter the number of views: "))
comment_count = int(input("Enter the comment count: "))
comments_disabled = int(input("Are comments disabled? (1 for yes, 0 for no): "))

input_features = pd.DataFrame({
    'likes': [likes],
    'dislikes': [dislikes],
    'sheild': [sheild],
    'views': [views],
    'comment_count': [comment_count],
    'comments_disabled': [comments_disabled],
    'duration': [1]
})

predicted_likelihood = cph.predict_survival_function(input_features)
percentage_likelihood = (1 - predicted_likelihood.iloc[0]) * 100

print(f'Predicted Likelihood of Trending: {percentage_likelihood.values[0]:.2f}%')


# In[ ]:


import transformers
import pandas as pd

# Load the BERT model
bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')

# Load the YouTube trending video data
df = pd.read_csv('youtube_trending_video_data.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['title', 'description', 'tags']], df['views'], test_size=0.25, random_state=42)

# Create a BERT regressor
bert_regressor = transformers.BertForSequenceRegression.from_pretrained('bert-base-uncased')

# Train the regressor on the training data
bert_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = bert_regressor.predict(X_test)

# Evaluate the regressor performance
print('R-squared score:', r2_score(y_test, y_pred))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the YouTube trending video data
df = pd.read_csv('youtube_trending_video_data.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['title', 'description', 'tags']], df['category_id'], test_size=0.25, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training and test data into TF-IDF vectors
X_train_tfidf = vectorizer.fit_transform(X_train['title'])
X_test_tfidf = vectorizer.transform(X_test['title'])

# Create a logistic regression classifier
clf = LogisticRegression()

# Train the classifier on the training data
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier performance
print(classification_report(y_test, y_pred))


# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a synthetic dataset
np.random.seed(42)
data = {'X': np.random.randn(100), 'Y': 2 * np.random.randn(100) + 3, 'Category': np.random.choice(['A', 'B'], size=100)}
df = pd.DataFrame(data)

# Descriptive Statistics
mean_X = np.mean(df['X'])
median_Y = np.median(df['Y'])
variance_X = np.var(df['X'])
std_dev_Y = np.std(df['Y'])

print(f"Mean of X: {mean_X:.2f}")
print(f"Median of Y: {median_Y:.2f}")
print(f"Variance of X: {variance_X:.2f}")
print(f"Standard Deviation of Y: {std_dev_Y:.2f}")

# Hypothesis Testing (t-test)
sample1 = df[df['Category'] == 'A']['X']
sample2 = df[df['Category'] == 'B']['X']
t_stat, p_value = stats.ttest_ind(sample1, sample2)

print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Linear Regression
X = df['X']
X = sm.add_constant(X)
y = df['Y']
model = sm.OLS(y, X).fit()
intercept, slope = model.params

print(f"Linear Regression - Intercept: {intercept:.2f}, Slope: {slope:.2f}")
print(model.summary())

# Chi-Square Test
observed = pd.crosstab(df['Category'], columns='count')
chi2, p, _, _ = stats.chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")

if p < 0.05:
    print("Reject the null hypothesis (variables are dependent)")
else:
    print("Fail to reject the null hypothesis (variables are independent)")

# Data Visualization
plt.figure(figsize=(10, 5))
sns.scatterplot(x='X', y='Y', data=df, hue='Category')
plt.title("Scatterplot of X and Y by Category")
plt.show()

