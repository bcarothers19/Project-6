from bs4 import BeautifulSoup
import urllib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import metrics
import nltk
import string
from nltk.stem.porter import PorterStemmer
from imdbpie import Imdb
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

'''Part 1: Acquire the Data'''
# Read in the html from the specified url
url = 'http://www.imdb.com/chart/top?ref_=nv_mv_250_6'
html = urllib.urlopen(url)
# Create a BeautifulSoup object from the html
soup = BeautifulSoup(html, 'html.parser')
# Scrape the top 250 movies page
movies = []
for table in soup.findAll('tbody',{'class':'lister-list'}):
    for row in table.findAll('tr'):
        movie = []
        # Parse through the html to extract relevant information
        link = row.find('td',{'class':'titleColumn'}).find('a')['href']
        tconst = row.find('td',{'class':'titleColumn'}).find('a')['href'].split('/')[2]
        name = row.find('td',{'class':'titleColumn'}).find('a').renderContents()
        year = row.find('td',{'class':'titleColumn'}).find('span').renderContents()[1:5]
        rating = row.find('td',{'class':'ratingColumn imdbRating'}).find('strong').renderContents()
        votes = row.find('td',{'class':'ratingColumn imdbRating'}).find('strong')['title'].split(' ')[3]
        # Append values to the movie list
        movie.append(name)
        movie.append(year)
        movie.append(rating)
        movie.append(votes)
        movie.append(tconst)
        movie.append(link)
        # Append this movie to the list of movies
        movies.append(movie)

# Create a DataFrame from the movies list
cols = ['name','release_year','score','votes','id','link']
top250 = pd.DataFrame(movies,columns=cols)
# Add new rows for information we are going to scrape from individual pages
for new_row in ['rating','length','genre']:
    top250[new_row] = ''
# Create a new DataFrame containing only the top 100 movies
top100 = top250.loc[0:99,:]

# This function converts a string in the form of '_h __min' into minutes
def to_mins(length):
    split = length.strip().split()
    # Check if length includes hours and mins or just hours
    if len(split) > 1:
        hours = int(split[0][:-1])
        mins = int(split[1][:-3])
        return hours*60 + mins
    else: # Length only contains hours
        hours = int(split[0][:-1])
        return hours*60

# The base url, which we append each movie id to
base_url = 'http://www.imdb.com/title/'
# Go through each movie in top 100 and pull additional info
for num,movie in top100.iterrows():
    # Create the url and read the html with BeautifulSoup
    link = base_url + movie.id
    movie_html = urllib.urlopen(link)
    movie_soup = BeautifulSoup(movie_html, 'html.parser')
    # Parse through the html to extract relevant information
    info = movie_soup.find('div',{'class':'subtext'})
    length = info.find('time').renderContents()
    length_mins = to_mins(length)
    # Use RegEx to get the movie rating
    rating_long = str(info.meta)
    rating = re.search('"(.+?)"',rating_long).group(1)
    # There can be multiple genres so we create an empty list to hold them
    genres = []
    for genre in info.findAll('span',{'class':'itemprop','itemprop':'genre'}):
        genres.append(genre.renderContents())
    # Update the DataFrame with the information we just scraped
    top100.set_value(num,'rating',rating)
    top100.set_value(num,'length',length_mins)
    top100.set_value(num,'genre',genres)

# Add a column that we will use to merge this DF with the reviews DF below
top100['movie_num'] = range(100)

# Convert release_year to an int
top100.release_year = pd.to_numeric(top100.release_year, errors='coerce')

'''Part 2: Wrangle the Text Data'''
# Create an Imdb object from the imdbpie package
imdb = Imdb()
# Build a list of all reviews and ratings for each movie in the top 100
movie_reviews = []
for num,movie in top100.iterrows():
    reviews = imdb.get_title_reviews(movie.id,max_results=10000)
    for review in reviews:
        this_review = []
        this_review.append(num)
        this_review.append(review.rating)
        this_review.append(review.text)
        movie_reviews.append(this_review)

# Convert our list of reviews to a DataFrame
reviews_df = pd.DataFrame(movie_reviews,columns=['movie_num','review_score',
                                                'text'])

# Drop all reviews that didn't give a score (only has review text)
reviews_df.dropna(axis=0,how='any',inplace=True)

# This function removes all characters that aren't a letter, number, or space
def remove_nonalnum(s):
    return "".join([c for c in s if c.isalnum() or c.isspace()])
# This function replaces all new line characters with a space
def remove_newline(s):
    return re.sub('\n',' ',s)

# Remove the new line characters from all review texts
reviews_df['text'] = reviews_df['text'].apply(remove_newline)
# Remove non-alphanumeric (or space) characters from all review texts
reviews_df['text'] = reviews_df['text'].apply(remove_nonalnum)

# Initialize a stemmer that we will use when vectorizing our review texts
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems
# Initialize a TfidfVectorizer
stem_tvec = TfidfVectorizer(ngram_range=(1,2),stop_words='english',binary=False,
                        max_features=200,tokenizer=tokenize)
# Fit the TfidfVectorizer to our review texts
stem_tvec.fit(reviews_df.text.values)
# Transform our tokenized reviews into a DataFrame
reviews_vect = pd.DataFrame(stem_tvec.transform(reviews_df.text.values).todense(),
                    columns=stem_tvec.get_feature_names(),
                    index=reviews_df.index.values)

# Add a column to reviews_df and reviews_vect that we will use to join them
reviews_df['num'] = reviews_df.index
reviews_vect['num'] = reviews_vect.index
# Join our reviews DataFrames to match each tokenized reviews to the reviews
# text and score
reviews_joined = reviews_df.join(reviews_vect,on='num',rsuffix='numr')
# Drop the unneccasry columns that were used to join the DataFrames
reviews_joined.drop(['num','numnumr'],axis=1,inplace=True)
# Merge our reviews DataFrame with the top100 DataFrame to relate each reviews
# with the movie information for that review
movies_df = pd.merge(top100,reviews_joined,on='movie_num',how='inner')
# This function will remove the comma's from a string of numbers
def remove_comma(s):
    return re.sub(',','',s)
# Remove the commas from the number of votes column, and convert that column to
# have dtype int
movies_df.votes = movies_df.votes.apply(remove_comma)
movies_df.votes = pd.to_numeric(movies_df.votes,errors='coerce')

# Drop the column that contains the url link for each individual movie
movies_df.drop('link',axis=1,inplace=True)


'''Part 3: Exploratory Data Analysis'''
# We are only looking at the top 100 rated movies so a majority of our reviews
# are going to be extremely positive, which might cause issues with predicitng
# the score someone gave a movie based off of their review text.

# Over 52% of reviews people gave are 10/10, so the distribution of review
# scores is very skewed.

# Plot a histogram of review scores given by users
plt.hist(movies_df.review_score)
plt.xlim(1,10)
plt.xlabel('Review Score')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Review Scores')
plt.show()

# Plot a histogram of the top 100 movies release years
plt.hist(top100.release_year,bins=range(1920,2015,5))
plt.xlabel('Year Released (5 year bins)')
plt.ylabel('Number of Movies')
plt.title('Distribution of Top 100 Movies Release Year')
plt.show()

# Plot a histogram of genres in top 100
# Some movies have multiple genres, so the total won't = 100
# First we need to flatten the two dimensional list of genres
genres_flat = [x for sublist in top100.genre.values for x in sublist]
sns.countplot(genres_flat)
plt.xticks(rotation=90)
plt.title('Number of Movies in Top 100 by Genre')
plt.show()

# Plot a histogram of the length in minutes of the top 100 movies
plt.hist(top100.length,bins=range(60,230,10))
plt.xlabel('Movie Length (10 minute bins)')
plt.ylabel('Number of Movies')
plt.title('Distribution of Top 100 Movie Lengths')


'''Part 4: Decision Tree Clasifiers'''
# Set our target and features
# Our feature variables will be the word vector columns generated from reviews
X = movies_df.iloc[:,11:]
# Our target varible is the score the user gave the movie
y = movies_df.review_score

# Build and cross-validate a DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
cvscores = cross_val_score(dtc,X,y,cv=3,n_jobs=-1)
print 'Accuracy score for a simple decision tree classifier: %f \n' % cvscores.mean()

# This function will print out the best parameters and score for GridSearches
def grid_print(grid, model_type):
    print 'The best parameters for a %s: %s' % (model_type,str(grid.best_params_))
    print 'These parameters give a score of: %f \n' % grid.best_score_

# GridSearch the optimal parameters for our DecisionTreeClassifier
dtc_params = {'max_depth':range(1,10)}
grid = GridSearchCV(dtc,dtc_params,cv=5,n_jobs=-1,verbose=1)
grid.fit(X,y)
grid_print(grid, 'optimal decision tree classifier')
# We see that our accuracy improves from .369 to .523 when we use a RandomForest
# instead of just a DecisionTree


'''Part 5'''
# GridSearch the optimal parameters for a RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
params = {'n_estimators':range(5,16), 'max_depth':range(1,11)}
rfc_grid = GridSearchCV(rfc,params,cv=5,n_jobs=-1,verbose=1)
rfc_grid.fit(X,y)
grid_print(rfc_grid,'random forest classifier')

# GridSearch the optimal parameters for a ExtraTreesClassifier
etc = ExtraTreesClassifier(random_state=0)
etc_grid = GridSearchCV(etc,params,cv=5,n_jobs=-1,verbose=1)
etc_grid.fit(X,y)
grid_print(etc_grid,'extra trees classifier')

# GridSearch the optimal parameters for a AdaBoostClassifier
ada = AdaBoostClassifier(random_state=0)
ada_params = {'n_estimators':range(5,16)}
ada_grid = GridSearchCV(ada,ada_params,cv=5,n_jobs=-1,verbose=1)
ada_grid.fit(X,y)
grid_print(ada_grid,'AdaBoost classifier')

# We see that the AdaBoostClassifier performs slightly better than the two
# bagged classifiers

# Extract the feature importances from the Random Forest regressor and make
# a Series pairing variable names with their variable importances
rfc_best = RandomForestClassifier(n_estimators=10,max_depth=9,random_state=0)
rfc_best.fit(X,y)
importances = pd.Series(rfc_best.feature_importances_,index=X.columns.values)
# Sort the importances from most to least
importances.sort_values(ascending=False,inplace=True)
print 'The 10 most important features for predicting review score:'
print importances.head(10)
