import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


# declare variables
posData = []
negData = []
posRevData = []
negRevData = []
Reviews = []
outputColumn = []

# Directories of reviews
posRevDir = os.path.dirname(os.path.abspath("__file__")) + '\\txt_sentoken\\pos\\'
negRevDir = os.path.dirname(os.path.abspath("__file__")) + '\\txt_sentoken\\neg\\'



# read the positive reviews
for file in os.listdir(posRevDir):
    f = open(posRevDir + file)
    data = f.read()
    posData.append(data)
    outputColumn.append(1)



# read the negative reviews
for file in os.listdir(negRevDir):
    f = open(negRevDir + file)
    data = f.read()
    negData.append(data)
    outputColumn.append(0)


# remove stopwords from positive reviews
stopWords = set(stopwords.words("english"))
for review in posData:
    reviewTokenized = word_tokenize(review)
    review_without_sw = [word for word in reviewTokenized if not word in stopWords]
    newReview = (" ").join(review_without_sw)
    posRevData.append(newReview)



# remove stopwords from negative reviews
stopWords = set(stopwords.words("english"))
for review in negData:
    reviewTokenized = word_tokenize(review)
    review_without_sw = [word for word in reviewTokenized if not word in stopWords]
    newReview = (" ").join(review_without_sw)
    negRevData.append(newReview)


# all reviews
Reviews = posRevData + negRevData



# tfidf vectors
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
tfidf_vectorized_vectors = tfidf_vectorizer.fit_transform(Reviews)
first = tfidf_vectorized_vectors[0]


#df1 = pd.DataFrame(first.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=['tfidf'])
#print(df1)


df2 = pd.DataFrame(tfidf_vectorized_vectors.todense(), columns=tfidf_vectorizer.get_feature_names())
print(df2)

# add the lable column
df2['lable'] = outputColumn

# shuffle the data
df2 = df2.sample(frac=1).reset_index(drop=True)



# prepare the model for training
X = df2[tfidf_vectorizer.get_feature_names()]
Y = df2['lable']
model = LinearSVC()
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.20, random_state=42)


# Start Training
model.fit(X_train,y_train)

# calculate the accuracy of the model on testing data
predicted = model.predict(X_test)
print(f"the accuracy of the model after testing it on the testing set = {np.mean(predicted == y_test) * 100}")


#predicted = model.predict(X_train)
#np.mean(predicted == y_train)


#userInput = input(f"Please Enter a Review to classify it : ")
userInput = open('input.txt', 'r').read()
#tmp = ["after watching _a_night_at_the_roxbury_ , you'll be left with exactly the same"]
X_new_tfidf = tfidf_vectorizer.transform([userInput])


predicted_test = model.predict(X_new_tfidf)
#print(userInput)
print(f"\n\nThe predicted class = {predicted_test[0]}")


# def predict(m, review):
#     tmp = tfidf_vectorizer.transform([review])
#     tmp2 = m.predict(tmp)
#     print(f"The predicted class = {tmp2[0]}")
#
# predict(model,
#         "baldwin seems more interested in parillaud's nest egg ( so that he can pave paradise and put up a parking lot ) than he does in her . ")
# predict(model, "after watching _a_night_at_the_roxbury_ , you'll be left with exactly the same . ")
# predict(model, "this film is beautiful")
# predict(model, "this film is trash")
# predict(model, "damn that y2k bug .")
# predict(model, open('txt_sentoken/neg/cv584_29549.txt', 'r').read())
# predict(model, open('txt_sentoken/neg/cv213_20300.txt', 'r').read())
# predict(model, open('txt_sentoken/pos/cv001_18431.txt', 'r').read())
# predict(model, open('txt_sentoken/pos/cv014_13924.txt', 'r').read())