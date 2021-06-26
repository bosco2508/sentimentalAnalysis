import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
url= "./AllProductReviews.csv"
df = pd.read_csv(url)
df[0:5000]

X = df['ReviewBody']
y= df['ReviewStar']

#change text to lower case and remov of white spaces
lower_text = []
for i in range(0,len(X)):
  s = str(X[i])
  s1 = s.strip()
  lower_text.append(s1.lower())

punc_text = []
for i in range(0,len(lower_text)):
  s2 = (lower_text[i])
  s3 = re.sub(r'[^\w\s2]',"",s2)
  punc_text.append(s3)

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,strip_accents='ascii', stop_words='english')
X_tfidf = vectorizer.fit_transform(punc_text)


X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=0)

print("Naive Bayes")
clf= naive_bayes.MultinomialNB()
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print('Confusion Matrix\n',confusion_matrix(Y_test,y_pred))
print('\n')
print('Classification Report\n',classification_report(Y_test,y_pred))
print('\n')
print('Accuracy : ',accuracy_score(Y_test,y_pred)*100)
print('\n')
print('\n')
print("------------------Support Vector Machine----------------------------------------------- ")
clf2= LinearSVC()
clf2.fit(X_train,Y_train)
y_pred2 = clf2.predict(X_test)
print('Confusion Matrix\n',confusion_matrix(Y_test,y_pred2))
print('\n')
print('Classification Report\n',classification_report(Y_test,y_pred2))
print('\n')
print('Accuracy : ',accuracy_score(Y_test,y_pred2)*100)
