import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from flask import Flask, render_template, request, jsonify
import boto3
import botocore

app = Flask(__name__)

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
print(data.head(5))
classification = {"ham":0,"spam":1}
data['label']=data['label'].map(classification)

X_train, X_test, y_train, y_test = train_test_split(data['message'],data['label'],test_size=0.2)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(counts,y_train)

'''
counts_test = vectorizer.transform(X_test)
prediction = classifier.predict(counts_test)
print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
print('Precision score: {}'.format(precision_score(y_test, prediction)))
print('Recall score: {}'.format(recall_score(y_test, prediction)))
print('F1 score: {}'.format(f1_score(y_test, prediction)))
prediction'''

counts_test = vectorizer.transform(['free'])
predicted= classifier.predict(counts_test)
print(predicted)

if predicted==[1]:
    print('Spam')
else:
    print('Not Spam')

@app.route('/')
def index():
	print('check1')
	return render_template('gui.html')

@app.route('/', methods=['POST'])
def hello():
	print('check2')
	msg = request.form['text']
	counts_test = vectorizer.transform([msg])
	predicted= classifier.predict(counts_test)
	print(msg,predicted)

	if predicted==[1]:
		op='Spam'
		print(op)
	else:
		op='Not Spam'
		print(op)
	return render_template('gui2.html', name = op)

if __name__ == '__main__':
   app.run(port = 5000, debug=True)