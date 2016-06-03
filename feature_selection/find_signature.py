import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
### these files should have been created from the previous (Lesson 10) mini-project.
words_file = "./your_word_data.pkl"
authors_file = "./your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )

### test_size is the percentage of events assigned to the test set (remainder go into training)
### feature matrices changed to dense representations for compatibility with classifier
### functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
print ("signiture word: " + str(vectorizer.get_feature_names()[21323]))

### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime

features_train = features_train[:150].toarray()
print("total train: " + str(len(labels_train)))
labels_train   = labels_train[:150]

print("Used train: " + str(len(features_train)))

### your code goes here
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)
print("Get accuracy score: " + str(accuracy_score(pred, labels_test)))


importances = clf.feature_importances_
importance_list = []
index = 0
num =[]
for feature in importances:
	if feature >=0.2:
		importance_list.append(feature)
		num.append(index)
	index += 1

print (importance_list, num)
