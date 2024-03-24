from data import atributos, resultados
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

X = np.array(atributos)
y = np.array(resultados)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# KNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Test accuracy of models
nb_accuracy = gaussian_nb.score(X_test, y_test)
dt_accuracy = dt_classifier.score(X_test, y_test)
knn_accuracy = knn_classifier.score(X_test, y_test)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("KNN Accuracy:", knn_accuracy)

# Save models
joblib.dump(gaussian_nb, './gaussian_nb_model.pkl')
joblib.dump(dt_classifier, './dt_classifier_model.pkl')
joblib.dump(knn_classifier, './knn_classifier_model.pkl')

print("Models saved successfully.")
