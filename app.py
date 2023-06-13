import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def iris_classification():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Iris Classification")
    st.write("Accuracy:", accuracy)

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def kmeans_clustering():
    data = np.random.randn(100, 2)

    k = st.slider("Select the number of clusters", min_value=1, max_value=10)

    centroids = initialize_centroids(data, k)
    kmeans = KMeans(n_clusters=k, init=centroids, random_state=42)
    kmeans.fit(data)
    labels = kmeans.predict(data)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
    plt.title("k-means Clustering_1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot()


    st.subheader("Cluster Centers:")
    st.write(centroids)

    st.subheader("Assigned Labels:")
    st.write(labels)
    
def main():
    st.title("Machine Learning Examples")

    iris_classification()

    kmeans_clustering()

if __name__ == '__main__':
    main()
