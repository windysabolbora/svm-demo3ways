#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    st.title('Logistic Regression, Naive Bayes Classifiers and Support Vector Machine')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
 
    st.write('Logistic Regression:')
    text = """Strengths: \nMore flexible: Can capture complex relationships between 
    features and classes, even when they are non-linear. No strong independence assumption: 
    Doesn't rely on the assumption that features are independent, which can be 
    helpful for overlapping clusters."""
    st.write(text)
    text = """Weaknesses: \nOverfitting potential: Can overfit the training data when 
    dealing with high dimensionality 
    or small datasets."""
    st.write(text)

    st.write('Naive Bayes')
    text = """Strengths: \nEfficient: Works well with high-dimensional datasets 
    due to its simplicity. 
    Fast training: Requires less training time compared to logistic regression. 
    Interpretable: Easy to understand the contribution of each feature to the prediction."""
    st.write(text)

    text = """Weaknesses:\nIndependence assumption: Relies on the strong 
    assumption of feature independence, which can be violated in overlapping clusters, 
    leading to inaccurate predictions."""
    st.write(text)

    st.write('Support Vector Machine')
    st.write("""Strong in complex, high-dimensional spaces, 
             but computationally expensive.""")

    st.write("""Strengths: Handles high dimensions, maximizes separation, efficient memory use, 
              and offers some non-linearity through kernels. Weaknesses: Computationally 
              demanding, can be difficult to interpret, and requires careful parameter tuning. 
              SVMs are powerful for complex problems, but their efficiency and 
              interpretability need consideration.""")

    # Create a slider with a label and initial value
    n_samples = st.slider(
        label="Number of samples (200 to 4000):",
        min_value=200,
        max_value=4000,
        step=200,
        value=1000,  # Initial value
    )

    cluster_std = st.number_input("Standard deviation (between 0 and 1):")

    random_state = st.slider(
        label="Random seed (between 0 and 100):",
        min_value=0,
        max_value=100,
        value=42,  # Initial value
    )
   
    n_clusters = st.slider(
        label="Number of Clusters:",
        min_value=2,
        max_value=6,
        value=2,  # Initial value
    )

    # Create the selecton of classifier
    clf = GaussianNB() 
    options = ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine']
    selected_option = st.selectbox('Select the classifier', options)
    if selected_option =='Logistic Regression':
        clf = LogisticRegression(C=1.0, class_weight=None, 
            dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='auto',
            n_jobs=1, penalty='l2', random_state=42, solver='lbfgs',
            tol=0.0001, verbose=0, warm_start=False)
    elif selected_option=='Support Vector Machine':
        clf = svm.SVC(kernel='linear', C=1000)
    else:
        clf = GaussianNB()
        
    if st.button('Start'):
        centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                    cluster_std=cluster_std, centers = centers,
                    random_state=random_state)       
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=42)
        
        clf.fit(X_train,y_train)
        y_test_pred = clf.predict(X_test)
        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))
        st.subheader('VIsualization')
        visualize_classifier(clf, X_test, y_test_pred)
        st.session_state['new_cluster'] = False

def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Specify the title
    ax.set_title(title)
    
    # Choose a color scheme for the plot
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    
    # Overlay the training points on the plot
    ax.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # Specify the boundaries of the plot
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    
    # Specify the ticks on the X and Y axes
    ax.set_xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    ax.set_yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    
    st.pyplot(fig)

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points

#run the app
if __name__ == "__main__":
    app()
