# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
options = ['Recommender systems', 'Additional Movie Information', 'Data Behaviour(EDA)']
selection = st.sidebar.selectbox("Choose Option", options)

if selection == 'Recommender systems':
    st.title('The Recommender System')
    from PIL import Image
    img = Image.open('learn.jpeg')
    st.image(img, width = 500, caption = 'Data segmentation')
    st.subheader('A Closer Look At Recommender System Models')
    system = st.radio('Choose A Recommender Systems:',('SVD', 'Neural Networks', 'KNN'))
    if system == ('SVD'):
        st.write('Singular Value Decomposition(SVD), is a linear algerbra concept')
        st.write('SVD is a type of matrix decomposition and describes a matrix using its constituent elements. This method is popularly used for compressing, denoising and data reduction.')
        st.write('A = U . Sigma . V^T')
        st.write('A : real matrix m * n')
        st.write('U: matrix m * m')
        st.write('V^T transposed n * n matrix')
        st.write('The Sigma matrix are the singular values of  matrix A. The columns of the U matrix are the left singular vectors; columns of V are the right singular vectors of A.')
        st.write('Iterative numerical methods are used to calculate SVD and every rectangular matrix has a singular value decomposition. SVD is used in the calculation of other matrix operations such as matrix inverse and data reduction, least squares linear regression, image compression and data denoising. The function can be called using the svd() function, which takes in a matrix and returns U, Sigma(as a vector), V^T(transposed matrix)')
        st.write('The svd function is also popularly used for dimensionality reduction where the number of features are greater than the number of observations, the function reduces the data to a smaller subset that are more relevant to the prediction. This is done by selecting the largest singular values in Sigma for the columns and the V^T for the rows. ')
        img1 = Image.open('svd.true.jpg')
        st.image(img1, width = 500)
    if system == ('Neural Networks'):
        st.write(' Neural networks were first proposed in 1944 by Warren McCullough and Walter Pitts and are loosely modelled on the human brain. A Neural network consists of thousands or million of simple processing nodes that are densely interconnected.')
        st.write('Neural networks consists of nodes that feed forward (data moves in one direction). Each node is likely connected to several nodes in the layer beneath it, to which it sends data. For each incoming connection, a node will assign a number known as a weight. Once the node is active it will receive data, the data will be multiplied by the weight, it will then add the resulting products together yielding a single number.')
        st.write('If the data exceeds the threshold value, the node fires, and the adjusted data is sent to the next node in the line.')
        st.write('The weights and thresholds are initially set to random values, the training begins at the bottom layer and works its way through the network until it arrives transformed at the output layer. The weights and threshold are continuously adjusted until the training data reaches an optimal output.')
        st.write('One of the common methods used for activation is called ReLU. ReLU takes a real number as input and will return the maximum between 0 and the number. The ReLU function basically just "turns off" a neuron if its input is less than 0, and is linear if its input is greater than zero. ')
        st.write('Neural networks is an effective algorithm for both supervised and unsupervised learning. Meaning, it can be used on both structured and unstructured data. It has however been incredibly successful on unstructured data. The neural network model is highly effective as the amount of data increases.')
        img2 = Image.open('NN.jpg')
        st.image(img2, width = 500)
    if system == ('KNN'):
        st.write('K-nearest neighbours or KNN, is a non-parametric algorithm, this means that the data distribution contains unspecified parameters, in other words it does not make any assumptions on the underlying data distribution.')
        st.write('This algorithm will use a database in which the data points are separated into several classes to predict the classification of a new sample point. KNN does not require training data points to do its generalisation work.')
        st.write('KNN uses feature similarity to categorise the data outputting a class membership. An object is classified by a majority vote of its neighbours, with the object being assigned to the class most common among its k nearest neighbours.')
        st.write('This method can also be used on a regression model by outputting the average or median of the values of its k nearest neighbours. This method requires k to be specified before the calculations are made.')
        img3 = Image.open('u1.jpg')
        st.image(img3, width = 500)
if selection == 'Data Behaviour(EDA)':
    st.title('How Our Movie Data Behaves')
    from PIL import Image
    img4 = Image.open('halloween.jpg')
    st.image(img4, width=500)
    st.subheader('Distribution of Users and Ratings')
    eda = st.radio('Distribution of Data:',('Distribution of Ratings', 'Distribution of Top Users'))
    if eda == ('Distribution of Ratings'):
        from PIL import Image
        img5 = Image.open('rating distribution.jpg')
        st.image(img5, width=500)
        st.write('Ratings above three occur more frequently indicting that users who rate the films are either generous or that users are more likely to rate the films if they found if satisfactory or good')
    if eda == ('Distribution of Top Users'):
        img6 = Image.open('user distribution.jpg')
        st.image(img6, width=500)
        st.write('This data respresents the number of users who rated over 2000 films. Very few users rated many films, the films rated by many users represented the distribution of popular films')
    st.subheader('A Look At The Titles Of Popular Or Influential Films')
    wordcloud = st.radio('Highly Rated Films:',('Highly Rated Films', 'Films With A Low Rating' , 'Films Rated by The Greatest Number Of People'))
    if wordcloud == 'Highly Rated Films':
        img7 = Image.open('highly rated films wc.jpg')
        st.image(img7,width=700)
    if wordcloud == 'Films With A Low Rating':
        img8 = Image.open('Low rated films wc.jpg')
        st.image(img8,width=700)
    if wordcloud == 'Films Rated by The Greatest Number Of People':
        img9 = Image.open('rated by most people wc.jpg')
        st.image(img9, width=700)