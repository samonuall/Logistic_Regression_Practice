#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
from scipy import ndimage

get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


#Dataset from https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset
root_dir = 'PetImages/'
num_cats = 100
num_dogs = 100
num_test = 60 #number of test images in all, half will be cat and half will be dog
img_size = 200 #dimension for square cropout of images from top left

cat_train, cat_end = get_images('Cat', num_cats, img_size)
#make last pixel 0 or 1 as a way to label images for later shuffling
cat_train[:, 0, -1, -1, :] = np.array([1, 1, 1])

dog_train, dog_end = get_images('Dog', num_dogs, img_size)
dog_train[:, 0, -1, -1, :] = np.array([0, 0, 0])

train_set = np.concatenate((cat_train, dog_train))
#shuffle order of images, keeping labels and pixels intact
np.random.shuffle(train_set)

cat_test, _ = get_images('Cat', num_test // 2, img_size, counter=cat_end)
cat_test[:, 0, -1, -1, :] = np.array([1, 1, 1])

dog_test, _ = get_images('Dog', num_test // 2, img_size, counter=dog_end)
dog_test[:, 0, -1, -1, :] = np.array([0, 0, 0])

test_set = np.concatenate((cat_test, dog_test))
np.random.shuffle(test_set)

#Make label vector and then delete the last pixel that had the label to stop
#model from learning that that pixel is the label.
train_labels = train_set[:, 0, -1, -1, 0].reshape(num_cats+num_dogs, 1)
train_set = train_set[:, :, :-1, :-1, :]

test_labels = test_set[:, 0, -1, -1, 0].reshape((num_test // 2)*2, 1)
test_set = test_set[:, :, :-1, :-1, :]


# In[82]:


#Flatten images so they're a column of R G and B vals, then transpose labels to match their shape
flattened_train = train_set.reshape(train_set.shape[0], -1).T
flattened_test = test_set.reshape(test_set.shape[0], -1).T
train_labels = train_labels.T
test_labels = test_labels.T
#Standardize data, new variable names so previous cell doesn't mess stuff up
train = flattened_train / 255.
test = flattened_test / 255.


# In[77]:


#Maybe can somehow vectorize the while loop, later try doing cropping from middle to see the change in effectiveness
def get_images(folder_name, num_images, img_size, counter=0):
    """
    Inputs:
        folder_name: name of folder in archive, has to be Cat or Dog
        num_images: number of images to be extracted
        img_size: minimum dimension of an image to be added to list, same number for row and column
        counter: where to start image search, defaults to 0

    Returns:
        column vector of np.array images that were cropped from top left, shape = (num_images, 1, img_size, img_size, 3),
        and the other part of the tuple is the index of the last used image so that test sets can know where
        the training set left off in order to have unique images
    """
    imgs = np.array([0]) #initialize with any number

    while imgs.shape[0] < num_images:
        img = np.array(Image.open(root_dir+'{}/{}.jpg'.format(folder_name, counter)))

        #for some reason some image is corrupted and only is shape (_, _), so have to check for that
        #before indexing in the next step to avoid errors
        if len(img.shape) < 3:
            counter += 1
            continue
        img = img.reshape(1, 1, img.shape[0], img.shape[1], img.shape[2])
        if img.dtype is not object and img.shape[2] >= img_size and img.shape[3] >= img_size:
            #if imgs hasn't been set to an image yet, set it to the first image so
            #other images can now be appended to the array
            if len(imgs.shape) < 3:
                imgs = img[:, :, :img_size, :img_size, :]
                imgs = imgs.reshape(1, 1, img_size, img_size, 3)
            else:
                #append along rows so end result is a column vector of images
                imgs = np.append(imgs, img[:, :, :img_size, :img_size, :], axis=0)
        counter += 1
    return (imgs, counter)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    x[x < 0] = 0
    return x

def init_with_zeros(dim):
    """
    Inputs:
        dim: number of weights to be intialized

    Returns:
        w: numpy array of zeros with size (dim, 1)
        b: 0, bias
    """
    w = np.zeros(dim).reshape(dim, 1)
    w += .01
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y, activation=sigmoid):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    z = w.T @ X + b #row vector of weighted sums, X.shape[0] is num_images
    a = activation(z) #activation function, apply given function to each weighted sum
    cost = -np.sum(Y*np.log(a) + (1 - Y) * np.log(1-a)) / X.shape[1]

    dw = np.matmul(X, (a - Y).T) / X.shape[1]
    dw = dw.reshape(w.shape)
    db = np.sum(a - Y) / X.shape[1]


    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X, activation=sigmoid):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    z = w.T @ X + b #row vector of weighted sums, X.shape[0] is num_images
    a = activation(z) #activation function, apply sigmoid function to each weighted sum
    a[np.where(a > .5)] = 1
    a[np.where(a <= .5)] = 0
    Y_prediction = a.reshape(1, X.shape[1])


    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = init_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w, b = params['w'], params['b']
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d



# In[91]:


d = model(train, train_labels, test, test_labels, num_iterations = 700, learning_rate = 0.01, print_cost = True)


# In[ ]:





# In[ ]:
