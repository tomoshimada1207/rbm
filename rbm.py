import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

##### Save Point Info #####

def p_h_given_v(v,W,b,c):
    # Sigmoid Function for p(h|v)
    return 1.0/(1.0 + np.exp(-(np.matmul(W, v) + c)))

def p_v_given_h(h,W,b,c):
    # Sigmoid Function for p(h|v)
    return 1.0/(1.0 + np.exp(-(np.matmul(h.T, W).T + b)))

def samp_h_given_v(v,W,b,c):
    # Sampling from the distribution: p(h|v)

    h_given_v_val = p_h_given_v(v,W,b,c)
    unif_samp = np.random.uniform(0,1, h_given_v_val.shape)

    h_ = (unif_samp < h_given_v_val) * 1 

    return h_, h_given_v_val, unif_samp

def samp_v_given_h(h,W,b,c):
    # Sampling from the distribution: p(v|h)
    v_given_h_val = p_v_given_h(h,W,b,c)
    assert v_given_h_val.shape == b.shape
    unif_samp = np.random.uniform(0,1, size=v_given_h_val.shape)

    v_ = (unif_samp < v_given_h_val) * 1 

    return v_, v_given_h_val, unif_samp

def contrastive_divergence(num_of_hidden, num_of_visible, W, b, c, v_train):
    
    assert v_train.shape == (num_of_visible, 1)

    # Initialize the delta parameters
    dW = np.zeros((num_of_hidden, num_of_visible), dtype=np.longdouble)
    db = np.zeros((num_of_visible, 1), dtype=np.longdouble)
    dc = np.zeros((num_of_hidden, 1), dtype=np.longdouble)

    # k value for CD-k Algorithm
    k = 1

    v = v_train

    # Gibbs sampling section (approximating the gradient)
    for i in range(k):
        h_, h_given_v_val, _ = samp_h_given_v(v, W, b, c)
        v_, v_given_h_val, _ = samp_v_given_h(h_, W, b, c)
        h_2, h_given_v_val2, _ = samp_h_given_v(v_, W, b, c)

        assert v_.shape == (num_of_hidden, 1)

        # Save the first sample which is needed next
        if i == 0:
            first_v_ = v
            first_h_given_v_val = h_given_v_val

    ## Updating our parameters (using CD-k algo in the paper) ##    
    # for W:
    for i in range(len(h_[:,0])):
        dW[i,:] += first_h_given_v_val[i,:] * first_v_.reshape((num_of_visible,)) - h_given_v_val2[i,:] * v_.reshape((num_of_visible,))

    # for b:
    db += v - v_
    
    # for c:
    dc += first_h_given_v_val - h_given_v_val2
    

    return dW, db, dc


if __name__ == '__main__':
    ### Creating BAS Data Set ###
    # Generate BAS data set

    BAS_imgs = []

    for i in range(16):
        x = np.zeros((4,4), dtype=int)
        y = np.zeros((4,4), dtype=int)

        # Finding the indexes to swap with [1,1,1,1]
        i_bin = bin(i)
        ind = [i for i, ltr in enumerate(i_bin[::-1]) if ltr == '1']
        x[ind] = np.array([1,1,1,1])
        y = x.T

        BAS_imgs.append(x)
        BAS_imgs.append(y)  

    k = 0
    fig, axes = plt.subplots(4,4, figsize=(10,10))
    for i in range(4):
        for j in range(4):
            temp = BAS_imgs[::2]
            axes[i,j].imshow(temp[k], cmap='gray', vmin=0, vmax=1)
            k+=1

    plt.show()

    k = 0
    fig, axes = plt.subplots(4,4, figsize=(10,10))
    for i in range(4):
        for j in range(4):
            temp = BAS_imgs[1::2]
            axes[i,j].imshow(temp[k], cmap='gray', vmin=0, vmax=1)
            k+=1
            
    plt.show()


    # Training using BAS data set
    train_data_formatted = np.empty((16,0))

    for BAS_img in BAS_imgs:
        train_data_formatted = np.column_stack((train_data_formatted, BAS_img.reshape(16,1)))

    # Number of hidden and visible nodes
    num_of_hidden = 16
    num_of_visible = len(train_data_formatted[:,0])
    learn_rate = 0.1
    # print(num_of_visible)

    # Initializing the weights and biases
    W = 0.01*np.random.randn(num_of_hidden, num_of_visible)
    b = np.zeros((num_of_visible, 1), dtype=np.longdouble)
    c = np.zeros((num_of_hidden, 1), dtype=np.longdouble)

    # Actual training of data (CD-1 and Parameter updates)
    for i in range(4000):
        for j in range(len(train_data_formatted[0,:])):
            v_train = train_data_formatted[:,j].reshape((num_of_visible, 1))
            # print(v_train)
            dW, db, dc = contrastive_divergence(num_of_hidden, num_of_visible, W, b, c, v_train)
            # print(dc)
            W += learn_rate*dW
            b += learn_rate*db
            c += learn_rate*dc

    v_new = np.array([1,1,1,1,
                      0.5,0.5,0.5,0.5,
                      0.5,0.5,0.5,0.5,
                      0.5,0.5,0.5,0.5]).reshape((16,1))
    v_1 = v_new
    k = 4

    for i in range(k):
        h_1, _, _ = samp_h_given_v(v_1,W,b,c)
        v_1, _, _ = samp_v_given_h(h_1,W,b,c)
        v_1[0] = 1
        v_1[1] = 1
        v_1[2] = 1
        v_1[3] = 1

    img_iter = v_1.reshape((4, 4))

    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    axes[0].imshow(v_new.reshape((4,4)), cmap='gray', vmin=0,vmax=1)
    axes[0].set_title('Clamped Input')
    
    axes[1].imshow(img_iter, cmap='gray')
    axes[1].set_title('RBM Output after ' + str(k) + ' Steps of Gibbs Sampling')

    plt.show()
