import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    ### YOUR CODE HERE
    
    # Assume kernel is ODD_Number* ODD_Number
    pad_H, pad_W = Hk//2, Wk//2
    for hi in range(Hi):
        for wi in range(Wi):
            for hk in range(-pad_H, pad_H+1):
                for wk in range(-pad_W, pad_W+1): 
                    # Be careful boundaries !!!
                    if hi-hk >= 0 and hi-hk < Hi and wi-wk >= 0 and wi-wk < Wi:
                        # g[m, n] = k[i, j] * i[m-i, n-j]
                        out[hi, wi] += kernel[hk+pad_H, wk+pad_W] * image[hi-hk, wi-wk]
    ### END YOUR CODE
    
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros([H+2*pad_height, W+2*pad_width])
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_H, pad_W = Hk//2, Wk//2 
    pad_img = zero_pad(image, pad_H, pad_W)
    
    conv_kernel = np.flip(np.flip(kernel,axis=0),axis=1)
    
    #for m in range(pad_H, Hi-pad_H):
    #    for n in range(pad_W, Wi-pad_W):
    #        out[m, n] = np.sum(conv_kernel * pad_img[m:m+Hk, n:n+Wk])
            
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(conv_kernel * pad_img[m:m+Hk, n:n+Wk])
            
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    cross_kernel = np.flip(np.flip(g, axis=0), axis=1)
    out = conv_fast(f, cross_kernel)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    new_g = g - g_mean
    out = cross_correlation(f, new_g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_mean, g_std = np.mean(g), np.std(g)
    normal_kernel = (g-g_mean)/g_std
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))

    pad_H, pad_W = Hg//2, Wg//2 
    pad_img = zero_pad(f, pad_H, pad_W)
               
    for m in range(Hf):
        for n in range(Wf):
            image_patch = pad_img[m:m+Hg, n:n+Wg]
            f_mean, f_std = np.mean(image_patch), np.std(image_patch) 
            out[m, n] = np.sum(normal_kernel * (image_patch-f_mean)/f_std )
          
    ### END YOUR CODE

    return out
