import numpy as np
import cv2


def myConv2(A, B, param):
    Β = np.flipud(np.fliplr(Β)) # Flip matrix B
    output = np.zeros_like(A)     
    # param can be 'pad' or 'same'
    if param=='pad':
        # Add zero padding to the input matrix A
        A_padded = np.zeros((A.shape[0] + 2, A.shape[1] + 2))   
        A_padded[1:-1, 1:-1] = A
        for x in range(A.shape[1]):    # Loop over every pixel of the image
            for y in range(A.shape[0]):
                # element-wise multiplication of the B and A matrices
                output[y,x]=(B*A_padded[y:y+3,x:x+3]).sum()
    elif param=='same':
        for x in range(A.shape[1]):    # Loop over every pixel of the image
            for y in range(A.shape[0]):
                # element-wise multiplication of the B and A matrices
                output[y,x]=(B*A[y:y+3,x:x+3]).sum()
    return output


def myImNoise(A, param):
    # add noise according to the parameter
    if param == 'gaussian':
      row,col = A.shape
      # define mean and standard deviation
      mean = 0
      sigma = 0.1
      gauss = np.random.normal(mean,sigma,(row,col)) # draw samples from the distribution
      gauss = gauss.reshape(row,col)
      noisy = A + gauss # add noise to image
      return noisy
    elif param=='saltandpepper':
      row,col = A.shape
      # define the density of s&p
      s_vs_p = 0.5
      amount = 0.1
      noisy = np.copy(A)
      # Salt mode
      # choose the pixels that you will add white dots
      num_salt = np.ceil(amount * A.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))for i in A.shape]
      noisy[coords] = 1

      # Pepper mode
      # choose the pixels that you will add black dots
      num_pepper = np.ceil(amount* A.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))for i in A.shape]
      noisy[coords] = 0
      return noisy


def myImFilter(A, param):
    # fitler image A according to the parameter
    if param=='median':
        members = [(0,0)] * 9 # create a 3x3 mask
        height,width = A.shape
        newimg = np.zeros((height, width, 3), np.uint8)
        # fill the matrix with the desired pixels from A
        for i in range(1,A.shape[0]-1):
            for j in range(1,A.shape[1]-1):
                members[0] = A[i-1,j-1]
                members[1] = A[i-1,j]
                members[2] = A[i-1,j+1]
                members[3] = A[i,j-1]
                members[4] = A[i,j]
                members[5] = A[i,j+1]
                members[6] = A[i+1,j-1]
                members[7] = A[i+1,j]
                members[8] = A[i+1,j+1]
                members.sort()
                newimg[i,j] = members[4] # choose the median from the sorted array
        return newimg
    elif param=='mean':
        newimg = A
        # calculate and choose the mean of a 3x3 part of pixels from A
        for i in range(1,A.shape[0]-1):
            for j in range(1,A.shape[1]-1):
                block = A[i-1:i+2, j-1:j+2]
                m = np.mean(block,dtype=np.float32)
                newimg[i,j] = int(m)
        return newimg

        


A = cv2.imread('test.jpg', 0)  # read image - black and white
cv2.imshow('image', A)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

noise_param = input("gaussian or saltandpepper kind of noise? ") # ask user's choice
B = myImNoise(A, noise_param)
cv2.imwrite('noisy.jpg', B)  # save image to disk
cv2.imshow('image', B)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window

filter_param = input("mean or median filtering? ") # ask user's choice
C = myImFilter(B, filter_param)
cv2.imwrite('filtered.jpg', C)  # save image to disk
cv2.imshow('image', C)  # show image
cv2.waitKey(0)  # wait for key press
cv2.destroyAllWindows()  # close image window
