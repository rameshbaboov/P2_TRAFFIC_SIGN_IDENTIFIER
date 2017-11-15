# Udacity Traffic sign identifier
Below code loads the pickled data into Python and ensure that labels and features are of equal length for all three environments Test, Train and Validation

```python
# load pickled data consisting of three environments
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#enable inline visualization
%matplotlib inline


training_data = "./data/train.p"
testing_data = "./data/test.p"
valid_data = "./data/valid.p"

with open(training_data,mode='rb') as f:
    train = pickle.load(f)
with open(testing_data,mode='rb') as f:
    test = pickle.load(f)
with open(valid_data,mode='rb') as f:
    valid = pickle.load(f)
    

X_train,Y_train = train['features'],train['labels']
X_test,Y_test = test['features'],test['labels']
X_valid,Y_valid = valid['features'],valid['labels']

assert(len(X_train) == len(Y_train))
assert(len(X_valid) == len(Y_valid))
assert(len(X_test) == len(Y_test))

print(" X train shape is ", X_train.shape)
print(" Y train shape is ", Y_train.shape)
print(" X test shape is ", X_test.shape)
print(" Y test shape is ", Y_test.shape)
print(" X valid shape is ", X_valid.shape)
print(" Y valid shape is ", Y_valid.shape)

```
## OUTPUT
```
X train shape is  (34799, 32, 32, 3)
Y train shape is  (34799,)
X test shape is  (12630, 32, 32, 3)
Y test shape is  (12630,)
X valid shape is  (4410, 32, 32, 3)
Y valid shape is  (4410,)
```
# Verify a random image
Print a random image to check if image is printed correctly and also verify the dimensions
```python
# print a random image

image_num = random.randint(0,len(X_train))
image = X_train[image_num,:,:,:]

# Show image
print('This image is:', type(image), 'with dimesions:', image.shape)
print('Showing image #', image_num, '...')
plt.imshow(image)  
```

## OUTPUT

```
This image is: <class 'numpy.ndarray'> with dimesions: (32, 32, 3)
Showing image # 13300 ...
Out[72]:
<matplotlib.image.AxesImage at 0x7f5abe8d4b38>
```

# Check the No of entries
```python

# get number of training entries
no_train = len(X_train)

# get number of testing entries
no_test = len(X_test)

# get number of valid entries
no_valid = len(X_valid)

#get image shape
image_shape = X_train[0].shape

#get unique classes in teh dataset

no_classes = len(np.unique(Y_train))

#print all information

print('No of training entries', no_train)
print('No of training entries - y', len(Y_train))
print('No of testing entries', no_test)
print('No of valid entries', no_valid)

print('No of unique classes', no_classes)
print('image shape is ', image_shape)
```
## OUTPUT

```
No of training entries 34799
No of training entries - y 34799
No of testing entries 12630
No of valid entries 4410
No of unique classes 43
image shape is  (32, 32, 3)
```

# Data Visualization
```python
# start data visualization
# select random of 10 images
fig, axs = plt.subplots(4,5,figsize =(15,6))
fig.subplots_adjust(hspace=.2,wspace=0.001)
axs = axs.ravel()

for i in range(20):
    index = random.randint(0,len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(Y_train[index])   
```

# Print Histogram by classes

```python
# Create histogram of label frequency
hist,bins = np.histogram(Y_train,bins=no_classes)
width=0.7 *(bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center,hist,align='center',width=width)
plt.show()
```

# convert to grey scale to improve performance
```python
# convert to grey scale

X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)


print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)

#overwrite variables with grey shape

X_train = X_train_gry
X_test = X_test_gry
X_valid = X_valid_gry
```

## output
```
RGB shape: (34799, 32, 32, 3)
Grayscale shape: (34799, 32, 32, 1)
```

# compare grey scale and RGB image to check if they are matching
## output
```python
n_rows = 8
n_cols = 10
offset = 9000
fig, axs = plt.subplots(n_rows,n_cols, figsize=(18, 14))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()
for j in range(0,n_rows,2):
    for i in range(n_cols):
        index = i + j*n_cols
        image = X_train_rgb[index + offset]
        axs[index].axis('off')
        axs[index].imshow(image)
    for i in range(n_cols):
        index = i + j*n_cols + n_cols
        image = X_train_gry[index + offset - n_cols].squeeze()
        axs[index].axis('off')
        axs[index].imshow(image, cmap='gray')
```

# Check if data has to be shuffled
```python
# check if data has to be shuffled
print(np.unique(Y_train))
print(Y_train[0:500])
print(Y_train[501:1000])
```

## output
```
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42]
[41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31]
[31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 36
 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36
 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36
 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36
 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36]

```

# Normalize the data to mean close to zero
```python
X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)

X_test_mean = np.mean(X_test)
X_test_std = np.std(X_test)

X_valid_mean = np.mean(X_valid)
X_valid_std = np.std(X_valid)


print("mean for X_train", np.mean(X_train))
print("mean for X_test",np.mean(X_test))
print("mean for X_valid",np.mean(X_valid))


print("stddev for X train", np.std(X_train))
print("stdev for X test", np.std(X_test))
print("stdev for X valid", np.std(X_valid))


X_train_normalized = (X_train - X_train_mean)/X_train_std
X_test_normalized = (X_test - X_test_mean)/X_test_std
X_valid_normalized = (X_valid - X_valid_mean)/X_valid_std


print("standardized mean for X_train", np.mean(X_train_normalized))
print("standardized mean for X_test", np.mean(X_test_normalized))
print("standardized mean for X_valid", np.mean(X_valid_normalized))



print("standardized stdev for X_train", np.std(X_train_normalized))
print("standardized stddev for X_test", np.std(X_test_normalized))
print("standardized stddev for X_valid", np.std(X_valid_normalized))

X_train_max = np.max(X_train)
X_train_min = np.min(X_train)
X_train_normalized = (X_train - X_train_min)/(X_train_max - X_train_min)

X_test_max = np.max(X_test)
X_test_min = np.min(X_test)
X_test_normalized = (X_test - X_test_min)/(X_test_max - X_test_min)

X_valid_max = np.max(X_valid)
X_valid_min = np.min(X_valid)
X_valid_normalized = (X_valid - X_valid_min)/(X_valid_max - X_valid_min)

print("Normalized mean for X_train", np.mean(X_train_normalized))
print("Normalized mean for X_test", np.mean(X_test_normalized))
print("Normalized mean for X_valid", np.mean(X_valid_normalized))



print("Normalized stdev for X_train", np.std(X_train_normalized))
print("Normalized stddev for X_test", np.std(X_test_normalized))
print("Normalized stddev for X_valid", np.std(X_valid_normalized))

```

## outout
```
mean for X_train 82.677589037
mean for X_test 82.1484603612
mean for X_valid 83.5564273756
stddev for X train 66.0097957522
stdev for X test 66.7642435759
stdev for X valid 67.9870214471
standardized mean for X_train 8.21244085217e-16
standardized mean for X_test 1.58103494227e-15
standardized mean for X_valid 3.86689924223e-17
standardized stdev for X_train 1.0
standardized stddev for X_test 1.0
standardized stddev for X_valid 1.0
Normalized mean for X_train 0.314367065134
Normalized mean for X_test 0.311348447654
Normalized mean for X_valid 0.314225709503
Normalized stdev for X_train 0.262638444637
Normalized stddev for X_test 0.265993002295
Normalized stddev for X_valid 0.271948085788

```

# check randomly to see if images are fine
```python

# check randmoly few images to check if images are fine


print("Original shape:", X_train.shape)
print("Normalized shape:", X_train_normalized.shape)
for i in range(1,5):
    
    index = random.randint(0,len(X_train_normalized))
    fig, axs = plt.subplots(1,2, figsize=(10, 3))
    axs = axs.ravel()
    axs[0].axis('off')
    axs[0].set_title('normalized')
    axs[0].imshow(X_train_normalized[index].squeeze(), cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('original')
    axs[1].imshow(X_train[index].squeeze(), cmap='gray')
```

# Preprocessing function
define all preprocessing functions. test with a radom image to check if images are fine prior and after the preprocessing
```python
import cv2
print(len(X_train))
print(len(Y_train))

index = random.randint(0,len(X_train))

def random_translate(img):
    rows,cols,_ = img.shape
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

# test this function
test_img = X_train_normalized[index]
test_dst = random_translate(test_img)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')
axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(), cmap='gray')
axs[1].set_title('translated')
print('shape in/out:', test_img.shape, test_dst.shape)

def random_scaling(img):
    rows,cols,_ = img.shape
    # transform limits
    px = np.random.randint(-2,2)
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(rows,cols))
    dst = dst[:,:,np.newaxis]
    return dst

test_dst = random_scaling(test_img)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')
axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(), cmap='gray')
axs[1].set_title('scaled')
print('shape in/out:', test_img.shape, test_dst.shape)
    
def random_warp(img):
    rows,cols,_ = img.shape
    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06 # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06
    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    pts1 = np.float32([[y1,x1],
    [y2,x1],
    [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
    [y2+rndy[1],x1+rndx[1]],
    [y1+rndy[2],x2+rndx[2]]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

test_dst = random_warp(test_img)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')
axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(), cmap='gray')
axs[1].set_title('warped')
print('shape in/out:', test_img.shape, test_dst.shape)
    

def random_brightness(img):
    shifted = img + 1.0 # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

test_dst = random_brightness(test_img)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs[0].axis('off')
axs[0].imshow(test_img.squeeze(), cmap='gray')
axs[0].set_title('original')
axs[1].axis('off')
axs[1].imshow(test_dst.squeeze(), cmap='gray')
axs[1].set_title('brightness adjusted')
print('shape in/out:', test_img.shape, test_dst.shape)
print(len(X_train))
print(len(Y_train))


```

# histogram
```python
# histogram of label frequency (once again, before data augmentation)
hist, bins = np.histogram(Y_train, bins=no_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
```

# check Bin count and unique data and calculate the minimum samples for all lables
```python
print(np.unique(Y_train), np.bincount(Y_train))
print("minimum samples for any label:", min(np.bincount(Y_train)))
```

## output
```
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42] [ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920  690
  540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240
  390  690  210  599  360 1080  330  180 1860  270  300  210  210]
minimum samples for any label: 180
```

# apply preprocessing
```python
      
    print('X, y shapes:', X_train_normalized.shape, Y_train.shape)
    input_indices = []
    output_indices = []
    for class_n in range(no_classes):
        class_indices = np.where(Y_train == class_n)
        n_samples = len(class_indices[0])
        print("no of samples is ", n_samples)
        if n_samples < 500:
            for i in range(500 - n_samples):
                input_indices.append(class_indices[0][i%n_samples])
                output_indices.append(X_train_normalized.shape[0])
                new_img = X_train_normalized[class_indices[0][i % n_samples]]
                new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
                X_train_normalized = np.concatenate((X_train_normalized, [new_img]), axis=0)
                Y_train = np.concatenate((Y_train, [class_n]), axis=0)
    print('X, y shapes:', X_train_normalized.shape, Y_train.shape)
    print(len(X_train))
    print(len(Y_train))
    
```
    
```python
print(len(X_train_normalized))
print("train normalized mean is", np.mean(X_train_normalized))
```


```python
print(len(X_train))
print(len(Y_train))
choices = list(range(len(input_indices)))
picks = []
for i in range(5):
    rnd_index = np.random.randint(low=0,high=len(choices))
    picks.append(choices.pop(rnd_index))
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(5):
    image = X_train_normalized[input_indices[picks[i]]].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap = 'gray')
    axs[i].set_title(Y_train[input_indices[picks[i]]])
for i in range(5):
    image = X_train_normalized[output_indices[picks[i]]].squeeze()
    axs[i+5].axis('off')
    axs[i+5].imshow(image, cmap = 'gray')
    axs[i+5].set_title(Y_train[output_indices[picks[i]]])

```

# Shuffle all data

```python
## Shuffle the training dataset
from sklearn.utils import shuffle
X_train_normalized, Y_train = shuffle(X_train_normalized, Y_train)
X_train = X_train_normalized
print('done')

X_test_normalized, Y_test = shuffle(X_test_normalized, Y_test)
X_test = X_test_normalized
print('done')

X_valid_normalized, Y_valid = shuffle(X_valid_normalized, Y_valid)
X_valid = X_valid_normalized
print('done')


# check if data is shuffled

print(np.unique(Y_train))
print(Y_train[0:500])
print(Y_train[501:1000])
```

## output

```
done
done
done
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42]
[24 37  2  9 25  1  9  2  5 38  2  7 12  9  2  2  3 12  5 18 41  3 38 16 24
 33  8 28 10  9 12 29 25  5  4 23  5 11 38  6 11 14 10  8 18 22  5 16 30 13
  3 38  9 25 36 25 16 12 12  6  5 29 18  4 20  5 20  9  9 10 25 40  1 24 13
  2 23 30 10 22  4  2  3  1 26  2 17  8 17 11 12 21  2  7 34  8  8  1 13 25
 20 17 21 12 16 23 35 23 10 35 39 18 13 21 15  6  4 12 25 38 16 38 13  1  9
  1  5 28 33 38 33  1  4 10 17 18 15 12  7  4 17 35 32 13 17 10 10  4 13 11
 12 35  1 18 25  1 17 21 11 27  4 35 11  6 14  1 17  2 13  9 11 39  4  7  5
 15 35 35 10 38 12  5  4 11  8 11 38 31 40  1  2  7  1  9  1 19  1  7 11  2
 15  7  2 28 12 15  5 31 33 37  4  4  4 20  9 11 20  3 12 20 12  2 25 38 20
  9 17 35 13  6 23 17  7 12  1  6 38  8 10 10 10 38 12  3  3  3 42  8  1  2
 13  4 35 12 39  8 11  5 41 38  7 34 30 26  0 13 36 23  5 33  4  2 34 24 38
 26 34  5 38 10  3 10 15  8 17  4 11 20 20 23 31  2 11 16  1 25  1 25 19 27
  2 35  9 11  5 35  3  2  7 37  0 28  4 13  3 42 14 40 28  4 37  6 14 18 26
 38 12 34  4  6  1  1 11 13 18 12 25 31  2 20 13 18 18  8  5  6 29 12 31 23
  1 35 12  4 26 35  4 28 25 13 17  1  8  3 36 11  2 30 42 13 13 35 21 12 12
 24 10 33 12 22  2 38 40 30 25 29 14  4  9 35  3 30  4 22 13 13 25  9 11  4
  5 37 40 18 33 29 17  8 21 38  8 20 13 33 38 30 29  9 12  4 13  2  1 35 30
  2 40  2 34  5  7  7 38  1 18 36  8 41 25 10 39 33  4  3  5  8 25  2  7  9
 12 14  1  9  5  1  7  3 39 10  4  4 31 29  7 29 24 26 10 31 38  1 23 27 22
  1  7  2 16 13  5 41  5 15 18 13 38  7 13  9 19  3 36 18 25  5  1 22 11 25]
[ 1  1  8 12 20  1 25  5 11 37 40 12 27 15 25 41 12 33  3  7 17  9 25 13 40
 33 12 38 14  7 20  4 29 13 42 13 17 36 19 13  3  2 12 38  7 38 18  9 35  2
  9 37 35 17  5 34 10 17 10 25  1 15 23 21 25 35  9 42 18 12 19  8 36 12  4
 13 10 19 30 38 10  4  1 37 16  5 39 12  7  5 10 35 42 37 31  3  6 36  8 14
  2  4  4 14  1  2  5  7  6  7 16 40 12  4 23 30  8  7  1  2 18  5  6  1 12
  7 31 39  7 20 25 28 28 38 21  0  4  7 11 39  8 16 16  2 17  9 25 27 13 13
 40  9  4 32  2  2 10 31  4 30 30  2 35  9 38 34 32 11  9  0  2 11 37 32 25
  1 38 26  0 38 11 28 32 22 29 27 38  5 26  5 13 35 19 28 30  9  5  7 25  4
  7  7  6  7 38 18 33 14 27  7 32  3  7 18  2  7 39 10 38 24 42 13  5 11  2
 12  8  5  3  4 41 14 42  5 13  2 13 31  1 12  2 28 19  2 12 29  4 22 11 12
 13 18  7 12 13 13 37 29 13  5 18 15  7  1 33 13  3  3  8  4 10  4  3 15 12
  2 34 38 17 41 38  8 25  4 10 18 35 30 40 41 18 34  8 41 13  7 18 24  8  5
 12  2 38 11 40  8 11 38 22  9 10  3 12 12  7  9  2 10  4 27  1 12 16  0  1
  9 16  4 25  1 12 41 25  5 27  6 17 12 30 32 35  2 38 38  3 34 31 14  4 10
 14  9  5  6 32 29 26 25  1  7  5 13  0  4 37 39 21 32  7  5 21 25  8  5 12
 12 14  9 38 11 14 13 13  2 13 13  1 29 14 13 23 13  5  2  8 16 16  3  9 37
 15 12 36 14 17  1  4  1 10 41 18 12 15 26  4 17  8  2 26  4  4 36 38 28 41
 35  4  4 26 42 12 17 28  5 33 12 35  9 29  7 13 27 17 37 12 33 35 12 35  1
 38  7  4  1  3  8  0 17 32 38 13 30 13  5  5 35  7 13 29  5 41 13 25 29 18
 24 37 40  7  5  8  5 26  1  4 29 31 34  5  3  8 12  2  8 42  1  6  4 25]
```


# set EPOCH AND BATCH SIZE
```python
import tensorflow as tf
EPOCHS = 40
BATCH_SIZE = 128
print('done')

```

# define the LeNet architecture
The architecture sets mu as 0 and sigma as 0.1. First is convolutional layer and second uses RELU activation function. Next is pooling followed by convolution layer and relu activation function. Next is Pooling and flatten. Last layer is fully connected layer. The function returns the Logits


```python

from tensorflow.contrib.layers import flatten
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
print('done')
```

# define place holders
define x and y place holders and one hot function
```python
x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
keep_prob  = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,43)
print('done')
```

# Regression function
define learning rate. Calculate the cross entropy using logits and one hot function. 
optimize the model using the learning training
```python

rate = 0.0009
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

```

# function to validate the accuracy
```python
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
```

```python
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
print('done')
```

# Train the model using training data and validate the accuracy using validation data
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training...")
 
   
   
    for i in range(EPOCHS):
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        validation_accuracy = evaluate(X_valid, Y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess, 'lenet')
    print("Model saved")
    
 ```
 
 ## output
```
 
 Training...
EPOCH 1 ...
Validation Accuracy = 0.691

EPOCH 2 ...
Validation Accuracy = 0.800

EPOCH 3 ...
Validation Accuracy = 0.838

EPOCH 4 ...
Validation Accuracy = 0.860

EPOCH 5 ...
Validation Accuracy = 0.863

EPOCH 6 ...
Validation Accuracy = 0.875

EPOCH 7 ...
Validation Accuracy = 0.891

EPOCH 8 ...
Validation Accuracy = 0.891

EPOCH 9 ...
Validation Accuracy = 0.890

EPOCH 10 ...
Validation Accuracy = 0.902

EPOCH 11 ...
Validation Accuracy = 0.903

EPOCH 12 ...
Validation Accuracy = 0.914

EPOCH 13 ...
Validation Accuracy = 0.901

EPOCH 14 ...
Validation Accuracy = 0.905

EPOCH 15 ...
Validation Accuracy = 0.906

EPOCH 16 ...
Validation Accuracy = 0.911

EPOCH 17 ...
Validation Accuracy = 0.905

EPOCH 18 ...
Validation Accuracy = 0.918

EPOCH 19 ...
Validation Accuracy = 0.915

EPOCH 20 ...
Validation Accuracy = 0.913

EPOCH 21 ...
Validation Accuracy = 0.921

EPOCH 22 ...
Validation Accuracy = 0.914

EPOCH 23 ...
Validation Accuracy = 0.920

EPOCH 24 ...
Validation Accuracy = 0.925

EPOCH 25 ...
Validation Accuracy = 0.919

EPOCH 26 ...
Validation Accuracy = 0.925

EPOCH 27 ...
Validation Accuracy = 0.893

EPOCH 28 ...
Validation Accuracy = 0.931

EPOCH 29 ...
Validation Accuracy = 0.928

EPOCH 30 ...
Validation Accuracy = 0.931

EPOCH 31 ...
Validation Accuracy = 0.911

EPOCH 32 ...
Validation Accuracy = 0.908

EPOCH 33 ...
Validation Accuracy = 0.922

EPOCH 34 ...
Validation Accuracy = 0.916

EPOCH 35 ...
Validation Accuracy = 0.931

EPOCH 36 ...
Validation Accuracy = 0.927

EPOCH 37 ...
Validation Accuracy = 0.909

EPOCH 38 ...
Validation Accuracy = 0.937

EPOCH 39 ...
Validation Accuracy = 0.923

EPOCH 40 ...
Validation Accuracy = 0.917

Model saved

```
# validate the model using test data
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, Y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
```

## output
```
Test Accuracy = 0.900

```
