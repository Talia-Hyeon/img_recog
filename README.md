# img_recog
## 1. Bag-of-Words feature extraction
#### 1. extract dense SIFT descriptors & randomly sample the descriptors
* #### keypoints
![rand_feature](https://github.com/Talia-Hyeon/img_recog/assets/97673250/ac5b5a00-e1c4-4e20-bb14-2da05d1598dd)

#### 2. Reduce dimension (PCA)
Maintaining a 95% dispersion (128d -> 74d)

#### 3. Learn a visual dictionary (codebook) by k-means clustering
k=1500
#### 4. Encode an image vector

## 2. Support-Vector-Machine(SVM) training & test
* ####confusion matrix
* ![heatmap_1500](https://github.com/Talia-Hyeon/img_recog/assets/97673250/9cc80677-5c7f-4442-974f-74d89d0f782a)

## 3. image retrieval
![retrieval2](https://github.com/Talia-Hyeon/img_recog/assets/97673250/8ca296dc-baed-4f20-867d-9f975322cd8e)
