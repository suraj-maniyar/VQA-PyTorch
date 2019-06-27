# VQA-PyTorch

## Implementation Details
My implementation of VQA in PyTorch. I have used Balanced Real Images from the [VQA dataset](https://visualqa.org/download.html). Download the annotations and questions and extract it into the data/ folder.  
I have used GloVe embedding (300 dimensional) for modeling the input questions. Download the GloVe embedding from [this](http://nlp.stanford.edu/data/glove.6B.zip) link and extract the file glove.6B.300d.txt into the data folder.  


## Approach
I have only used those images for which the answer appears atleast twice in the dataset. Based on this approach, I have created a list of answers which appear only once in the entire dataset and hance are ignored. These answers are pickled in the list ignore_list.pkl. This helps reduce the vocabulary size which happens to be the dimension of the output vector.  

The CNN I used for feature extraction from iamges is ResNet-18. The input images are resized to 224x224 before feeding into the network. So, I have already extracted the feature vectors for each of those images and dumped it into a pickle file. Download the features from [this](https://www.dropbox.com/sh/2funpz0cf1lx2tu/AACpyUQlj19GJDf-cvwOUhFna?dl=0) link and put it into the dumps/ folder.  

For data augmentation, I have flipped the training images. The image_features.pkl file consists of list of lists in this format: [[feature1, feature2], path, question, answer] where feature1 is the 512 dimensional features vector for that image and feature2 is the feature vector for a 180 degree flipped image. In case of validation data, the val_features.pkl file consists of list of lists in the format: [feature, path, question, answer].  



I used an LSTM which has 2 layers each with 64 hidden units. 

### Approach 1
The image features (512 dimensional) is passed through a ense layers to reduce it to a dimension of 256. The output from LSTM model is also passed through a dense layer to make it of dimension 256. These 2 feature vectors were **concatenated** before feeding them to a dense layer of dimension equal to the vocabulary size. The output is passed through softmax for classification. This did not give very good results and the validation accuracy was stuck at 29%.  

### Approach 2
One reason for the low accuracy was that the 256 dimensional feature vectors extracted from both the image and the question were having different distributions (mean and variances). So I applied a BatchNorm to each of the features before concatenation. This increased the accuracy to 33%.  

### Approach 3
Another alternative I tried was to **multiply** the features instead of cancatenating them. So now the 2 feature vectors of 256 dimension are multiplied element-wise which gives a validation accuracy of 38%.  


## Results
I have trained the network for over 300 epochs and it very easily overfits the data. The best validation accurracy I obtained is 38.11%.  
The following figure shows the learning curves I got:  

<p align="center">
  <img src="assets/learning_curves.png" width="100%">
</p>


Following are few examples of images from the validation set:  

<p align="center">
  <img src="assets/COCO_val2014_000000174042.jpg" width="50%">
</p>

Question: "What is the dog doing?"  
Answer: "running"  


<p align="center">
  <img src="assets/COCO_val2014_000000035594.jpg" width="50%">
</p>

Question: "What is in the image"  
Answer: "cat"  

Question: "Where is this picture taken"  
Answer: "mountain"  

Question: "What animal is shown in the picture"  
Answer: "eleph"  


<p align="center">
  <img src="assets/COCO_val2014_000000123642.jpg" width="50%">
</p>

Question: "What food is this"  
Answer: "pizza"  

Question:  "What is the stuffing"  
Answer: "ketchup"  

Question: "What bread is used"  
ANswer:  "wheat"

