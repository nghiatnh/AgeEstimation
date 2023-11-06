Vietnamese Age Estimation
=========================

## I. Introduction

Nowadays, marketing campaigns are often divided into small groups based on different age groups. Each age group has different characteristics, needs, and interests. Therefore, in order to segment customers into appropriate campaigns, it is necessary to know their ages. From there, we want to build a model to estimate the age of Vietnamese people. However, it is difficult and not realistic to predict a person's age accurately, so we focus on predicting the age group.

## II. Data Mining 

To build a model to predicts the age of Vietnamese people, we need a dataset that shows the dependences of Vietnamese people's age on their characteristics. 
In this project, we chose Vietnamese celebrities to collect data because their background information (age, image) is easy to find. However, using the images of celebrities also has many challenges. One of them is that they are often younger than their actual age.


### 1. Crawl data

- First, we come up with a list of celebrities (file txt) that covers most of the ages from 25 to 55 because this age range often appears in marketing campaigns. For each celebrity, we only take a certain number (about 70) of images in order to avoid the model biasing towards one or a few celebrities. 

- So how do we get their data? We use [Selenium](https://www.selenium.dev/) - a useful tool for collecting data automatically. Selenium simulates accurately how we operate with mouse and keyboard: access Google images, search for images of celebrities in the list, select ``"Open image in new tab"`` and get the image link.

- After this step, we have a list of image links corresponding to the artists. Then, we use the Python library ``requests`` to download and save to the corresponding folder of celebrities’ names to get the raw dataset.

<p align="center">
<img src="./imgs/data_structure.png" alt="structure" height="400"/> <img src="./imgs/baoanh_9.jpg" alt="demo" height="400"/>
</p>

### 2. Crop faces

- After observing a few images, we noticed some issues that need to be addressed:
The image may contain more than one person.
People in the images may not be the celebrities targeting.
Context factors (lighting, background, etc).
- For the solutions, we use the [OpenCV](https://opencv.org/) face detection model to identify all faces in images (including target celeb and others). To increase accuracy, we use **haarcascade_frontalface_default** model with ``minNeighbors=30``.
- As a result, our dataset only contains faces.

<p align="center">
<img src="./imgs/crop_faces.png" alt="demo" height="200"/>
</p>

### 3. Filter

- As mentioned above, the dataset is still confusing between celeb’s faces and unwanted objects. To remove this, we use a model that can verify whether the image is the face of the target celeb.
- First, we select one representative image of each celeb (called ``sample.jpg``). 
- In the next step, we use [Face-Comparison](https://github.com/m-lyon/face-comparison) to verify this side of view. This model uses an autoencoder to compress the images into embedding vectors and then uses L2Norm to calculate the distance between them. In case the distance is less than ``0.7``, the two faces are considered to belong to the same person. Otherwise, this image would be eliminated from the dataset because it belongs to unwanted objects.
- After this step, we have a dataset with **375** face images of over **20** different celebs. At this point, each celeb’s folder only contains their own faces, not mixed with other artists. Now, we could label the folder following the artist’s age.

### 4. Group data

To achieve the original goal of this project, we need to group the images into appropriate data groups (age). We divided our dataset into groups (25–30, 31–35, 36–40, 41–45, 46–50, and 51–55). Therefore, the number of classes is now 6 instead of 31, which makes the prediction better and simpler.

<p align="center">
<img src="./imgs/histplot.png" alt="demo" height="400"/> 
</p>

## III. Models

### 1. AlexNet architecture

<p align="center">
  <img src="imgs/alexnet.png" >
</p>

### 2. VGGNet architecture

- We using transfer learning technique with ``VGGNet19``, pretrained weight is [imagenet](https://www.image-net.org/).
- This is our model:

<p align="center">
  <img src="imgs/vgg19.png" >
</p>

- We do not retrain VGGNet19, we only train last 3 full-connection layers and get some good results.

### 3. Results

- Chúng tôi chia tập data thành 80% train và 20% test. Chúng tôi sử dụng ``cross entropy`` loss function và ``accuracy`` metric.
- Distribution của 2 tập data như sau:
<p align="center">
  <img src="imgs/test_distribution.png"/>
</p>

- Khi sử dụng mô hình Alexnet, chúng tôi nhận được kết quả là **34.57%** cho tập train và **31.58%** cho tập test. Có vẻ mô hình Alexnet đang bị underfitting.
- Khi sử dụng mô hình VGG19, chúng tôi được kết quả là **97%** cho tập train và **45.33%** cho tập test. Nhận thấy model đang bị overfitting khi bị bias vào một class chứa nhiều data nhất ở tập train, tôi đã thử upsampling data lên thành (``725`` samples) rồi chia lại tập train/test. Kết quả là tập validation được cải thiện đáng kể. Tuy nhiên vẫn xuất hiện overfitting khi mà accuary của tập train cao hơn rất nhiều so với tập test (**99.31%** cho tập train và **77.24%** cho tập test).
- Khi so sánh 2 model Alexnet và VGG19, chúng tôi có bảng benchmark như sau:

<div style="margin-left: auto; margin-right: auto; width: 550px;">

|          |Baseline| Alexnet | VGG19 | VGG19 no sampling |
|-----|--------|---------|-------|:-----------------:|
|Train|20.86%  |34.57%   |99.31% |97.00%             |
|Test |23.45%  |31.58%   |77.24% |45.33%             |

<p style="text-align: center;"><b>(Alexnet vs VGG19)</b></p>
</div>

- Có thể giải thích nguyên nhân việc model Alexnet bị overfitting vì nó không detect được những features của hình ảnh trong khi VGG19 đã học được những features đó nhờ pretrained weights của ``imagenet``.

## IV. Conclusion

- Sau khi thu thập data và qua những bước xử lý cẩn thận chúng tôi đã có một bộ data có thể sử dụng để train model dự đoán tuổi cho người Việt. Tuy bộ data còn ít sample nhưng vẫn cho thấy một tiềm năng có thể scale trong tương lai do không tốn quá nhiều công sử xử lý của sức người.
- Kết quả thu được từ 2 model chúng tôi đã sử dụng cho thấy việc sử dụng transfer learning có vẻ sẽ hữu ích hơn trong tương lai. Bên cạnh đó còn giúp chúng tôi lưu ý thêm về  đặc điểm của model trước khi apply để trán vấn đề overfiting và underfiting.

## IV. Far and further

### 1. Enhance dataset

- Our current dataset is still small and limited, so we want to improve its quantity and quality. We also want to include images of people besides celebrities. 
- We want to build a more accurate face verification model to improve the overall quality of the process.

### 2. Improve model:

- We want to use some boosting techniques (K-Fold, AdaBoost,...) to improve the performance of the model.
- Currently, the model is not performing well when using augmentation. In the future, we will try to find out the cause and apply more augmentation methods to increase the amount of data and improve the model for images with special factors (e.g., blurred, tilted, etc).
