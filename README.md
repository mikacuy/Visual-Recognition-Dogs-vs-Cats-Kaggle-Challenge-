# Visual-Recognition-Dogs-vs-Cats-Kaggle-Challenge-

Uses Tensorflow in Python2.7 with OpenCV 2.4


Trained deep learning model from scratch and achieved a loss of 0.3355 in the Kaggle challenge. Output prediction can be found under results/
* To train the smaller network model:
```
$python train_small_network.py [directory of training images]
```
* To train the denser network model:
```
$python train_dense_network.py [directory of training images]
```
* To generate csv file containing test predictions for smaller network:
First unzip the smaller network in models/
```
$python generate_results_small_network.py [directory of test images] [output_filename.csv]
```
* To generate csv file containing test predictions for smaller network:
First create model file via training
```
$python generate_results_dense_network.py [directory of test images] [model filename] [output_filename.csv]
```
