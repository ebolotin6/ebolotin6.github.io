---
layout: page
title: Portfolio
show-title: false
permalink: /portfolio/
---

Hi, FYI this page hasn't been updated for some time! Updates are coming!

<h2 class="notopmargin">Current work</h2>
* Section being updated

<h2 class="notopmargin">Prior work</h2>
---
* Completed Master's Thesis on representation learning and face generation with variational autoencoders
* Implemented interesting variational autoencoders in Pytorch:
	* DFC-VAE, Beta-VAE, VAE-GAN, and researched implementing Google's DRAW architecture(Deep Recurrent Attention Writer)
* Finished a ton of preprocessing/prep work on Caption-from-image generation VAE
* Practing with GANs in pytorch. Notebook <a href="https://github.com/ebolotin6/nn/blob/master/gan_pytorch.ipynb" target="_blank">here</a>
* Base autoencoder in pytorch. Notebook <a href="https://github.com/ebolotin6/nn/blob/master/autoencoder_pytorch.ipynb" target="_blank">here</a>
* Completed course 1 of <a href="https://www.coursera.org/specializations/aml" target="_blank">Advanced Machine Learning Specialization</a> from the Higher School of Economics in Moscow (National Research University). 
	* Demo notebooks <a href="https://github.com/ebolotin6/HSE_public" target="_blank">here</a> 
	* Completed with honors: certificate <a href="https://www.coursera.org/account/accomplishments/certificate/YP8PCGTZ249L" target="_blank">here</a>
* Completed **Deep Learning Specialization** from Coursera. This is a <a href="https://www.coursera.org/specializations/deep-learning#courses" target="_blank">5-course</a> specialization on deep learning, which teaches both theory and application of deep learning. This specialization is taught by Prof. Andrew Ng (Adjunct Professor at Stanford University, who was also founding lead at Google Brain). 
	* See my certificate <a href="/DL_Specialization.pdf" target="_blank">here</a>. Some interesting projects that I completed were:

<h4 class="subheading">Convolutional networks</h4>
* Art generation with neural style transfer
* Face recognition
* YOLO for autonomous driving (car detection)
* Building ResNet (residual network) architecture from scratch

<h4 class="subheading">Recurrent networks</h4>
* Machine translation
* Trigger word detection
* Character level language model
* Jazz solo with an LSTM network
* Creating an “emojifier” to make text messages more expressive 


<h2 class="topmargin">Research Interests</h2>
---
The following section contains topics and areas that I am interested in.

<h4 class="subheading">Deep reinforcement learning</h4>
* The <a href="https://github.com/mila-iqia/babyai/tree/iclr19" target="_blank">BabyAI platform</a> (see <a href="https://openreview.net/forum?id=rJeXCo0cYX" target="_blank">paper</a> here), imitation learning, and grounded language learning with DL. The BabyAI platform is a testbed for improving the sample efficiency of deep RL algorithms with the goal of making dramatic advances in grounded language learning.
* I've reproduced the sample efficiency results posted in Table 3 of the <a href="https://openreview.net/forum?id=rJeXCo0cYX" target="_blank">BabyAI</a> paper for imitation learning from bot.
<br />

<h4 class="subheading">Neural networks</h4>
* Sequence to sequence RNN models for machine translation (pytorch)
* Transformers for neural machine translation
* Prototypical networks for few-shot classification
* Upcoming: graph neural networks

<h2 class="topmargin">Projects</h2>

The following section contains my (mostly completed) projects, organized by topic/area of data science.

### Deep Learning
---

#### <a href="https://github.com/ebolotin6/DS745/blob/master/project_3/Project_3_Bolotin.pdf" target="_blank"><strong>New York Times Comments Analysis</strong></a> (Fall 2019)
* **Description**: The objective of this project is perform text classification to predict New York Times article topics from the comments written by readers.
* **Dataset**: Data is from <a href="https://www.kaggle.com/aashita/nyt-comments" target="_blank">this kaggle competition</a>. This open-source dataset contains NYT reader’s comments and articles from January to May 2017 and January to April 2018. The comments data contains “over 2 million comments and 34 features”, while the articles data contains more than 9000 articles and 16 features.
* **Packages/languages**: R and Keras.

#### <strong>Object detection: understanding clouds from satellite Images</strong> (Fall 2019)
* **Description**: The objective of this project is to predict the presence of specific cloud types in satellite images. However, this is not a standard object detection challenge because each image may contain multiple labels and multiple classes *that are overlapping*. That is, one training image may have an area that is labeled as 2 or more distinct classes. So this is the main challenge: how to deal with overlapping labels for objects that have indefinite shapes (like clouds).
* **Dataset**: Data is from <a href="https://www.kaggle.com/c/understanding_cloud_organization/overview" target="_blank">this kaggle competition</a>. Dataset consists of ~9,300 unique photos. 5,600 in training, 3,700 in test.
* **Packages/languages**: Convolutional neural network written in Python + Keras + Tensorflow. Trained on GPU on Kaggle.
* **<font color="red">Work in progress</font>**


#### <a href="https://github.com/ebolotin6/iic_project/blob/master/notebooks/kaggle_notebooks/resnet_iic_kaggle.ipynb" target="_blank"><strong>Nature scene image classification</strong></a> (Summer 2019)
* **Description**: The objective of this project is to classify nature scene images into 6 distinct categories: *buildings, forest, glacier, mountain, sea* and *street*. See notebooks with *local* or *kaggle* suffix in the link above.
* **Dataset**: <a href="https://www.kaggle.com/puneet6060/intel-image-classification" target="_blank">Intel image competition</a> based dataset of 25k images labeled under 6 categories.
* **Accuracy**: 89% with 15 epochs (and possibly higher with more epochs)
* **Packages/languages**: Both pre-trained and original neural networks created for this purpose. Convolutional neural networks written in Python + Keras + Tensorflow. Trained on Kaggle with GPU.
<br /><br />

### Machine Learning
___
#### <a href="https://github.com/ebolotin6/DS740_portfolio/blob/master/final_project/Final_Project.pdf" target="_blank"><strong>Classification: predicting whether it will rain tomorrow</strong></a> (Summer 2019)
* **Description**: The objective of this project is to use non-neural ML methods to predict the occurrence of rain. See executive summary in repo above for high-level details.
* **Dataset dimensions**: 142k observations x 23 predictors
* Purpose of this project is predict whether it will rain tomorrow in Australia.
* **Methods used**: random forest, SVM, LDA (linear discriminant analysis), and a neural network. Recursive feature elimination with random forest used for subset selection.
* **Accuracy**: 85%
* **Language**: R

#### <a href="https://github.com/ebolotin6/DS740_portfolio/blob/master/midterm/midterm.pdf" target="_blank"><strong>Regression: predicting retail price of vehicles</strong></a> (Summer 2019)
* **Description**: The objective of this project is to use non-neural ML methods to predict the retail price of vehicles.  See midterm.pdf at link above for details.
* **Dataset dimensions**: 428 observations x 14 predictors
* This project demonstrates the prediction of retail price using 7 regression modeling methods. 
* **Methods used**: random forest, bagging (bootstrap aggregation), boosting, multiple linear regression, shrinkage/regularization methods of Ridge, Lasso, and Elastic-Net regression, and K-nearest neighbors. regression subsets selection (regsubsets) used for model selection.
* **Accuracy**: 96% R^2
* **Language**: R

#### <a href="https://github.com/ebolotin6/DS705_loan_defaults/blob/master/Project_Loan_Defaults.pdf" target="_blank"><strong>Classification: predicting loan defaults with logistic regression</strong></a> (Spring 2019)
* **Description**: The objective of this project is to improve bank margins by optimizing loan-making decisions. Said differently: the goal is to predict the financial risk that each customer poses to the bank.  See project pdf at link above for details.
* **Dataset dimensions**: The dataset used to train this model includes 50,000 loans and 30 variables.
* **Methods used**: Logistic modeling
* **Accuracy**: 80%
* **Language**: R

#### <a href="https://github.com/ebolotin6/Fargo_Health_Group_Case/" target="_blank"><strong>Regression: predicting health examinations</strong></a> (Summer 2018)
* **Description**: The objective of this project is to create a predictive model to forecast medical examinations for a health organization. Analysis is based on a Harvard Business Review case titled "The Fargo Health Group (FHG) Case". This project demonstrates: (1) data imputation and (2) creating a predictive model using multiple time-series autoregressive forecasting methods. Read report summary at link above for details.
* **Methods used**: ARIMA and Holt's exponential smoothing.
* **Language**: R
<br /><br />

### Optimization
---

#### <a href="https://github.com/ebolotin6/DS775/blob/master/project_3/Project_3_Report.ipynb" target="_blank"><strong>Simulation and Decision Analysis</strong></a> (Fall 2019)
* **Description**: The objective of this project is to perform simulation and decision analysis to determine the best decision (among a set of alternatives) for a given business scenario.
* **Packages/languages**: Python and ORtools (Google library)

#### <a href="https://github.com/ebolotin6/DS775/blob/master/project_2/Project_02_Homework/Project_2_Report.ipynb" target="_blank"><strong>Hyperparameter tuning with Scikit</strong></a> (Fall 2019)
* **Description**: The objective of this project is to determine the optimal hyperparameters for 2 models: RandomForest regressor and XGBoost classifier. Hyperparameter tuning is performed with 4 methods: GridSearchCV, RandomSearchCV, BayesianOptimization, and TPOT.
* **Dataset**: Sklearn diabetes dataset.
* **Packages/languages**: Python and Scikit-learn (sklearn)
<br /><br />

### Hadoop / Big Data
___
The list below consists of projects that solve big data-related business questions using the Hadoop framework. Languages/software used: Pig, Hive, Spark, Scala, Zeppelin, Python, Java, AWS (EMR, S3, EC2, Athena, Glue). (Spring 2019)
* **Final Project** -
	* **Part 1**: <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Final_Project_Part_1.md" target="_blank"><strong>Big data analysis with Hive and Scala on Spark</strong></a>
	* **Part 2**: <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Final_Project_Part_2.md" target="_blank"><strong>Parallel programming in Java: "Mailman's dilemma" algorithm</strong></a>. In part 2, I created an algorithm to solve for the most efficient way for a mailman to deliver mail, given N number of buildings.
	* **Part 3**: <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Final_Project_Part_3.md" target="_blank"><strong>Big data on AWS: Flight analysis project</strong></a>. In part 3, I selected and hosted a big data set of flight arrival times data on AWS, and answered questions about this dataset using Scala.
<br /><br />
* **Other projects** - 
	* <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Project_1_MapReduce.md" target="_blank"><strong>MapReduce and Python</strong></a>
	* <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Project_2_Pig.md" target="_blank"><strong>Apache Pig</strong></a>
	* <a href="https://github.com/ebolotin6/hadoop_ds730/blob/master/Project_3_Hive.md" target="_blank"><strong>Apache Hive</strong></a>
<br /><br />

### Natural Language Processing
___
#### <a href="https://github.com/ebolotin6/Twitter_Sentiment_Analysis/" target="_blank"><strong>Twitter Sentiment Analysis</strong></a> (Fall 2018)
* **Description**: The objective of this project is to answer the question: are people that talk about fitness happier than people that talk about media (tv, movies, youtube, etc.)? 
* **Methods used**: Twitter data is collected using REST and Stream APIs, then cleaned, organized, and (sentiment) analyzed. Sentiment analysis is performed using Natural Language Toolkit VADER Sentiment Analysis. All of this is done in Python. Statistical analysis performed in R. 
* **Other**: <a href="https://github.com/ebolotin6/Twitter_Sentiment_Analyzer/" target="_blank"><strong>Twitter Sentiment Analyzer</strong></a> is the standalone program created to perform sentiment analysis on Twitter data.
<br /><br />

### Visualization
___
#### <a href="https://github.com/ebolotin6/DS745/blob/master/project_1/Visualization_Report.pdf" target="_blank"><strong>Visualizing World Bank Data</strong></a> (Fall 2018)
* **Description**: The objective of this project is to demonstrate the iterative development of visualizations using principles of good design by Edward Tufte.
* **Languages**: Data cleaned, processed, and plotted in R.

#### <a href="https://github.com/ebolotin6/DS745/blob/master/project_2/Project_2_Bolotin.docx" target="_blank"><strong>Network Analysis of US Airports</strong></a> (Fall 2019)
* **Description**: This project demonstrates network analysis in R with data of air travel in the United States.
* **Packages/languages**: R, igraph, statnet

