# Netflix Movies and TV Shows Clustering

Soumabha Sarkar

### **Abstract**

Netflix is a company that manages a large collection of TV shows and movies, streaming itanytime via online. This business is profitable because users make a monthly payment to access the platform. However, customers can cancel their subscriptions at any time. Therefore, the company must keep the users hooked on the platform and not lose their interest. This is where recommendation systems start to play an important role, providing valuable suggestions to users is essential.

### **Introduction**

Netflix’s recommendation system helps them increase their popularity among service providers as they help increase the number of items sold, offer a diverse selection of items, increase user satisfaction, as well as user loyalty to the company, and they are very helpful in getting a better understanding of what the user wants. Then it’s easier to get the user to make better decisions from a wide variety of movie products. With over 139 million paid subscribers (total viewer pool -300 million) across 190 countries, 15,400 titles across its regional libraries and 112 Emmy Award Nominations in 2018 — Netflix is the world’s leading Internet television network and the most-valued largest streaming service in the world. The amazing digital success story of Netflix is incomplete without the mention of its Recommender systems that focus on personalization. There are several methods to create a list of recommendations according to your preferences. You can use (Collaborative-filtering) and (Content-based Filtering) for recommendation.

### **Problem Statement**

This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third- party Netflix search engine. In 2018, they released an interesting report which shows that the number of TV shows on Netflix 
has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. In this project, you are required to do 
1.  Exploratory Data Analysis.
2. Understanding what type content is available in different countries.
3. Is Netflix increasingly focused on TV rather than movies in recent years?
4. Clustering similar content by matching text-based features.

### **Objectives**

Netflix Recommender recommends Netflix movies and TV shows based on a user's favorite movie or TV show. It uses a Natural Language Processing (NLP) model and a K-Means Clustering model to make these recommendations. These models use information about movies and TV shows such as their plot descriptions and genres to make suggestions. The motivation behind this project is to develop a deeper  understanding of recommender systems and create a model that can perform Clustering on comparable material by matching text-based attributes. Specifically, thinking about how Netflix create algorithms to tailor content based on user interests and behavior.

### **Data Description**

Attribute Information:

The dataset provided contains 7787 rows and 12 columns.

The following are the columns in the dataset:

● Show id: Unique identifier of the record in the dataset.

● Type: Whether it is a TV show or movie.

● Title: Title of the show or movie.

● Director: Director of the TV show or movie.

● Cast: The cast of the movie or TV show.

● Country: The list of the country in which a show/ movie is released or watched.

● Date added: The date on which the content was on boarded on the Netflix platform.

● Release year: Year of the release of the show/movie.

● Rating: The rating informs about the suitability of the content for a specific age group.

● Duration: Duration is specified in terms of minutes for movies and in terms of the number of seasons in the case of TV shows.

● Listed in: This columns species the category/genre of the content.

● Description: A short summary about the storyline of the content.

### **Tools Used**

The whole project was done using python, in google Collaboratory. Following libraries were used for analyzing the data and visualizing it and to build the model to predict the Netflix clustering 

● Pandas: Extensively used to load and wrangle with the dataset.

● Matplotlib: Used for visualization.

● Seaborn: Used for visualization.

● Datetime: Used for analyzing the date variable.

● Warnings: For filtering and ignoring the warnings.

● NumPy: For some math operations in predictions.

● Sklearn: For the purpose of analysis and prediction.

### Steps Involved:

The following steps are involved in the project:

**1. Handling missing values:**

We will need to replace blank countries with the mode (most common) country. It would be better to keep a director because it can be fascinating to look at a specific filmmaker's movie. As a result, we substitute the null values with the word 'unknown' for further analysis. There are very few null entries in the date_added fields thus we delete them.

**2. Duplicate Values Treatment:**

Duplicate values do not contribute anything to accuracy of results. Our dataset does not contain any duplicate values. 

### **Exploratory Data Analysis:**

Exploratory Data Analysis (EDA) as the name suggests, is used to analyze and investigate datasets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions. It also helps to understand the relationship between the variables (if any) and it will be useful for feature engineering. It helps to understand data well before making any assumptions, to identify obvious errors, as well as better understand patterns within data, detect Outliers, anomalous events, find interesting relations among the variables. After mounting our drive and fetching and reading the dataset given, we performed the Exploratory Data Analysis for it. To get the understanding of the data and how the content is distributed in the dataset, its type and
details such as which countries are watching more and which type of content is in demand etc. has been analyzed in this step.

Explorations and visualizations are as follows:

I. Proportion of type of content

II. Country-wise count of content

III. Total release for the last 10 years.

IV. Type and Rating-wise content count

V. Top 10 Actors on Netflix.

VI. Length distribution of movies.

VII. Season-wise distribution of TV shows.

VIII. Season wise distribution of TV shows.

IX. Longest TV shows.

X. Top 10 topics on Netflix.

XI. Extracting the features and creating the document term matrix.

XII. Top 10 actors in movies.

XIII. Top 10 actors in TV shows.

### **Data Preprocessing:**

Label encoding: We have used label encoder in type, country, rating and listed_in columns. By this process the values of the columns each label is assigned a unique integer based on alphabetical ordering.

Standardization: We have used StandardScaler to transform the data.

PCA: We have done PCA to show the clusters.

Min-Max Scaling: For each value in a feature, MinMaxScaler subtracts the minimum value in the feature and then divides by the range. It preserves the shape of the original distribution.

**Clustering:** 
Clustering (also called cluster analysis) is a task of grouping similar instances into clusters. More formally, clustering is the task of grouping the population of unlabeled data points into clusters in a way that data points in the same cluster are more similar to each other than to data points in other clusters. The clustering task is probably the most important in unsupervised learning, since it has many applications.

For example:

• Data analysis: often a huge dataset contains several large clusters, analyzing which separately, you can come to interesting insights.

• Anomaly detection: as we saw before, data points located in the regions of low density can be considered as anomalies.

• Semi-supervised learning: clustering approaches often help you to automatically label partially labeled data for classification tasks.

• Indirectly clustering tasks (tasks where clustering helps to gain good results): recommender systems, search engines, etc. 

• Directly clustering tasks: customer segmentation, image segmentation, etc.

**Building a clustering model:**

Clustering models allow you to categorize records into a certain number of clusters. This can help you identify natural groups in your  data. Clustering models focus on identifying groups of similar records and labeling the records according to the group to which they  belong. This is done without the benefit of prior knowledge about the groups and their characteristics. In fact, you may not even know exactly how many groups to look for. This is what distinguishes clustering models from the other machine-learning techniques—there is no predefined output or target field for the model to predict. These models are often referred to as unsupervised learning models, since there is no  external standard by which to judge the model's classification performance.

**Clusters Model Implementation**

1. Agglomerative Clustering
2. K-means Clustering
3. Agglomerative Clustering

The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. Next, pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects.

**K-means Clustering**

K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known, or labeled, outcomes. K-means algorithm works: To process the learning data, the K-means algorithm in data mining starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids. It halts creating and optimizing clusters when either:

• The centroids have stabilized — there is no change in their values because the clustering has been successful.

• The defined number of iterations has been achieved. K-means algorithm is an iterative algorithm that tries to partition the dataset into K pre-defined distinct non overlapping subgroups where each data point belongs to only one group. Ideal clustering K-means clustering is a  ethod of vector quantization, originally from signal processing that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. We created the sample data using build blobs and used range n_clusters to specify the number of clusters we wanted to utilize in k means. 

**Silhouette Coefficient or silhouette score**

Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1  to 1. Means clusters are well apart from each other and clearly distinguished. a= average intra-cluster distance i.e., the average distance between each point within a cluster.

**1. Silhouette’s Coefficient**

If the ground truth labels are not known, the evaluation must be performed utilizing the model itself. The Silhouette Coefficient is an  example of such an evaluation, where a more increased Silhouette Coefficient score correlates to a model with better-defined clusters. The Silhouette Coefficient is determined for each sample and comprised of two scores:

● Mean distance between the observation and all other data points in the same cluster. This distance can also be called a mean intra cluster distance. The mean distance is denoted by a.

● Mean distance between the observation and all other data points of the next nearest cluster. This distance can also be called a mean nearest-cluster distance. The mean distance is denoted by b.

The Silhouette Coefficient s for a single sample is then given as:

s =b−a
max(a,b)

Silhouette score is used to evaluate the quality of clusters created using clustering algorithms such as K-Means in terms of how well samples are clustered with other samples that are similar to each other. The Silhouette score is calculated for each sample of different clusters. To calculate the Silhouette score for each observation/data point, the following distances need to be found out for each observation belonging to all the clusters:

**2. Elbow Curve:**

The Elbow Curve is one of the most popular methods to determine this optimal value of k. The elbow curve uses the sum of squared distance (SSE) to choose an ideal value of k based on the distance between the data points and their assigned clusters. 

### **Conclusion**

1. We started by removing Nan values and converting the Netflix added date to year, month, and day using date time format.
2. We did feature engineering, which involved removing certain variables and preparing a dataframe to feed the clustering algorithms.
3. For the clustering algorithm, we utilized type, director, nation, released year, genre, and year.
4. Agglomerative Clustering, and K-means Clustering were utilized to build the model.
5. We've done null value treatment, feature engineering, and EDA since loading the dataset then completed assigned tasks.
6. Data set contains 7787 rows and 12 columns in that cast and director features contains a large number of missing values so we can drop it and we have 10 features for the further implementation.
7. We have two types of content TV shows and Movies (30.86% contains TV shows and 69.14% contains Movies) 
8. Most films were released in the years 2018, 2019, and 2020 and the United nation has the maximum content on Netflix.
9. The months of October, November, December and January had the largest number of films and television series released.
10. On Netflix, the Dramas genre contains the maximum content among all of the genres and the most of the content added in December month and less content in February.
11. By applying different clustering algorithms to our dataset. We get the optimal number of clusters is equal to 4.
12. We started by removing Nan values and converting the Netflix added date to year, month, and day using date time format.
13. We did feature engineering, which involved removing certain variables and preparing a dataframe to feed the clustering algorithms.
14. For the clustering algorithm, we utilized type, director, nation, released year, genre, and year.
15. Agglomerative Clustering, and K-means Clustering were utilized to build the model.
16. The final model we used was k-means clustering, which consisted of 2,3,4,5,6 clusters. 4 numbers of clusters give us good fitting.
