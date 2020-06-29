#!/usr/bin/env python
# coding: utf-8

# ## STEP1 download and prepare the dataset

# In[1]:


"""
Build a semantic contennt recommendation system with AWS SageMaker

Data source: 
    20newsgroups dataset contained in scikit-learn package

Usage: 
    The model automatically extract semantic information from the news data, and 
    identify similar news articles from the corpus and provide content recommendations
    for similar news items based on the ones that are selected
    
    The model is trained and deployed in the Amazon SageMaker Notebook instance,
    using built-in algorithms for Neural Topic Model(NTM) and K-Nearest Neighbour(K-NN)

"""
import numpy as np
import os
import matplotlib.pyplot as plt
import sagemaker
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_header, strip_newsgroup_quoting, strip_newsgroup_footer
newsgroups_train = fetch_20newsgroups(subset='train')['data']
newsgroups_test = fetch_20newsgroups(subset = 'test')['data']
NUM_TOPICS = 30
NUM_NEIGHBORS = 10
BUCKET = 'sagemaker-content-recommender'
PREFIX = '20newsgroups'


# In[4]:


for i in range(len(newsgroups_train)):
    newsgroups_train[i] = strip_newsgroup_header(newsgroups_train[i])
    newsgroups_train[i] = strip_newsgroup_quoting(newsgroups_train[i])
    newsgroups_train[i] = strip_newsgroup_footer(newsgroups_train[i])


# In[5]:


newsgroups_train[1]


# In[6]:


# Tokenize the data to numeric format by assigning a token to each word
# Use WordNetLemmatizer as a lemmatizer from NLTK pacakage
# Use CountVectorizer to perform token counting
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if len(t) >= 2 and re.match("[a-z].*",t) 
                and re.match(token_pattern, t)]


# In[7]:


import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vocab_size = 2000
print('Tokenizing and counting, this may take a few minutes...')
start_time = time.time()
vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
                             tokenizer=LemmaTokenizer(), max_features=vocab_size, max_df=0.95, min_df=2)
vectors = vectorizer.fit_transform(newsgroups_train)
vocab_list = vectorizer.get_feature_names()
print('vocab size:', len(vocab_list))

# random shuffle
idx = np.arange(vectors.shape[0])
newidx = np.random.permutation(idx) # this will be the labels fed into the KNN model for training
# Need to store these permutations:

vectors = vectors[newidx]

print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))


# In[11]:


import scipy.sparse as sparse
vectors = sparse.csr_matrix(vectors, dtype=np.float32)
print(type(vectors), vectors.dtype)


# In[14]:


# Convert data into training and validation data
n_train = int(0.8 * vectors.shape[0])

# split train and test
train_vectors = vectors[:n_train, :]
val_vectors = vectors[n_train:, :]

# further split test set into validation set (val_vectors) and test  set (test_vectors)

print(train_vectors.shape,val_vectors.shape)


# In[15]:


# define training and validation path
from sagemaker import get_execution_role

role = get_execution_role()

bucket = BUCKET
prefix = PREFIX

train_prefix = os.path.join(prefix, 'train')
val_prefix = os.path.join(prefix, 'val')
output_prefix = os.path.join(prefix, 'output')

s3_train_data = os.path.join('s3://', bucket, train_prefix)
s3_val_data = os.path.join('s3://', bucket, val_prefix)
output_path = os.path.join('s3://', bucket, output_prefix)
print('Training set location', s3_train_data)
print('Validation set location', s3_val_data)
print('Trained model will be saved at', output_path)


# In[16]:


# convert the raw vectors into RecordIO format, and use n_parts to breaks dataset into shards for destributed training
def split_convert_upload(sparray, bucket, prefix, fname_template='data_part{}.pbr', n_parts=2):
    import io
    import boto3
    import sagemaker.amazon.common as smac
    
    chunk_size = sparray.shape[0]// n_parts
    for i in range(n_parts):

        # Calculate start and end indices
        start = i*chunk_size
        end = (i+1)*chunk_size
        if i+1 == n_parts:
            end = sparray.shape[0]
        
        # Convert to record protobuf
        buf = io.BytesIO()
        smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
        buf.seek(0)
        
        # Upload to s3 location specified by bucket and prefix
        fname = os.path.join(prefix, fname_template.format(i))
        boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
        print('Uploaded data to s3://{}'.format(os.path.join(bucket, fname)))
split_convert_upload(train_vectors, bucket=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)
split_convert_upload(val_vectors, bucket=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)


# ## STEP2 Train and deploy the topic model
# 
# #### create and run the training job
# The built in algorithms are stored as docker conatiners in Amazon ECR, need to specify container and region

# In[17]:


import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'ntm')


# In[18]:


# Use estimator to specify the infrastructure for model training
sess = sagemaker.Session()
ntm = sagemaker.estimator.Estimator(container,
                                    role, 
                                    train_instance_count=2, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path=output_path,
                                    sagemaker_session=sess)
# set the hyperparamter
ntm.set_hyperparameters(num_topics=NUM_TOPICS, feature_dim=vocab_size, mini_batch_size=128, 
                        epochs=100, num_patience_epochs=5, tolerance=0.001)


# Use ShardedBys3Key to spread data to different workers, each worker receives a different portion of the full data set, this method is more efficient that FullyReplicated data.

# In[19]:


# Model Training, the topic model is trained with the NTM algorithms
from sagemaker.session import s3_input
s3_train = s3_input(s3_train_data, distribution='ShardedByS3Key') 
ntm.fit({'train': s3_train, 'test': s3_val_data})


# ### deploy the topic model
# 
# two ways to deploy the model:
# 1. Create one inference at a time, set up a persistent endpoint using SageMaker hosting services
# 2. To get inferences for an entire dataset, use SageMaker batch transform

# In[24]:


# deploy the topic model with batch transform
np.savetxt('trainvectors.csv',
           vectors.todense(),
           delimiter=',',
           fmt='%i')
batch_prefix = '20newsgroups/batch'

train_s3 = sess.upload_data('trainvectors.csv', 
                            bucket=bucket, 
                            key_prefix='{}/train'.format(batch_prefix))
print(train_s3)
batch_output_path = 's3://{}/{}/test'.format(bucket, batch_prefix)

ntm_transformer = ntm.transformer(instance_count=1,
                                  instance_type ='ml.m4.xlarge',
                                  output_path=batch_output_path
                                 )
ntm_transformer.transform(train_s3, content_type='text/csv', split_type='Line')
ntm_transformer.wait()


# In[25]:


# Option B: Use SageMaker hosting service

# ntm_predictor = ntm.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# Ensure that the input payload is seralized in a machine readable format and the output is deserialized into a 
# human readbale format
# from sagemaker.predictor import csv_serializer, json_deserializer

# ntm_predictor.content_type = 'text/csv'
# ntm_predictor.serializer = csv_serializer
# ntm_predictor.deserializer = json_deserializer

# Extract the topic vectors for the training data that we use in the K-NN model
# predictions = []
# for item in np.array(vectors.todense()):
#     np.shape(item)
#     results = ntm_predictor.predict(item)
#     predictions.append(np.array([prediction['topic_weights'] for prediction in results['predictions']]))
    
# predictions = np.array([np.ndarray.flatten(x) for x in predictions])
# topicvec = train_labels[newidx]
# topicnames = [categories[x] for x in topicvec]


# ### Explore the topic model
# 
# Use the T-SNE(Distributed Stocahstic Neighbour Embedding) plot to visualize the topic vectors
# 
# The T-SNE is a non-linear technique for dimensionality reduction which aims to ensure that the distance between nearest neighbors in the original high dimensional space is preserved in the resulting lower dimensional space.
# 
# Setting the number of dimensions to 2 so that we could visualize the topic in 2D space

# In[29]:


from sklearn.manifold import TSNE
import pandas as pd
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
tsne_results = tsne.fit_transform(predictions)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tsnedf = pd.DataFrame()
tsnedf['tsne-2d-one'] = tsne_results[:,0]
tsnedf['tsne-2d-two'] = tsne_results[:,1]
tsnedf['Topic']=topicnames
plt.figure(figsize=(25,25))
sns.lmplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue='Topic',
    palette=sns.color_palette("hls", NUM_TOPICS),
    data=tsnedf,
    legend="full",
    fit_reg=False
)
plt.axis('Off')
plt.show()


# ## STEP3 train and deploy the content recommendation model

# In[31]:


# create a dictionary which links the shuffled labels to the original labels in the trainin data
labels = newidx 
labeldict = dict(zip(newidx,idx))


# In[32]:


# store the training data in S3 bucket
import io
import sagemaker.amazon.common as smac


print('train_features shape = ', predictions.shape)
print('train_labels shape = ', labels.shape)
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, predictions, labels)
buf.seek(0)

bucket = BUCKET
prefix = PREFIX
key = 'knn/train'
fname = os.path.join(prefix, key)
print(fname)
boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# ### create a K-NN estimator
# 
# The model will be trained to return the nearest neighbors given a particular test topic

# In[33]:


def trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data=None):
    """
    Create an Estimator from the given hyperparams, fit to training data, 
    and return a deployed predictor
    
    """
    # set up the estimator
    knn = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
        get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.c4.xlarge',
        output_path=output_path,
        sagemaker_session=sagemaker.Session())
    knn.set_hyperparameters(**hyperparams)
    
    # train a model. fit_input contains the locations of the train and test data
    fit_input = {'train': s3_train_data}
    knn.fit(fit_input)
    return knn

hyperparams = {
    'feature_dim': predictions.shape[1],
    'k': NUM_NEIGHBORS,
    'sample_size': predictions.shape[0],
    'predictor_type': 'classifier' ,
    'index_metric':'COSINE'
}
output_path = 's3://' + bucket + '/' + prefix + '/knn/output'
knn_estimator = trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path)


# ### Deploy the content recommendation model
# The k_NN model will return all the cosine distances 
# To build the engine, we need to get top-k suggestions from the engine

# In[34]:


def predictor_from_estimator(knn_estimator, estimator_name, instance_type, endpoint_name=None): 
    knn_predictor = knn_estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                        endpoint_name=endpoint_name,
                                        accept="application/jsonlines; verbose=true")
    knn_predictor.content_type = 'text/csv'
    knn_predictor.serializer = csv_serializer
    knn_predictor.deserializer = json_deserializer
    return knn_predictor
import time

instance_type = 'ml.m4.xlarge'
model_name = 'knn_%s'% instance_type
endpoint_name = 'knn-ml-m4-xlarge-%s'% (str(time.time()).replace('.','-'))
print('setting up the endpoint..')
knn_predictor = predictor_from_estimator(knn_estimator, model_name, instance_type, endpoint_name=endpoint_name)


# In[35]:


# process the test data so that you can run inferences
def preprocess_input(text):
    text = strip_newsgroup_header(text)
    text = strip_newsgroup_quoting(text)
    text = strip_newsgroup_footer(text)
    return text    
    
test_data_prep = []
for i in range(len(newsgroups_test)):
    test_data_prep.append(preprocess_input(newsgroups_test[i]))
test_vectors = vectorizer.fit_transform(test_data_prep)

test_vectors = np.array(test_vectors.todense())
test_topics = []
for vec in test_vectors:
    test_result = ntm_predictor.predict(vec)
    test_topics.append(test_result['predictions'][0]['topic_weights'])

topic_predictions = []
for topic in test_topics:
    result = knn_predictor.predict(topic)
    cur_predictions = np.array([int(result['labels'][i]) for i in range(len(result['labels']))])
    topic_predictions.append(cur_predictions[::-1][:10]) 


# ### Explore content recommendation model
# 
# Now that the predictions are obtained. Plot the topics distributions of the 
# test topics, compared to the closest K topics recommedned by the k-NN model

# In[36]:


# set your own k.
def plot_topic_distribution(topic_num, k = 5):
    
    closest_topics = [predictions[labeldict[x]] for x in topic_predictions[topic_num][:k]]
    closest_topics.append(np.array(test_topics[topic_num]))
    closest_topics = np.array(closest_topics)
    df = pd.DataFrame(closest_topics.T)
    df.rename(columns ={k:"Test Document Distribution"}, inplace=True)
    fs = 12
    df.plot(kind='bar', figsize=(16,4), fontsize=fs)
    plt.ylabel('Topic assignment', fontsize=fs+2)
    plt.xlabel('Topic ID', fontsize=fs+2)
    plt.show()


# In[37]:


plot_topic_distribution(18)


# In[38]:


plot_topic_distribution(25)


# In[39]:


plot_topic_distribution(5000)


# ## Cleanup
# 
# 

# In[40]:


ntm_predictor.delete_endpoint()
knn_predictor.delete_endpoint()


# In[ ]:




