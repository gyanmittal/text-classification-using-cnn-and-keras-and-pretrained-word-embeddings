Text Classification Example using CNN and Keras and pretrained Embeddings

- Dataset Link: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

- Dataset References: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

Create the data directory 

mkdir data

Downlad the data file zip into data directory:

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip -P data/

cd data

unzip smsspamcollection.zip 

you get your data file SMSSpamCollection in data directory

Reference:

- Convolutional Neural Networks for Sentence Classification (https://arxiv.org/pdf/1408.5882.pdf)

- A Convolutional Neural Network for Modelling Sentences (https://arxiv.org/pdf/1404.2188.pdf)

- Convolutional Neural Networks (http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes08-CNN.pdf)

Option 1:
Use Embedding generated at follwoing location:
- https://github.com/gyanmittal/word2vec_create

Option 2:
Use Pretrained Embeddings 
It can be downloaded from many locations, one of the embeddings repository location is as following:
- http://vectors.nlpl.eu/repository/


python3 train.py

It will create the relevant files in model folder

For working in  Virtual Environemnt:
https://docs.python.org/3/library/venv.html

Installing virtualenv:
- python3 -m pip install --user virtualenv
Creating a virtual environment:
- python3 -m venv env
Activating a virtual environment:
- source env/bin/activate
Installing the required packages:
- pip3 install -r requirement.txt
Leaving the virtual environment:
- deactivate

Other useful Files:
- retrain.py to further train existing trained model
- eval.py to evaluate to hardcoded results in the file
- eval-full.py to evaluate all the sms available in SMSSpamCollection data
