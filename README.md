This text classifier is based on LSTM with Word2Vec features. This implementation is based on TensoeFlow. 

The network is trained on DBpedia documents, a structured version of Wikipedia. The corpus (included in data folder of this repo) contains 630K documents with 14 classes (list of classes are available at ./data/classes.txt). The corpus is divided into training (530K documents), validation (30K documents) and test (70K documents) sets.


A WORKING DEMO OF THIS IMPLEMENTATION IS AVAILABLE AT:
  http://demo-innovation.viseo.net/#/categorization

STEPS TO RUN THE CODE:
1. All the necessary paths are defined in paths.properties file. Make sure that the paths defined in this file already exist.

2. For preprocessing the documents, launch
       python3 preprocess.py

   It will create train_preprocessed.pickle validation_preprocessed.pickle and test_preprocessed.pickle files under data folder.

3.Training Word2Vec embeddings: Word2Vec embeddings is to be trained using
       python3 word_embedder_gensim.py

   To know all the available options run
       python3 word_embedder_gensim.py -h


4. Training the network: the LSTM network is to be trained using tnn_w2v.py. For all available options run
       python3 rnn_w2v.py -h

   The same file can be used to batch test with the test documents in ./data/test.csv file.


5. Demo; to launch and test the trained model on any piece of text, launch
       python3 TextCategorizer.py