# Training

In this section we will show you how to train the models yourself.

# Obtaining the Dataset

## IMDB
1. Obtain the IMDB dataset, you can download the Large Movie Review Dataset [HERE](https://ai.stanford.edu/~amaas/data/sentiment/)
2. Extract the zipped file.
3. Make a folder in the `datasets/raw` called `imdb_reviews`. And make folders inside that called `pos` and `neg`.
4. Go into the folders of the unzipped file you downloaded, until you find some labeled `test` and `train`, we want to combine these datasets, to do that, go into the `test` and `train` folders, you should see some folders named `pos` and `neg`, these are the positive and negative reviews. Copy the text files in the `neg` folders into the `imdb_reviews/neg` folder, do the same for the pos folders, but make sure to copy to the `imdb_reviews/pos` folder.
5. Now you have your reviews sorted into positive and negative, in plaintext files.

## Yelp
Secondly obtain the Yelp Reviews Dataset, you can do this by:
1. Going to the [Yelp Dataset](https://www.yelp.com/dataset/download) page.
2. Input your details.
3. Click Download
4. Click on Download JSON
5. Unzip the Dataset
6. Copy the `yelp_academic_dataset_review.json` to the `datasets/raw` folder in the repo. And rename the file `yelp_academic_dataset_review.json` file to `yelp_reviews.json`
7. Now you have your Yelp dataset in place. good job!

# Aligning the datasets
Next we need to combine these datasets into a CSV file with 2 fields, `text` and `sentiment`, the former being the text of the review and the later being weather its positive or negative(1 or 0).

1. Run the `prep_imdb_reviews.py` file in the `prepare_datasets` directory.
2. Run the `prep_yelp_reviews.py` file in the `prepare_datasets` directory.

# Combining the datasets
Next we need to combine the datasets into a single file.
1. Run the `sample_from_datasets.py` file in the `prepare_datasets` directory, this will make a combined CSV file with the data inside, feel free to tweak the `TARGET_SAMPLE_SIZE` to whatever you wish(as long as the dataset is big enough), its just the number of entries it will try and put into the final file.
2. Now you have a combined dataset

# Preproccessing the datasets
Next we need to preproccess the datasets so it can be used for training.
1. Run the `preprocess.py file`
2. Now you have a preproccessed dataset, and a tokenizer(in the tokenizer directory), which we will use later.

# Installing the Embeddings
Now you need to download some pretrained embeddings, we are using GloVe for this project.
1. Download the `840B tokens` model, from (HERE)[https://nlp.stanford.edu/projects/glove/]
2. Extract the compressed file and put the txt file in `./datasets/raw`.

# Training the Model
Now we need to train the model to get it output as a file.
1. Run the `train.py` file
2. Now you have your model, you can now make predictions 