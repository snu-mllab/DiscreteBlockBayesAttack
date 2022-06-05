from utils import create_input_files, train_word2vec_model

if __name__ == '__main__':
    create_input_files(csv_folder='yelp',
                       output_folder='',
                       sentence_limit=50,
                       word_limit=50,
                       min_word_count=2)

    train_word2vec_model(data_folder='',
                         algorithm='skipgram')
