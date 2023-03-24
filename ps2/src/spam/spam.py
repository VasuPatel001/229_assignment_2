import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    split_msg = []
    normalize_msg = []
    split_msg = message.split(' ')
    for word in split_msg:
        normalize_msg.append(word.lower())
    return normalize_msg
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    all_words = []
    for message in messages:
        words = get_words(message)
        all_words.append(words)
    list_words = [word for sublist in all_words for word in sublist]

    unique_list = set(list_words)

    dict = {}
    i = 0
    for word in unique_list:
        if list_words.count(word) >= 5:
            dict[word] = i
            i+=1
    return dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    arr = np.zeros((len(messages), len(word_dictionary)), dtype = int)
    i = 0
    for message in messages:
        words = get_words(message)
        for key in word_dictionary.keys():
            if key in words:
                arr[i, word_dictionary[key]] = words.count(key)
            else:
                arr[i, word_dictionary[key]] = 0
        i += 1
    return arr
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    state = {}
    phi_k_y1 = np.zeros(matrix.shape[1])
    phi_k_y0 = np.zeros(matrix.shape[1])
    matrix_y1 = matrix[labels==1]
    matrix_y0 = matrix[labels==0]
    for k in range(matrix.shape[1]):
        phi_k_y1[k] = (1 + sum(matrix_y1[:,k])) / (matrix.shape[1] + sum(sum(matrix_y1)))
        phi_k_y0[k] = (1 + sum(matrix_y0[:,k])) / (matrix.shape[1] + sum(sum(matrix_y0)))

    count_label = np.bincount(labels)
    phi_y1 = count_label[1] / len(labels)
    phi_y0 = 1 - phi_y1

    state = {"phi_k_y1": phi_k_y1,
             "phi_k_y0": phi_k_y0,
             "phi_y1": phi_y1,
             "phi_y0": phi_y0}
    return state
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***

    pred = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        a = matrix[i,:] * model["phi_k_y1"]
        a1 = np.prod(a[a>0])

        b = matrix[i,:] * model["phi_k_y0"]
        b1 = np.prod(b[b>0])

        c = a1 *  model["phi_y1"] / (a1 *  model["phi_y1"] + b1 *  model["phi_y0"])
        if c>0.5:
            pred[i] = 1

        print(pred[i])

    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    #token_dict = {}
    #for k in range(len(dictionary)):
    token = np.log(model["phi_k_y1"]/model["phi_k_y0"])
    sorted_token = np.sort(token)
    inc_sorted_token = np.argsort(token)
    #dec_sorted_token = inc_sorted_token.reverse()
    print(sorted_token, inc_sorted_token)
    lst = []
    lst_dct = []
    for key in dictionary.keys():
        lst_dct.append(key)
    for i in range(5):
        index = inc_sorted_token[len(inc_sorted_token) -1 - i]
        word = lst_dct[index]
        lst.append(word)
    return lst
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    lst_accuracy = []
    for radius in radius_to_consider:
        predict = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius_to_consider)
        lst_accuracy.append(np.mean(predict == val_labels))
    lst_accuracy = np.array(lst_accuracy)
    return radius_to_consider[np.argmax(lst_accuracy)]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
