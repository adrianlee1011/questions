import nltk
import sys
import os
import string
import numpy as np
import collections

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    while True:
        # Prompt user for query
        query = set(tokenize(input("Query: ")))

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)

        stop = input("Continue? y/n: ")
        if stop == 'n':
            break


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    items = os.listdir(directory)
    files_dict = dict()
    for text in items:
        path = os.path.join(directory, text)
        with open(path, 'r', encoding="utf-8") as f:
            data = f.read().replace('\n', ' ')
            files_dict[text] = data
    return files_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    word_lst = nltk.word_tokenize(document)
    ret_word = []
    for word in word_lst:
        if not any(char in word for char in string.punctuation) and word not in nltk.corpus.stopwords.words("english"):
            ret_word.append(word.lower())

    return ret_word


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    docs_list = documents.keys()
    word_idf = dict()
    for current_doc, words in documents.items():
        for word in words:
            if word not in word_idf:
                count = 0
                for doc in docs_list:
                    if word in documents[doc]:
                        count += 1
                idf = np.log(len(docs_list)) / count
                word_idf[word] = idf

    return word_idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    all_files = {filename: 0 for filename in files}
    for key in files:
        for word in query:
            # if word in files[key]:
            word_count = 0
            for item in files[key]:
                if item == word:
                    word_count += 1
            tf_idf = word_count * idfs[word]
            all_files[key] += tf_idf

    n_top_files = {k: v for k, v in sorted(all_files.items(), key=lambda item: item[1])}
    top_files_list = list(n_top_files.keys())
    top_files_list.reverse()
    return top_files_list[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_list = list(sentences)
    sentence_idf = {sentence: 0 for sentence in sentence_list}
    for sentence in sentence_list:
        for word in query:
            if word in sentences[sentence]:
                sentence_idf[sentence] += idfs[word]
    
    n_top_sentences = {k: v for k, v in sorted(sentence_idf.items(), key=lambda item: item[1], reverse=True)}
    sort_by_term_density = []
    i = 0
    for key, value in n_top_sentences.items():
        if len(sort_by_term_density) == 0:
            sort_by_term_density.append((key, value))
        if value == sort_by_term_density[i][1]:
            sort_by_term_density.append((key, value))
    
    td = {}
    for i in range(len(sort_by_term_density)):
        words = sentences[sort_by_term_density[i][0]]
        count = 0
        for word in words:
            if word in query:
                count += 1
        qtd = count / len(words)
        td[sort_by_term_density[i][0]] = qtd

    top_sentences_sorted = {k: v for k, v in sorted(td.items(), key=lambda item: item[1], reverse=True)}

    top_sentences_list = list(top_sentences_sorted.keys())
    return top_sentences_list[:n]


if __name__ == "__main__":
    main()