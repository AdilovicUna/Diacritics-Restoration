from asyncore import read
from curses.ascii import isalpha, isupper
import sys
import numpy as np
import lzma
import pickle

from os import path
from sklearn.neural_network import MLPClassifier

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

def clean(text):
    # removes punctuation and lowers all of the charactersß
    result = []

    for word in text:
        word = word.lower()
        new_word = ""
        for i in word:
            if isalpha(i) or i in LETTERS_DIA:
                new_word += i
        if not new_word.isspace():
            result.append(new_word)

    return result


def read_file(filename,clean=True):
    # reads a file and splits it into "clean" words
    with open(filename, "r", encoding="utf-8-sig") as file:
        if clean:
            result = clean(file.read().split())
        else:
            result = file.read()
    file.close()

    return result

def build_dictionary(data,target):
    # creates a dictionary form the corpus
    result = {}

    for i in range(len(data)):
        word = data[i]
        d_word = target[i]

        if not word in result:
            result[word] = []

        if not d_word in result[word]:
            result[word].append(d_word)

    return result

def build_one_hot_encoder_dictionary():
    result = {}

    # assume we are using all "normal" characters
    n = 128
    chars = [chr(i) for i in range (n)]

    for i in range (1,n+1):
        zeros = ([0] * (i-1)) + [1] + ([0] * (n-i))
        result[chars[i-1]] = zeros
    
    return result

def encode_word(word,encoder):
    return [i for w in word for i in encoder[w]]

def get_best_match(word,word_list):
    # this version exists
    if word in word_list:
        return word

    # determine which one matches the best
    chars = list(word)
    result = {}
    for option in word_list:
        count = 0
        for op,ch in zip(option,chars):
            if op == ch:
                count += 1
        result[option] = count

    return max(result, key=result.get)

def reconstruct(original,dia):
    index = 0
    final = ""
    
    for ch in original:
        if isalpha(ch):
            if isupper(ch):
                final += dia[index].upper()
            else:
                final += dia[index]
            index += 1
        else:
            final += ch
    return final

def train(data,target,encoder):
    encoded_data = []
    encoded_target = []

    n = 3
    padding = n*" "

    for i in range(len(data)):

        length = len(data[i])
        word = padding + data[i] + padding
        t_word = padding + target[i] + padding

        for i in range (length):

            mid_index = i + n
            mid_char = word[mid_index]
            t_mid_char = t_word[mid_index]

            if mid_char in LETTERS_NODIA: # its a "special" letter
                window = word[i:(2*n+1)+i]
                encoded_word = encode_word(window,encoder)

                encoded_data.append(encoded_word)
                encoded_target.append(t_mid_char)

    model =  MLPClassifier()
    return model.fit(encoded_data,encoded_target)

def predict(model, dictionary, encoder, original):
    data = clean(original.split())
    result = []

    n = 3
    padding = n*" "

    for i in range(len(data)):

        curr = data[i]
        exists = False

        # is it in the dictionary?
        if curr in dictionary:
            exists = True

        length = len(curr)
        word = padding + curr + padding

        pred_word = ""

        for i in range (length):

            mid_index = i + n
            mid_char = word[mid_index]

            if mid_char in LETTERS_NODIA: # its a "special" letter
                window = word[i:(2*n+1)+i]
                encoded_word = np.array(encode_word(window,encoder)).reshape(1,-1)
                pred_word += model.predict(encoded_word)[0]
            else:
                pred_word += mid_char
        
        if exists:
            if len(dictionary[curr]) == 1:
                pred_word = dictionary[curr]
            else:
                pred_word = get_best_match(pred_word,dictionary[curr])

        result.append(pred_word)

    return reconstruct(original,''.join([''.join(i) for i in result]))

def test_accuracy(original_words,new_words):
    total = sum([len(i) for i in original_words])
    correct = sum([1 if original_words[i][j] == new_words[i][j] else 0 for i in range(len(original_words)) for j in range(len(original_words[i]))])
    return 100 * correct/total

def test(model, dictionary, encoder, original, output=False):
    text = original.translate(DIA_TO_NODIA)
    final = predict(model, dictionary, encoder, text)
    if output:
        print(final)
    print(test_accuracy(original.split(),final.split()))

def main(test_dataset,output):
    if not path.exists("diacritics_restoration.model"): # train a model
            
        data = read_file("NODIA.txt")
        target = read_file("DIA.txt")

        encoder = build_one_hot_encoder_dictionary()

        model = train(data,target,encoder)

        dictionary = build_dictionary(data,target)

        with lzma.open("diacritics_restoration.model", "wb") as file:
            pickle.dump((model,dictionary,encoder), file)

    else: # predict

        with lzma.open("diacritics_restoration.model", "rb") as file:
            model, dictionary, encoder = pickle.load(file)

        if not test_dataset:
            original = ""
            for line in sys.stdin:
                original += line
            test(model, dictionary, encoder, original,output)
        else:
            test(model, dictionary, encoder, read_file("diacritics-etest.txt",False),output)
            test(model, dictionary, encoder, read_file("diacritics-dtest.txt",False),output)

if __name__ == "__main__":
    test_dataset = False
    output = False

    if len(sys.argv) > 1 and sys.argv[1] == "--test_dataset":
        test_dataset = True

    if (len(sys.argv) > 1 and sys.argv[1] == "--output") or (len(sys.argv) > 2 and sys.argv[2] == "--output"):
        output = True
    
    main(test_dataset,output)