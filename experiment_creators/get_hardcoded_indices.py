"""Determine hard-coded indices used in data_util.py and decode_history.py."""

import ast
import sys

def get_indices(experiment_name, wordslist):
    # Get filler indices.
    if 'variablefiller' in experiment_name:
        filler_words = ['PersonFillerTrain', 'FriendFillerTrain', 'EmceeFillerTrain', 'PoetFillerTrain', 'DrinkFillerTrain', 'DessertFillerTrain', 'PersonFillerTest', 'FriendFillerTest', 'EmceeFillerTest', 'PoetFillerTest', 'DrinkFillerTest', 'DessertFillerTest']
    if 'fixedfiller' in experiment_name:
        filler_words = ['Mariko', 'Pradeep', 'Sarah', 'Julian', 'Jane', 'John', 'latte', 'water', 'juice', 'milk', 'espresso', 'chocolate', 'mousse', 'cookie', 'candy', 'cupcake', 'cheesecake', 'pastry', 'Olivia', 'Will', 'Anna', 'Bill', 'coffee', 'tea', 'cake', 'sorbet']
    filler_indices = []
    for i in range(len(wordslist)):
        if wordslist[i] in filler_words:
            filler_indices.append(i)
    print("Filler indices: %s" % filler_indices)

    # Get padding index.
    padding_word = "zzz"
    print("Padding index: %d" % wordslist.index(padding_word))

    # Get query indices.
    queries = ['QDessert_bought','QDrink_bought','QEmcee','QFriend','QPoet','QSubject']
    for query in queries:
        if query in wordslist:
            print("%s index: %d" % (query, wordslist.index(query)))

    return

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    wordslist = ast.literal_eval(sys.argv[2])
    print(experiment_name)
    get_indices(experiment_name, wordslist)
