import sys

# Should return a set() of all words in the scrabble dictionary passed in at the given file path.
# You should implement this file loader.
def LoadWords(scrabble_dictionary_path):
    scrabble_words_file = open(scrabble_dictionary_path)
    scrabble_words = scrabble_words_file.read()
    return set(scrabble_words.split("\n"))

# Return a set() of ALL valid words from the input set `all_words` that differ by exactly one letter from `word`
# Hint: you may want to try using regular expressions here.
# This method assumes that you cleaned all_words so that it only has words of the same length
def FindAllNeighbors(word, all_words):
    word_neighbors = set()
    
    for comparison_word in all_words:
        off_by_one = False

        for word_character, comparison_character in zip(word, comparison_word):
            if word_character != comparison_character:
                if off_by_one:
                    off_by_one = False
                    break
                else:
                    off_by_one = True
                    
        if off_by_one:
            word_neighbors.add(comparison_word)

    return word_neighbors

# Use breadth first search to identify a path of legal words that link `input_word` and `target_word`.
# This function should return the words from the chain in a list.
def Chain(input_word, target_word, all_words):
    word_length = len(input_word)
    new_words = set()
    for current_word in all_words:
        if len(current_word) == word_length:
            new_words.add(current_word)

    all_words = new_words

    exploration = [input_word]
    all_paths = {}
    visited = set()
    path = []
    while len(exploration) > 0 and not found:
        last_word = exploration.pop(0)
        for current_word in FindAllNeighbors(last_word, all_words):
            if current_word == target_word:
                path.append(target_word)
            if current_word not in visited:
                visited.add(current_word)
                exploration.append(current_word)
                all_paths.update({current_word:last_word})

    if len(path) == 0:
        return []

    while path[0] != input_word:
        path.insert(0, all_paths.get(path[0]))

    return path

# Called from command line like "wordchain.py path_to_scrabble_dict.txt"
if __name__ == '__main__':
    scrabble_dict_path = sys.argv[1]
    all_words = LoadWords(scrabble_dict_path)
    print(Chain("WARM", "COLD", all_words)) # Should print ["TAP", "TOP"] as these words are already separated by only one edit.
    print(Chain("SUSHI", "PASTA", all_words)) # Could print ["CAP", "TAP", "TOP"], or any other valid sequence of words between these two.