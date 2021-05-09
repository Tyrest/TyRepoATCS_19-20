""" This module is python practice exercises to cover more advanced topics.
    Put the code for your solutions to each exercise in the appropriate function.
    Remove the 'pass' keyword when you implement the function.
    DON'T change the names of the functions!
    You may change the names of the input parameters.
    Put your code that tests your functions in the if __name__ == "__main__": section
    Don't forget to regularly commit and push to github.
    Please include an __author__ comment so I can tell whose code this is.
"""
__author__ = "Tyresty"
__version__ = 4.1

import random

# List Comprehension Practice

def even_list_elements(input_list):
    """ Use a list comprehension to return a new list that has
        only the even elements of input_list in it.
    """
    return [i for i in input_list if i % 2 == 0]

def list_overlap_comp(list1, list2):
    """ Use a list comprehension to return a list that contains
        only the elements that are in common between list1 and list2.
    """
    return [i for i in list1 if i in list2]


def div7list():
    """ Use a list comprehension to return a list of all of the numbers
        from 1-1000 that are divisible by 7.
    """
    return [i for i in range(7, 1000, 7)]


def has3list():
    """ Use a list comprehension to return a list of the numbers from
        1-1000 that have a 3 in them.
    """
    return [i for i in range(1, 1000) if '3' in str(i)]


def cube_triples(input_list):
    """ Use a list comprehension to return a list with the cubes
        of the numbers divisible by three in the input_list.
    """
    return [i**3 for i in input_list if i % 3 == 0]


def remove_vowels(input_string):
    """ Use a list comprehension to remove all of the vowels in the
        input string, and then return the new string.
    """
    return str([i for i in input_string if i not in ['a','e','i','o','u']])


def short_words(input_string):
    """ Use a list comprehension to return a list of all of the words
        in the input string that are less than 4 letters.
    """
    return [i for i in input_string.split(' ') if len(i) < 4]


# Challenge problem for extra credit:

def div_1digit():
    """ Use a nested list comprehension to find all of the numbers from
        1-1000 that are divisible by any single digit besides 1 (2-9).
    """
    return [x for x in range(1, 1000) if len([1 for y in range(2,10) if x % y == 0]) > 0]


# More practice with Dictionaries, Files, and Text!
# Implement the following functions:

def longest_sentence(text_file_name):
    """ Read from the text file, split the data into sentences,
        and return the longest sentence in the file.
    """
    return open(text_file_name, 'r').read().replace(';', '.').replace('!', '.').replace('?', '.').split('.')[[len(sentence) for sentence in open(text_file_name, 'r').read().replace(';', '.').replace('!', '.').replace('?', '.').split('.')].index(max([len(sentence) for sentence in open(text_file_name, 'r').read().replace(';', '.').replace('!', '.').replace('?', '.').split('.')]))]


def longest_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return the longest word in the file.
    """
    return clean(open(text_file_name, 'r').read()).split()[[len(sentence) for sentence in clean(open(text_file_name, 'r').read()).split()].index(max([len(sentence) for sentence in clean(open(text_file_name, 'r').read()).split()]))]


def num_unique_words(text_file_name):
    """ Read from the text file, split the data into words,
        and return the number of unique words in the file.
        HINT: Use a set!
    """
    return len(set(clean(open(text_file_name, 'r').read()).split()))

def clean(text):
    return text.replace(';', '').replace('!', '').replace('?', '').replace('.', '').replace(',','').replace('\'', '').replace('\"', '').replace(')', '').replace('(', '').lower()

def most_frequent_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return a tuple with the most frequently occuring word
        in the file and the count of the number of times it appeared.
    """
    dict = {word:0 for word in open(text_file_name, 'r').read().replace(';', '').replace('!', '').replace('?', '').replace('.', '').replace(',','').replace('\'', '').replace('\"', '').replace(')', '').replace('(', '').lower().split()}
    for word in open(text_file_name, 'r').read().replace(';', '').replace('!', '').replace('?', '').replace('.', '').replace(',','').replace('\'', '').replace('\"', '').replace(')', '').replace('(', '').lower().split():
        dict[word] += 1
    return max(dict, key=dict.get), dict[max(dict, key=dict.get)]


def date_decoder(date_input):
    """ Accept a date in the "dd-MMM-yy" format (ex: 17-MAR-85 ) and
        return a tuple in the form ( year, month_number, day).
        Create and use a dictionary suitable for decoding month names
        to numbers.
    """
    return int(date_input.split('-')[2]), {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12,}.get(date_input.split('-')[1]), int(date_input.split('-')[0])


def isit_random(lowest, highest, num_tries):
    """ Create and return a dictionary that is a histogram of how many
        times the random.randInt function returns each value in the
        range from 'lowest' to 'highest'. Run the randInt function a
        total number of times equal to 'num_tries'.
    """
    Dict = {i:0 for i in range(lowest, highest + 1)}
    for i in range(num_tries):
        Dict[random.randint(lowest, highest)] += 1
    return Dict


# Extra challenge problem: Surpassing Phrases!

"""
Surpassing words are English words for which the gap between each adjacent
pair of letters strictly increases. These gaps are computed without
"wrapping around" from Z to A.

For example:  http://i.imgur.com/XKiCnUc.png

Write a function to determine whether an entire phrase passed into a
function is made of surpassing words. You can assume that all words are
made of only alphabetic characters, and are separated by whitespace.
We will consider the empty string and a 1-character string to be valid
surpassing phrases.

is_surpassing_phrase("superb subway") # => True
is_surpassing_phrase("excellent train") # => False
is_surpassing_phrase("porky hogs") # => True
is_surpassing_phrase("plump pigs") # => False
is_surpassing_phrase("turnip fields") # => True
is_surpassing_phrase("root vegetable lands") # => False
is_surpassing_phrase("a") # => True
is_surpassing_phrase("") # => True

You may find the Python functions `ord` (one-character string to integer
ordinal) and `chr` (integer ordinal to one-character string) useful to
solve this puzzle.

ord('a') # => 97
chr(97) # => 'a'
"""

# Using the 'words' file on haiku, which are surpassing words? As a sanity check, I expect ~1931 distinct surpassing words.

def is_surpassing_phrase(input_string):
    """ Returns true if every word in the input_string is a surpassing
        word, and false otherwise.
    """
    oldDistance = 0
    for i in range(len(input_string) - 1):
        newDistance = ord(input_string[i]) - ord(input_string[i+1])
        if newDistance < oldDistance:
            return False
        oldDistance = newDistance
    return True

# I have more funky challenge problems if you need them!


if __name__ == "__main__":
    print(__author__ + "'s results:")
    # put your test code here
