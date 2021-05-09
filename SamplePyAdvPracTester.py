""" This module has sample tests for the PythonAdvPractice_2019.py file
"""
import math
f=0
extra=0
num_exer=5

try:
    import PythonAdvPractice_2019 as pap

    listOdd = [ 1, 3, 5, 7 ]
    listEven = [ 2, 4, 6, 8 ]
    listMixed = [ 2, 3, 4 ]
    listdups1 = [ 6, 6, 7 ]
    listdups2 = [ 3, 3, 3, 2, 2, 1, 0]
    list3 = [ 4, 9, 2, 3, 3, 5]
    emptyList = []

    print("Testing with these lists:")
    print("listOdd = " + str(listOdd))
    print("listEven = " + str(listEven))
    print("listMixed = " + str(listMixed))
    print("listdups1 = " + str(listdups1))
    print("listdups2 = " + str(listdups2))
    print("list3 = " + str(list3))
    print("emptyList = " + str(emptyList))

# def even_list_elements(input_list):
    """ Use a list comprehension/generator to return a new list that has
        only the even elements of input_list in it.
    """
    try:
        test1 = pap.even_list_elements(listEven)
        test2 = pap.even_list_elements(listOdd)
        test3 = pap.even_list_elements(listMixed)
        test4 = pap.even_list_elements(listdups1)
        test5 = pap.even_list_elements(emptyList)
        if len(test1) != 4:
            print("FAILED: even_list_elements(listEven) returned: " + str(test1))
            f += 0.2
        else:
            print("passed: even_list_elements(listEven) with: " + str(test1))
        if len(test2) != 0:
            print("FAILED: even_list_elements(listOdd) returned: " + str(test2))
            f += 0.2
        else:
            print("passed: even_list_elements(listOdd) with: " + str(test2))
        if len(test3) != 2:
            print("FAILED: even_list_elements(listMixed) returned: " + str(test3))
            f += 0.2
        else:
            print("passed: even_list_elements(listMixed) with: " + str(test3))
        if len(test4) != 2:
            print("FAILED: even_list_elements(listdups1) returned: " + str(test4))
            f += 0.2
        else:
            print("passed: even_list_elements(listdups1) with: " + str(test4))
        if len(test5) != 0:
            print("FAILED: even_list_elements(emptyList) returned: " + str(test5))
            f += 0.2
        else:
            print("passed: even_list_elements(emptyList) with: " + str(test5))
    except Exception as ex:
        print(ex)
        print("FAILED: even_list_elements threw an exception.")
        f += 1


# def list_overlap_comp(list1, list2):
    """ Use a list comprehension/generator to return a list that contains
        only the elements that are in common between list1 and list2.
    """

    try:
        test1 = pap.list_overlap_comp(listOdd, listMixed)
        test2 = pap.list_overlap_comp(listEven, listMixed)
        test3 = pap.list_overlap_comp(listOdd, listEven)
        test4 = pap.list_overlap_comp(listOdd, emptyList)
        test5 = pap.list_overlap_comp(listOdd, listdups2)
        if len(test1) != 1:
            print("FAILED: list_overlap_comp(listOdd,listMixed) returned: " + str(test1))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listMixed) with: " + str(test1))
        if len(test2) != 2:
            print("FAILED: list_overlap_comp(listEven,listMixed) returned: " + str(test2))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listEven,listMixed) with: " + str(test2))
        if len(test3) != 0:
            print("FAILED: list_overlap_comp(listOdd,listEven) returned: " + str(test3))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listEven) with: " + str(test3))
        if len(test4) != 0:
            print("FAILED: list_overlap_comp(listOdd,emptyList) returned: " + str(test4))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,emptyList) with: " + str(test4))
        if len(test5) != 2:
            print("FAILED: list_overlap_comp(listOdd,listdups2) returned: " + str(test5))
            f += 0.2
        else:
            print("passed: list_overlap_comp(listOdd,listdups2) with: " + str(test5))
    except Exception as ex:
        print(ex)
        print("FAILED: list_overlap_comp threw an exception.")
        f += 1

# More practice with Dictionaries, Files, and Text!
# Implement the following functions:

    test_files = []
    test_files.append("rj_prologue.txt")
    test_files.append("permutation.txt")
    test_files.append("UncannyValley.txt")

# def longest_sentence(text_file_name):
    """ Read from the text file, split the data into sentences,
        and return the longest sentence in the file.
    """
    answers = []
    answers.append("The fearful passage of their death-mark'd love,\nAnd the continuance of their parents' rage,\nWhich, but their children's end, nought could remove,\nIs now the two hours' traffic of our stage")
    answers.append(" Objects out of sight didn't \"vanish\" entirely, if they influenced the ambient light, but Paul knew that the calculations would rarely be pursued beyond the crudest first-order approximations: Bosch's Garden of Earthly Delights reduced to an average reflectance value, a single grey rectangle — because once his back was turned, any more detail would have been wasted")
    answers.append("Later, in a room of his own, his bed had come with hollow metal posts whose plastic caps were easily removed, allowing him to toss in chewed pencil stubs, pins that had held newly bought school shirts elaborately folded around cardboard packaging, tacks that he'd bent out of shape with misaligned hammer blows while trying to form pictures in zinc on lumps of firewood, pieces of gravel that had made their way into his shoes, dried snot scraped from his handkerchief, and tiny, balled-up scraps of paper, each bearing a four- or five-word account of whatever seemed important at the time, building up a record of his life like a core sample slicing through geological strata, a find for future archaeologists far more exciting than any diary")
    try:
        for i in range(len(test_files)):
            output = pap.longest_sentence(test_files[i])
            if  output.strip().lower().rstrip('.!?;') != answers[i].strip().lower().rstrip('.!?;'):
                print("FAILED: longest_sentence(" + test_files[i] + ") returned: \n" + str(output) + "\n instead of: \n" + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: longest_sentence(" + test_files[i] + ") with: \n" + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: longest_sentence threw an exception.")
        f += 1


# def longest_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return the longest word in the file.
    """
    answers = []
    answers.append("misadventured")
    answers.append("soon-to-be-forgotten")
    answers.append("jurisprudentially")

    try:
        for i in range(len(test_files)):
            output = pap.longest_word(test_files[i])
            if len(output) != len(answers[i]):
                print("FAILED: longest_word(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: longest_word(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: longest_word threw an exception.")
        f += 1

# def num_unique_words(text_file_name):
    """ Read from the text file, split the data into words,
        and return the number of unique words in the file.
        HINT: Use a set!
    """
    answers = []
    answers.append(80)
    answers.append(1540)
    answers.append(2962)

    try:
        for i in range(len(test_files)):
            output = pap.num_unique_words(test_files[i])
            if math.fabs(output - answers[i]) > max(2,answers[i]/100):
                print("FAILED: num_unique_words(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: num_unique_words(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: num_unique_words threw an exception.")
        f += 1

# def most_frequent_word(text_file_name):
    """ Read from the text file, split the data into words,
        and return a tuple with the most frequently occuring word
        in the file and the count of the number of times it apapared.
    """
    answers = []
    answers.append(('their',6))
    answers.append(('the', 266))
    answers.append(('the', 720))
    try:
        for i in range(len(test_files)):
            output = pap.most_frequent_word(test_files[i])
            if  output[0].lower() != answers[i][0].lower() and math.fabs(output[1] - answers[i][1]) > max(2,answers[i][1]/100) :
                print("FAILED: most_frequent_word(" + test_files[i] + ") returned: " + str(output) + " instead of: " + str(answers[i]))
                f += 0.333333333
            else:
                print("passed: most_frequent_word(" + test_files[i] + ") with: " + str(output))
    except Exception as ex:
        print(ex)
        print("FAILED: most_frequent_word threw an exception.")
        f += 1

except Exception as ex:
    print(ex)
    print("FAILED: PythonAdvPractice2019.py file does not execute at all, or this file was not implemented.")
    f = 2

print("\n")
print("SUMMARY:")
print("Passed " + str(round(num_exer-f,2)) + " out of " + str(num_exer) + " exercises.")
