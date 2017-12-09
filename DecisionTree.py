"""
Name: Ashour
NetID: Adankh2
Homework 13

The following program is building a tree classifier. More description on the algorithm can be read from the book
Figure 18.3.5

"""

import pandas as pd
import math


def same_classification(examples):
    """
    Checks to see if all the examples passed in are of the same class. If so, we can conclude an answer.
    :param examples: Holds the rows from the csv file. Examples used to build our tree.
    :return:
    """

    classification = examples[0][1]
    total_examples = len(examples)

    for i in range(total_examples):
        if examples[i][1] != classification:
            return False
    return True


def plurality_value(examples):
    """
    Grabs the classification that wins amongst the examples provided.
    Aka Selects the most common output value among a set of examples. Ties are random.
    :param examples: Holds the examples to be evaluated.
    :return: Returns the classification that won.
    """

    classification_counter = {}

    for example in examples:
        if example[1] not in classification_counter.keys():
            classification_counter[example[1]] = 1
        else:
            classification_counter[example[1]] += 1

    winner = max(classification_counter, key=classification_counter.get)
    return winner


def B(q):
    """
    The entropy of a Boolean (B) random variable.

    :param q: Let q be the number of "yes" over the total number for the specific attribute type
    :return: returns the entropy of a Boolean random variable that is true with probability q
    """
    return -1 * (q * math.log2(q) + (1 - q) * math.log2(1 - q))


def importance(attribute_examples, classified_as):
    """
    This will calculate the information gain for the attribute.

    :param attribute_examples: This is a list of the a single attribute and all the types of that attribute provided
    by all the examples.
    So if we are evaluating Rain, it will give a list of len(examples) and list the answers for each example.
    :param classified_as: Holds the classification chosen for each example.
    :return: The information gain
    """

    attribute_types = {}

    total_ex = len(classified_as)
    for i, j in zip(attribute_examples, range(total_ex)):

        if classified_as[j][1] == 'yes':
            if i not in attribute_types.keys():
                attribute_types[i] = [1, 1]
            else:
                attribute_types[i][0] += 1
                attribute_types[i][1] += 1
        else:
            if i not in attribute_types.keys():
                attribute_types[i] = [0, 1]
            else:
                attribute_types[i][1] += 1

    entropy = 0
    for attr_type in attribute_types.keys():

        # Let p_x be the probability of the specific attribute type over the total number of examples
        p_x = float(attribute_types[attr_type][1] / total_ex)

        # Let q be the number of "yes" over the total number for the specific attribute type
        q = float(attribute_types[attr_type][0]) / float(attribute_types[attr_type][1])

        # Check to see if q and q - 1 is not 0. Otherwise we cannot take log of 0
        if q != 0 and q - 1 != 0:
            entropy += p_x * B(q)

    information_gain = 1 - entropy

    return information_gain


values_map = {}


def decision_tree_learning(examples, examples_attributes, attribute_names, parent_examples, is_first):
    """
    The learning algorithm shown in the book AIMA 3rd edition, Figure 18.5.

    Note: Variable names here are identical to what is written in the book to stay consistent.

    :param examples: Holds all the examples in a tuple of (example #, classed_as)
    :param examples_attributes: Holds the "answers" for each examples for all the attributes
    :param attribute_names: Holds the attribute names
    :param parent_examples: Holds where examples came from (initially empty list)
    :param is_first: Checks to see if it's the first call (just for some starting population of maps)
    :return: Returns the tree with all sub-trees/branches
    """

    global values_map

    # No more examples
    if len(examples) == 0:
        return plurality_value(parent_examples)

    # Same classification found for all examples, returns random class
    elif same_classification(examples):
        classification = examples[0][1]
        return classification

    # No more attributes to evaluate, so get the "best value" from the examples.
    elif len(examples_attributes) == 0:
        return plurality_value(examples)
    else:
        # A will hold all the attributes and their Information Gain at the current level of the tree.
        A = []

        # Self explanatory variables
        total_attributes = len(examples_attributes[0])
        total_examples = len(examples)

        # Traverse through all the attributes for every example.
        for attr in range(total_attributes):
            current_values = []
            attribute_examples = []

            # Traverse through all the the examples and store all the results for the current attribute
            for example in range(total_examples):
                attribute_examples.append(examples_attributes[example][attr])
                if examples_attributes[example][attr] not in current_values:
                    current_values.append(examples_attributes[example][attr])

            # When we reach here, we have traversed through all the examples
            # and have the results for the current attribute stored in the list attribute_example
            if is_first:
                values_map[attribute_names[attr]] = current_values
            # Now we add to A so we can send off the attribute and the result for each example to find the Info. Gain.
            A.append((attr, attribute_names[attr], importance(attribute_examples, examples)))
        largest_gain = ('Place_Holder', 'Place_Holder', -math.inf)

        # When I reach here, I want the ARG MAX of the attribute
        for attr in A:
            if attr[2] > largest_gain[2]:
                largest_gain = attr

        # Tree now holds the attribute that is "most important" aka highest information gain.
        tree = [(largest_gain[0], largest_gain[1])]
        idx = largest_gain[0]
        values = values_map[largest_gain[1]]

        for value in values:

            exs = []
            i = 1
            for example in examples_attributes:
                if example[idx] == value:
                    exs.append((i, example))
                i += 1

            new_examples = []
            new_examp_attr = []

            for i in range(len(exs)):
                new_examples.append((exs[i][0], examples[exs[i][0] - 1][1]))
                new_examp_attr.append(exs[i][1])
            subtree = decision_tree_learning(new_examples, new_examp_attr, attribute_names, examples, False)
            tree.append(((largest_gain[1] + ', Type: ' + value), subtree))

        return tree


def print_tree(final_tree):
    """
    Self explanatory. This prints the tree. Essentially what it does is it prints the leaf nodes first.
    :param final_tree: Holds the tree that is still left to be printed
    :return:
    """
    print_later = []
    for i in final_tree:
        if 'Attribute:' in i[1]:
            print(i[0], i[1])
        elif 'yes' in i[1] or 'no' in i[1]:
            print(i[0] + ', Decision:', i[1])
        else:
            print_later.append((i[0], i[1]))

    for i in print_later:
        print(i[0], ' Decision: -->')
        print_tree(i[1])


def main():

    file_name = 'TreeData.csv'

    tree_file = pd.read_csv(file_name, header=0)
    column_names = list(tree_file.columns.values)
    # print(column_names)

    # The list examples holds the example and the classification of that example
    examples_as_matrix = tree_file[column_names[0]].as_matrix()
    examples = []

    # attributes holds the attributes associated with each example
    attr_as_matrix = tree_file.drop([column_names[0], column_names[-1]], axis=1).as_matrix()
    attributes = []

    # classification holds the classifications (yes or no) for each example
    classes_as_matrix = tree_file[column_names[-1]].as_matrix()

    for i, j, k in zip(examples_as_matrix, attr_as_matrix, classes_as_matrix):
        examples.append((i, k))
        attributes.append(list(j))

    # print('Printing Examples:')
    # print(examples)
    #
    # print('Printing Attributes:')
    # print(attributes)

    final_tree = decision_tree_learning(examples, attributes, column_names[1:], [], True)

    # print('THE FINAL TREE:')
    # print(final_tree)
    print_tree(final_tree)

    print('\nThe arrows in the text printed above show the new attribute to take down the '
          'tree if that attribute type has been chosen.')
    print('The following program uses the example provided in the book to help illustrate my algorithm.')


if __name__ == '__main__':
    main()
