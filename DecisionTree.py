import operator
import pandas as pd
import math
import numpy as np


def same_classification(examples):
    """
    Checks to see if all the examples passed in are of the same class. If so, we can conclude an answer.
    :param examples: Holds the rows from the csv file. Examples used to build our tree.
    :return:
    """

    classification = examples[0][1]
    total_examples = len(examples[0])

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


def decision_tree_learning(examples, examples_attributes, attribute_names, parent_examples):
    print('PRINTING EXAMPLES IN DTL: ', examples)
    print('printing attr: ', examples_attributes)

    if len(examples) == 0:
        print('No more examples left.')
        return plurality_value(parent_examples)

    elif same_classification(examples):
        print('Same classification found for all examples.')
        classification = examples[0][1]
        return classification

    elif len(examples_attributes) == 0:
        print('There are no more attributes to evaluate.')
        return plurality_value(examples)
    else:
        A = []
        total_attributes = len(examples_attributes[0])
        total_examples = len(examples)
        values_map = {}


        # Traverse through all the attributes for every example.
        for attr in range(total_attributes):
            current_values = []
            attribute_examples = []
            for example in range(total_examples):
                attribute_examples.append(examples_attributes[example][attr])
                if examples_attributes[example][attr] not in current_values:
                    current_values.append(examples_attributes[example][attr])
            # When we reach here, we have traversed through all the examples
            # and have the results for the current attribute stored in the list attribute_examples

            # Now we add to A so we can send off the attribute and the result for each example to find the Info. Gain.
            values_map[attribute_names[attr]] = current_values
            A.append((attr, attribute_names[attr], importance(attribute_examples, examples)))

        largest_gain = ('Place_Holder', 'Place_Holder', -math.inf)
        print('A: ', A)
        # When I reach here, I want the ARG MAX of the attribute
        for attr in A:
            if attr[2] > largest_gain[2]:
                largest_gain = attr
        print('Winner: ', largest_gain)
        print('Values: ', values_map[largest_gain[1]])
        print('Attribute is at index: ', largest_gain[0])
        tree = [largest_gain[0]]
        idx = largest_gain[0]
        values = values_map[largest_gain[1]]
        for value in values:
            exs = []
            i = 1
            for example in examples_attributes:
                if example[idx] == value:
                    exs.append((i, example))
                i += 1

            print('Value: ', value, ' For The exs: ')
            print(exs)
            print('THE EXAMPLES:')
            print(examples)
            new_examples = []
            new_examp_attr = []

            for i in range(len(exs)):
                new_examples.append((exs[i][0], examples[exs[i][0] - 1][1]))
                new_examp_attr.append(exs[i][1])
            
            subtree = decision_tree_learning(new_examples, new_examp_attr, attribute_names, examples)


        # TODO: Recursive call
        # TODO: Add branch to tree.


        return tree


def main():

    file_name = 'TreeData.csv'

    tree_file = pd.read_csv(file_name, header=0)
    column_names = list(tree_file.columns.values)
    print(column_names)

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

    print('Printing Examples:')
    print(examples)
    print('Printing Attributes:')
    print(attributes)

    final_tree = decision_tree_learning(examples, attributes, column_names[1:], [])


if __name__ == '__main__':
    main()
