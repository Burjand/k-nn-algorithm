"""
Class for the validation methods

Fixed partition (This method will be improved in the future for better flexibility in parameters intake)

Description of parameters:
dataset: the dataset for which the validation method will be used
indexes: A list with the indexes to make the partition, it takes 4: a for the training set of class 1, (a,b] for the test set
        of class 1, (b,c] for the training set of class 2 and (c,d] for the test set of class 2.
"""

import csv
import random

class Validation_method():

    def __init__(self, dataset_file_path, file_delimiter):

        # Import dataset from file
        self.csv_file = open(dataset_file_path, newline='')
        self.dataset = list(csv.reader(self.csv_file, delimiter = file_delimiter))


    def fixed_partition(self, test_patterns_positions: tuple):

        training_set = []
        test_set = []

        for i in range(len(self.dataset)):

            if (i+1) in test_patterns_positions:

                test_set.append(self.dataset[i])

            else:

                training_set.append(self.dataset[i])

        return [training_set, test_set]
    


    def hold_out(self, percentage_training_subset, label_position=True):

        training_set = []
        test_set = []

        # Separate by classes
        label_index = (len(self.dataset[0]) - 1) if label_position else 0

        patterns_by_class = {}

        for pattern in self.dataset:

            if pattern[label_index] not in list(patterns_by_class.keys()):
                
                patterns_by_class[pattern[label_index]] = [[x for i,x in enumerate(pattern) if i!=label_index]]

            else:

                patterns_by_class[pattern[label_index]].append([x for i,x in enumerate(pattern) if i!=label_index])
        
        
        # Shuffle patterns in classes
        patterns_by_class_shuffled = {}

        for label in patterns_by_class.keys():

            patterns_in_class = len(patterns_by_class[label])
            indexes_of_patterns = list(range(patterns_in_class))

            random.shuffle(indexes_of_patterns)

            for index in indexes_of_patterns:

                if label not in list(patterns_by_class_shuffled.keys()):

                    patterns_by_class_shuffled[label] = [patterns_by_class[label][index]]

                else:
                    
                    patterns_by_class_shuffled[label].append(patterns_by_class[label][index]) 


        # Create training and testing subsets
        for label in list(patterns_by_class_shuffled.keys()):

            amount_of_patterns = len(patterns_by_class_shuffled[label])
            cardinality_training = round((percentage_training_subset/100) * amount_of_patterns)

            for i in range(len(patterns_by_class_shuffled[label])):

                if (i < cardinality_training):
                    
                    #print(type(patterns_by_class_shuffled[label][i]))
                    #print(patterns_by_class_shuffled[label][i])
                    patterns_by_class_shuffled[label][i].append(label)
                    training_set.append(list(map(float,patterns_by_class_shuffled[label][i])))

                else:

                    patterns_by_class_shuffled[label][i].append(label)
                    test_set.append(list(map(float,patterns_by_class_shuffled[label][i])))

        return [training_set, test_set]


        
    def leave_one_out(self, the_one):

        training_set = []
        test_set = []

        test_set = [self.dataset.pop(the_one)]
        training_set = self.dataset

        return [training_set, test_set]

                



            





