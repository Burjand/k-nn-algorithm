from Calculate_distance import Calculate_distance

class KNN_classifier():

    def __init__(self, training_and_test_sets):

        self.training_and_test_sets = training_and_test_sets

        # Convert string to numbers
        self.training_and_test_sets = self.cast_to_numbers(self.training_and_test_sets)


    def cast_to_numbers(self, list_of_lists_with_strings):

        list_of_lists_with_numbers = []

        for item in list_of_lists_with_strings:

            new_item = []
            
            for element in item:

                new_item.append(float(element))

            list_of_lists_with_numbers.append(new_item)

        return list_of_lists_with_numbers

    
    def algorithm_KNN(self, distance_type):

        distance_calculator = Calculate_distance()
        classified_test_set = []

        for test_pattern in self.test_set:
            distances = []

            # Calculate the distance between test pattern and each training pattern
            for training_pattern in self.training_set:
                distance = distance_calculator.calculate_distance(distance_type, training_pattern[:-1], test_pattern[:-1])
                distances.append((distance, training_pattern[-1]))

            # Find the training pattern with the minimum distance
            shortest_distance, temp_label = min(distances, key=lambda x: x[0])

            # Append the label to the test pattern and add it to the classified set
            classified_test_set.append(test_pattern + [temp_label])

        return classified_test_set




        




