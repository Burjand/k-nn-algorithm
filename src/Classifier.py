from Calculate_distance import Calculate_distance

class KNN_classifier():

    def __init__(self, training_set, test_set):

        self.training_set = training_set
        self.test_set = test_set

        # Convert string to numbers
        self.training_set = self.cast_to_numbers(self.training_set)
        self.test_set = self.cast_to_numbers(self.test_set)


    def cast_to_numbers(self, list_of_lists_with_strings):

        list_of_lists_with_numbers = []

        for item in list_of_lists_with_strings:

            new_item = []
            
            for element in item:

                new_item.append(float(element))

            list_of_lists_with_numbers.append(new_item)

        return list_of_lists_with_numbers

    
    def algorithm_KNN(self, distance_type, k):

        distance_calculator = Calculate_distance()
        classified_test_set = []

        for test_pattern in self.test_set:
            distances = []

            # Calculate the distance between test pattern and each training pattern
            for training_pattern in self.training_set:
                distance = distance_calculator.calculate_distance(distance_type, training_pattern[:-1], test_pattern[:-1])
                distances.append((distance, training_pattern[-1])) #Append the calculated distance and the class of the training pattern

            # Order the distances
            distances.sort(key=lambda x: x[0])

            classification = self.vote(distances, k)

            # Append the label to the test pattern and add it to the classified set
            classified_test_set.append(test_pattern + [classification])

        return classified_test_set
    

    def vote(self, distances, k):

        vote_count = {}

        for i in range(k):

            if distances[i][1] not in vote_count.keys():

                vote_count[distances[i][1]] = 1

            else:

                vote_count[distances[i][1]] += 1

        return max(vote_count, key=vote_count.get)




        




