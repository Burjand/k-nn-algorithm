from Validation_method import Validation_method
from Classifier import KNN_classifier
from Performance_measure import Performance_measure
import config
import copy
import csv

if __name__ == "__main__":

    # Dataset
    dataset_file_path = config.data_path
    file = "5_Nutt.csv"
    file_delimiter = ','

    csv_file = open(dataset_file_path + file, newline='')
    dataset = list(csv.reader(csv_file, delimiter = file_delimiter))

    count_patterns = len(dataset)

    # Apply Leave One Out and KNN

    classified_test_set = []

    for i in range(count_patterns):
    
        # Validation method
        validation_method = Validation_method(dataset_file_path + file, file_delimiter)
        training_set, test_set = validation_method.leave_one_out(i)


        # Algorithm
        classifier_KNN = KNN_classifier(copy.deepcopy(training_set), copy.deepcopy(test_set))
        classified_test_set.append(classifier_KNN.algorithm_KNN("euclidean", 27))


    for i in range(count_patterns):

        classified_test_set[i] = classified_test_set[i][0]        

    # Performance measure
    possitive_class = 0.0
    performance_measure = Performance_measure(classified_test_set, possitive_class)

    print(performance_measure.confusion_matrix)
    print("Accuracy = " + str(performance_measure.accuracy))
    print("Error rate = " + str(performance_measure.error_rate))
    print("Sensitivity = " + str(performance_measure.sensitivity))
    print("Specificity = " + str(performance_measure.specificity))
    print("Precision = " + str(performance_measure.precision))
    print("Balanced Accuracy = " + str(performance_measure.balanced_accuracy))
    print("F1 score = " + str(performance_measure.f1_score))
    print("MCC = " + str(performance_measure.mcc))