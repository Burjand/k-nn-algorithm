from Validation_method import Validation_method
from Classifier import Classifier
from Performance_measure import Performance_measure
import config
import copy

if __name__ == "__main__":

    # Dataset
    dataset_file_path = config.data_path
    file = "08_uci_haberman_58_2_att_27_patt.csv"
    file_delimiter = ','


    # Validation method
    test_patterns_positions = (11, 12, 13, 14, 17, 18, 27)
    validation_method = Validation_method(dataset_file_path + file, file_delimiter)
    sets = validation_method.fixed_partition(test_patterns_positions)
    training_set = sets[0]
    test_set = sets[1]


    # Algorithm
    classifier_1NN = Classifier(copy.deepcopy(training_set), copy.deepcopy(test_set))
    classified_test_set = classifier_1NN.algorithm_1NN("city_block")

    # Performance measure
    possitive_class = 2.0
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