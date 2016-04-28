import sys
import csv
import copy
import random
import operator
import numpy as np
from numpy import unravel_index
from sklearn.cross_validation import KFold
import scipy as sp


def read_data_file(file_name):
    data_file = []
    structure_file = []
    target_data = []
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data_file.append(list(row))
        random.shuffle(data_file)
        for row in data_file:
            target_data.append(row[-1])
            del row[-1]

    with open(file_name.replace(".txt", "") + '-structure.txt','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            structure_file.append(list(row))

    for i in range(len(data_file)):
        for j in range(len(data_file[i])):
            if data_file[i][j] == "?":
              data_file[i][j] = random.choice(structure_file[j + 1])

    return data_file, target_data, structure_file
    
def confident_interval_calculation(list_of_errors, confidence=0.95):
    a = 1.0*np.array(list_of_errors)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def error_calculation(validation_target, validation_output):
    count = 0
    for output_index in range(len(validation_output)):
        max_index = unravel_index(validation_output[output_index].argmax(), validation_output[output_index].shape)[0]
        if validation_target[output_index][max_index] == 1:
            count = count + 1

    return float(count)/len(validation_output)

def normalize_data(data_file, target_data, structure_file):
    normalized_target_data = []
    for i in range(len(data_file)):
        for j in range(len(data_file[i])):
            data_file[i][j] = float(structure_file[j + 1].index(data_file[i][j]) + 1)/len(structure_file[j+1])
    
    for i in range(len(target_data)):
        individual_target_data = [0]*len(structure_file[-1])
        individual_target_data[structure_file[-1].index(target_data[i])] = 1 
        normalized_target_data.append(individual_target_data)
    return data_file, normalized_target_data

def calculate_distance(train_data, test_data):
    distance = 0
    for x in range(len(train_data)):
        distance += pow((train_data[x] - test_data[x]), 2)
    return distance

def most_common(lst):
    return max(lst, key=lst.count)

if __name__ == "__main__":  
    file_name = sys.argv[1]
    k_neighbor = int(sys.argv[2])
    
    data_file, target_data, structure_file = read_data_file(file_name)
    normalized_data = copy.deepcopy(data_file)
    normalized_data, normalized_target_data = normalize_data(copy.deepcopy(data_file), copy.deepcopy(target_data), copy.deepcopy(structure_file))

    data_input = np.array(normalized_data)
    kf = KFold(len(data_input), n_folds=10)

    all_accuracy = []
    for train, test in kf:
        err_count = 0
        for x in range(len(data_input[test])):
            closest_neighbor = {}
            for y in range(len(data_input[train])): 
                distance = calculate_distance(data_input[train][y], data_input[test][x])
                if len(closest_neighbor) < k_neighbor:
                    closest_neighbor[y] = distance
                else:
                    if distance < max(closest_neighbor.iteritems(), key=operator.itemgetter(1))[1]:
                        del closest_neighbor[max(closest_neighbor.iteritems(), key=operator.itemgetter(1))[0]]
                        closest_neighbor[y] = distance
            common_attr = []
            
            # print "Target: {0}".format(target_data[test[x]])
            for z in closest_neighbor.keys():
                # print target_data[z]
                common_attr.append(target_data[z])
            # print "Common: {0}".format(most_common(common))
            if(target_data[test[x]] != most_common(common_attr)):
                err_count += 1
        all_accuracy.append(1-float(err_count)/len(test))

                
        
        # print 
        # print target_data[]
        # print data_input[train]
        # print "\n"
        # print data_input[train][0]
        # print data_input[test]
        # print data_input[test][0]
        # break
    # print all_accuracy
    print np.mean(all_accuracy)
    print np.std(all_accuracy)
    print confident_interval_calculation(all_accuracy)





















