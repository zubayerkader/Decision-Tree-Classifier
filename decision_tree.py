# from numpy.core.fromnumeric import partition, shape
import pandas as pd
import math

class Node:
    def __init__(self):

        self.parent = None
        self.children = []
        self.edge = None
        self.attribute = ""
        self.label = ""
        self.entropy = None

def grow(dataset, attributes, node):
    # print(attributes)
    # if node.parent is not None:
    #     print(node.parent.attribute)
    termination_condition = terminationVariables(dataset, attributes)
    if (termination_condition['majority_label_percent'] >= 0.9 or len(attributes) == 0):
        node.label = termination_condition['majority_label']
        # print ('returning', node)
        return node
    else:
        # calculate entropy of root
        if node.entropy == None:
            total_records = dataset.shape[0]
            income_gt_50k = dataset.where(dataset['income'] == '>50K').dropna().shape[0]      # df.shape -> (rows, cols)
            income_lte_50k = dataset.where(dataset['income'] == '<=50K').dropna().shape[0]
            # print (income_gt_50k, income_lte_50k, total_records, dataset)

            if income_gt_50k == 0:
                x = 0
                logx = 0
            else:
                x = income_gt_50k/total_records
                logx = math.log2(x)

            if income_lte_50k == 0:
                y = 0
                logy = 0
            else:
                y = income_lte_50k/total_records
                logy = math.log2(y)

            node.entropy = (-1)*((x*logx) + (y*logy))

        split_variables = getSplitVariables(dataset, attributes, node.entropy)
        node.attribute = split_variables['attribute']
        attributes.remove(split_variables['attribute'])
        if split_variables['type'] == 'categorical':
            partitions = split_variables['E_cat']
            for cat, entropy in partitions.items():
                child_node = Node()
                child_node.parent = node
                child_node.entropy = entropy
                child_node.edge = cat
                node.children.append(child_node)
                # print(split_variables['attribute'])
                
                # print(new_attributes)
                partitioned_data = dataset.where( dataset[split_variables['attribute']] == cat).dropna()

                if(partitioned_data.shape[0] == 0):
                    node.label = termination_condition['majority_label']
                    # print ('returning when dataset is empty', node)
                    return node
                grow(partitioned_data, attributes, child_node)
        else:
            partitions = split_variables['E_bin']
            # print(partitions)
            for bin, entropy in partitions.items():
                child_node = Node()
                child_node.parent = node
                child_node.entropy = entropy
                child_node.edge = bin
                node.children.append(child_node)
                # print(split_variables['attribute'])
                # print(attributes)
                bin_range = bin.split('_')
                # print(bin_range, type(bin_range[0]), bin_range[1])
                gt = float(bin_range[0])
                lte = float(bin_range[1])
                partitioned_data = dataset.where(  ( (dataset[split_variables['attribute']] > gt) & (dataset[split_variables['attribute']] <= lte) )  ).dropna()
                # print(partitioned_data, bin)
                if(partitioned_data.shape[0] == 0):
                    node.label = termination_condition['majority_label']
                    # print ('returning when dataset is empty', node)
                    return node
                grow(partitioned_data, attributes, child_node)

def getSplitVariables(dataset, attributes, parent_entropy):
    # find split attribute using information gain

    split_data = {}
    total_records = dataset.shape[0]
    num_cols = list(set(dataset._get_numeric_data().columns).intersection(attributes))
    cat_cols = list(set(attributes) - set(num_cols))

    # calculate entropy of categorical attributes
    for attr in cat_cols:
        categories = dataset[attr].unique()
        attr_data = {}
        E_attr = 0
        cat_data = {}

        # calculate entropy of each category
        for cat in categories:
            income_gt_50k_cat = dataset.where(  ( (dataset[attr] == cat) & (dataset['income'] == '>50K') )  ).dropna().shape[0]
            income_lte_50k_cat = dataset.where(  ( (dataset[attr] == cat) & (dataset['income'] == '<=50K') )  ).dropna().shape[0]
            total_records_in_cat = income_gt_50k_cat + income_lte_50k_cat

            # print (income_gt_50k_cat, income_lte_50k_cat, total_records_in_cat, dataset)

            if income_gt_50k_cat == 0:
                x = 0
                logx = 0
            else:
                x = income_gt_50k_cat/total_records_in_cat
                logx = math.log2(x)

            if income_lte_50k_cat == 0:
                y = 0
                logy = 0
            else:
                y = income_lte_50k_cat/total_records_in_cat
                logy = math.log2(y)

            E_cat = (-1)*((x*logx) + (y*logy))
            E_attr += (total_records_in_cat/total_records)*E_cat
            cat_data[cat] = E_cat
        
        attr_data['attribute'] = attr
        attr_data['info_gain'] = parent_entropy - E_attr
        attr_data['type'] = 'categorical'
        attr_data['E_cat'] = cat_data
        attr_data['E_bin'] = None
        split_data[attr] = attr_data

    # calculate entropy of numerical attributes
    for attr in num_cols:
        bins = pd.qcut(dataset[attr], q=4, duplicates='drop', retbins=True)[1]
        attr_data = {}
        E_attr = 0 
        num_data = {}

        for i in range(len(bins)-1):
            income_gt_50k_bin = dataset.where(  ( (dataset[attr] > bins[i]) & (dataset[attr] <= bins[i+1]) & (dataset['income'] == '>50K') )  ).dropna().shape[0]
            income_lte_50k_bin = dataset.where(  ( (dataset[attr] > bins[i]) & (dataset[attr] <= bins[i+1]) & (dataset['income'] == '<=50K') )  ).dropna().shape[0]
            total_records_in_bin = income_gt_50k_bin + income_lte_50k_bin

            if income_gt_50k_bin == 0:
                x = 0
                logx = 0
            else:
                x = income_gt_50k_bin/total_records_in_bin
                logx = math.log2(x)

            if income_lte_50k_bin == 0:
                y = 0
                logy = 0
            else:
                y = income_lte_50k_bin/total_records_in_bin
                logy = math.log2(y)

            E_bin = (-1)*((x*logx) + (y*logy))
            E_attr += (total_records_in_bin/total_records)*E_bin
            num_data[str(bins[i])+'_'+str(bins[i+1])] = E_bin
        
        attr_data['attribute'] = attr
        attr_data['info_gain'] = parent_entropy - E_attr
        attr_data['type'] = 'numerical'
        attr_data['E_cat'] = None
        attr_data['E_bin'] = num_data
        split_data[attr] = attr_data

    # find attribute with maximum information gain
    max_info_attr = split_data[next(iter(split_data))]
    for attr, attr_data in split_data.items():
        if (max_info_attr['info_gain'] >= attr_data['info_gain']):
            max_info_attr = attr_data

    return max_info_attr

def terminationVariables(dataset, attributes):
    # returns majority class label and percentage of label; also return number of attributes
    # cols = attributes
    # cols.append('income')
    # data = dataset[cols]
    data = dataset

    income_gt_50k = data.where(data['income'] == '>50K').dropna().shape[0]     # df.shape -> (rows, cols)
    income_lte_50k = data.where(data['income'] == '<=50K').dropna().shape[0]
    total_records = data.shape[0]
    
    majority_label = ''
    majority_label_percent = 0
    if income_gt_50k/total_records > income_lte_50k/total_records:
        majority_label = '>50K'
        majority_label_percent = income_gt_50k/total_records
    else:
        majority_label = '<=50K'
        majority_label_percent = income_lte_50k/total_records

    ret = {
        'majority_label': majority_label,
        'majority_label_percent': majority_label_percent,
    }

    return ret

predictions = None

def goToLeaf(row, attributes, node):
    global predictions
    if len(node.children) == 0:
        row['income'] = node.label
        predictions = predictions.append(row)
        # print(predictions)
        return
    else:
        attr = node.attribute
        attr_value = row[attr]
        attributes.remove(attr)
        for child in node.children:
            if type(attr_value) is str:
                if child.edge == attr_value:
                    goToLeaf(row, attributes, child)
                    return
            else:
                bin_range = child.edge.split('_')
                gt = float(bin_range[0])
                lte = float(bin_range[1])
                if attr_value > gt and attr_value <= lte:
                    goToLeaf(row, attributes, child)
                    return

def test (dataset, node):
    global predictions
    predictions = pd.DataFrame(columns=list(dataset.columns))

    for index, row in dataset.iterrows():
        goToLeaf(row, list(dataset.columns), node)

    predictions.reset_index()
    merged = pd.merge(predictions, dataset, left_index=True, right_index=True)
    if ('income_x' in merged):
        correct_prediction = merged[merged.income_x == merged.income_y]
        correct_prediction_count = correct_prediction.shape[0]
        accuracy = correct_prediction_count/dataset.shape[0]
    else :
        accuracy = 'Unknown because all results were predicted'

    return (predictions, accuracy)

def prune(dataset, node):
    return node
    
def crossValidation(dataset):

    folds = 5  
    cum_accuracy = 0

    for i in range(folds):
        test_data = dataset.sample(frac=0.1)
        training_data = dataset.drop(test_data.index, axis=0)
        attributes = list(training_data.columns)
        attributes.remove('income')

        root = Node()
        grow(training_data, attributes, root)
        # pruned_tree = prune(training_data, root)          ###########################################
        predictions, accuracy = test(test_data, root)
        cum_accuracy += accuracy

    avg_accuracy = cum_accuracy/folds
    return avg_accuracy


def main():
    training_data = pd.read_csv('./data/adult.data.csv')
    training_data.replace(' ?', value=None, inplace=True) # interpolates missing values using default method='pad'
    # print(training_data)

    test_data = pd.read_csv('./data/adult.test.csv')
    test_data.replace(' ?', value=None, inplace=True) 
    # print(test_data)

    attributes = list(training_data.columns)
    attributes.remove('income')

    avg_accuracy = crossValidation(training_data)
    print('5-fold-cross-validation average accuracy: ', avg_accuracy)

    root = Node()
    grow(training_data, attributes, root)
    # pruned_tree = prune(training_data, root)
    predictions, accuracy = test(test_data, root)
    predictions.to_csv('./predictions.csv')
    print('accuracy of predictions with test data: ', accuracy)
    

if __name__ == "__main__":
    main()