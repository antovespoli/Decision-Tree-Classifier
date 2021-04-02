import numpy as np
import matplotlib.pyplot as plt

# This is just to count number of nodes, we will remove it before submission
global node_count
node_count = 0

class Node:
	def __init__(self, value, attribute=None, left=None, right=None):
		global node_count # remove
		node_count = node_count + 1 # remove
		self.value = value
		self.attribute = attribute
		self.left = left
		self.right = right

"""
	entropy(dataset)

Retrieve the entropy of a given dataset. The entropy is the metric measuring 
information. It will be the one used to decide splits when learning our decision tree. 
"""
def entropy(dataset):
	last_column = dataset.T[7] # We need to look at the counts in the last column
	N = len(last_column)
	values, counts = np.unique(last_column, return_counts=True)
	entropy = 0
	# entropy has to be calculated based on the last column
	for i in range(len(values)):
		probability_of_value = counts[i]/N
		entropy = entropy - (probability_of_value * np.log2(probability_of_value))
	return entropy

"""
	find_split(dataset)

Find the best split, hence the one which result in the highest information gain.
"""
def find_split(dataset):
	split_real = 0
	G_real = -1
	attrib_real = 0
	for attr in range(len(dataset[0]) - 1):
		column_to_sort = dataset.T[attr]
		sort_indices = column_to_sort.argsort()
		DS = dataset[sort_indices]
		H = entropy(DS)
		split = 0
		gain = 0
		for i in range(len(DS)-1):
			if DS[i + 1][attr] != DS[i][attr]:
				remainder = (i + 1) / len(DS) * entropy(DS[:i]) + (len(DS) - i - 1) / len(DS) * entropy(DS[i + 1:])
				if H - remainder > gain:
					gain = H - remainder
					split = float((DS[i][attr] + DS[i + 1][attr]) / 2)
		if gain > G_real:
			G_real = gain
			split_real = split
			attrib_real = attr
	return attrib_real, split_real

"""
	decision_tree_learning(dataset, depth=0, depth_limit=0)

Learn a decision tree thanks to a given dataset. 
"""
def decision_tree_learning(dataset, depth=0, depth_limit=0):
    check_labels = np.unique(dataset.T[7]) # 7th column is label so get unique labels

    if (depth_limit != 0 and depth == depth_limit):
        return Node(most_common_label(dataset)), depth

    if len(check_labels) == 1: # if there is only one unique label return leaf node
    	return Node(check_labels[0]), depth # create leaf node with value of label and return
	
	# Dataset is not pure so find a split
    split_attribute, split_value = find_split(dataset)

	# Sort the dataset based on the split_attribute column
	# This is not strictly needed but makes it easier to read
	# column_to_sort = dataset.T[split_attribute]
	# sort_indices = column_to_sort.argsort()
	# dataset = dataset[sort_indices]

	# Need to divide the dataset into a left half and right half
    column_to_sort = dataset.T[split_attribute]

	# All elements less than split_value go to left
    left_dataset = dataset[column_to_sort < split_value]

	# All elements greater than split_value go to right
    right_dataset = dataset[column_to_sort >= split_value]

    new_node = Node(split_value, split_attribute) # create new node with split value

	# recursive call with left half of dataset
    if(left_dataset.size != 0):
    	new_node.left, l_depth = decision_tree_learning(left_dataset, depth + 1, depth_limit)

	# recursive call with left half of dataset
    if(right_dataset.size != 0):
    	new_node.right, r_depth = decision_tree_learning(right_dataset, depth + 1, depth_limit)

    return new_node, max(l_depth, r_depth)

"""
	predict_element(test_element, trained_tree)

OLD NAME: evaluation(test_db, trained_tree). Name was changed as evaluate() will be 
used to compute evaluation metrics and those are too similar. 

This function is used to predict the label of a simple input. It goes recursively through 
the tree until a leaf has been found and retrieve the leaf value when found. 
"""
def predict_element(test_element, trained_tree):
	if trained_tree.left == None and trained_tree.right == None:
		return trained_tree.value

	split_attribute = trained_tree.attribute
	result = 0
	if test_element[split_attribute] < trained_tree.value:
		result = predict_element(test_element, trained_tree.left)
	else:
		result = predict_element(test_element, trained_tree.right)
	return result

"""
	predict_db(test_db, trained_tree)

Predict the label of every element of a given database thanks to a trained decision tree.
It is simply calling predict_element(test_element, trained_tree) for each element of the dataset.
"""
def predict_db(test_db, trained_tree):
    y_predict = np.array([])
    nb_elements, _ = test_db.shape
    for i in range(0, nb_elements):
        test_element = test_db[i]
        result = predict_element(test_element, trained_tree)
        y_predict = np.append(y_predict, result)
    return y_predict

"""
	confusion_matrix(test_db, trained_tree)

Derive the confusion matrix of a trained decision tree classifier evaluated on test_db. 

Attention: this function automatically find the number of possible classes. This also means
that in the unluckily case were a sample misses one class, the confusion matrix will have 
a different size. 

-1-2---- predicted ----(n-1)-n-
1
-
2
-
-
true
-
-
(n-1)
-
n
-

------

NOTE: just make sure no error can be thrown
"""
def confusion_matrix(test_db, trained_tree):
    test_db_input, true_labels = test_db[:, :-1], test_db[:, -1]
    predicted_labels = predict_db(test_db_input, trained_tree)
    nb_classes = len(np.unique(true_labels))
    confusion_m = np.zeros((4, 4), dtype=int) #TODO: remove the hard-coded numbers
    for i in range(len(true_labels)):
        confusion_m[int(true_labels[i]) - 1, int(predicted_labels[i]) - 1] += 1
    return confusion_m

"""
	evaluate(test_db, trained_tree)

Compute the accuracy of the trained tree on the test_db dataset. The accuracy being the 
number of well classified inputs divided by the number of inputs. This function use the
confusion matrix to compute the accuracy.

accuracy = (TP + TN) / All
"""
def evaluate(test_db, trained_tree):
    confusion_m = confusion_matrix(test_db, trained_tree)
    n, _ = confusion_m.shape
    nb_well_predicted = np.trace(confusion_m)
    nb_predicted = np.ones((1, n))@confusion_m@np.ones((n, 1))
    return nb_well_predicted / nb_predicted[0, 0]

"""
	classes_metrics(confusion_m)

Get a bunch of evaluation metrics: accuracy, recall, precision, f1-score for each class. 
"""
def classes_metrics(confusion_m):
    n, _ = confusion_m.shape

    accuracies = np.array([])
    recalls = np.array([])
    precisions = np.array([])
    f1_scores = np.array([])

    for i in range(n):
        tp = confusion_m[i, i]
        tn = np.trace(confusion_m) - tp
        fn = sum(confusion_m[i, :i]) + sum(confusion_m[i, i+1:])
        fp = sum(confusion_m[:i, i]) + sum(confusion_m[i+1:, i])
        acc = (tp+tn)/(tp+tn+fn+fp)
        rec = tp/(tp+fn)
        prec = tp/(tp+fp)
        f1_sc = 2*rec*prec / (prec + rec)
        accuracies = np.append(accuracies, acc)
        recalls = np.append(recalls, rec)
        precisions = np.append(precisions, prec)
        f1_scores = np.append(f1_scores, f1_sc)

    return recalls, precisions, f1_scores, accuracies

"""
	macro_avg_metrics(confusion_m)

Get the macro averages of a few metrics. 
"""
def macro_avg_metrics(confusion_m):
    recalls, precisions, f1_scores, accuracies = classes_metrics(confusion_m)

    avg_metrics = {
		"recall": recalls.mean(),
		"precision": precisions.mean(),
		"f1-measure": f1_scores.mean(),
		"classification_rate": accuracies.mean()
	}

    return avg_metrics

"""
	cross_validation(dataset, k=10, seed=5)

Evaluation of the decision tree classifier algorithm thanks to a k-folds cross-validation
made on the given dataset. The evaluated metrics in the accuracy. 
"""
def cross_validation(dataset, k=10, seed=5):
    # copy to be safe with data shuffling
    data = np.copy(dataset)
	
    # shuffle the dataset first
    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    nb_labels = len(np.unique(data[:, -1]))
    avg_confusion_m = np.zeros((nb_labels, nb_labels), dtype=int)

    N, _ = data.shape
    a = N // k
    # last batch will be bigger than the others by (N % k, which is < k by definition)
    for i in range(0, k):
        test_db = data[a*i:a*(i+1), :]
        train_db = np.concatenate((data[:a*i, :], data[a*(i+1):, :]), axis=0)

        # train the decision tree
        trained_tree, _ = decision_tree_learning(train_db)

        # get the average confusion matrix of the decision tree
        confusion_m_i = confusion_matrix(test_db, trained_tree)
        if i == 0:
            avg_confusion_m = confusion_m_i
        else: # with this formula we avoid saving the confusion matrixs
            avg_confusion_m = (i / (i + 1)) * (avg_confusion_m + (confusion_m_i / i))

    return avg_confusion_m


############################## PRUNING ##############################

def cross_validation_with_pruning(dataset, test_db, k=9, seed=1): #k=9 IS DIFFERENT
    # copy to be safe with data shuffling
    data = np.copy(dataset)

    # shuffle the dataset first
    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    N, _ = data.shape
    a = N // k

    nb_labels = len(np.unique(data[:, -1]))
    avg_confusion_m = np.zeros((nb_labels, nb_labels), dtype=int)

    # last batch will be bigger than the others by N % k
    for i in range(0, k):
        val_db = data[a*i:a*(i+1), :]
        train_db = np.concatenate((data[:a*i, :], data[a*(i+1):, :]), axis=0)

        # train the decision tree and prune
        trained_tree, _ = decision_tree_learning(train_db)
        trained_tree = prune(trained_tree, val_db, train_db)

        # get the confusion matrix of the current tree
        confusion_m_i = confusion_matrix(test_db, trained_tree)
        if i == 0:
            avg_confusion_m = confusion_m_i
        else: # with this formula we avoid saving the confusion matrixs
            avg_confusion_m = (i / (i + 1)) * (avg_confusion_m + (confusion_m_i / i))

    return avg_confusion_m

def cross_cross_validation(dataset, k=10, seed=1):
    # copy to be safe with data shuffling
    data = np.copy(dataset)
    
    # shuffle the dataset first
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    
    nb_labels = len(np.unique(data[:, -1]))
    avg_confusion_m = np.zeros((nb_labels, nb_labels), dtype=int)

    N, _ = data.shape
    a = N // k
    # last batch will be bigger than the others by N % k
    for i in range(0, k):
        test_db = data[a*i:a*(i+1), :]
        train_db = np.concatenate((data[:a*i, :], data[a*(i+1):, :]), axis=0)

        # get the average confusion matrix of the decision trees in the inner cross validation
        confusion_m_i = cross_validation_with_pruning(train_db, test_db, k-1)
        if i == 0:
            avg_confusion_m = confusion_m_i
        else: # with this formula we avoid saving the confusion matrixs
            avg_confusion_m = (i / (i + 1)) * (avg_confusion_m + (confusion_m_i / i))

    return avg_confusion_m

# returns the most common label in a given dataset
def most_common_label(dataset):
    label, counts = np.unique(dataset.T[7], return_counts=True)

    return label[np.argmax(counts)]

# returns true if tree is a leaf, false otherwise
def isLeaf(tree):
    return (tree.left == None and tree.right == None)

# visits all the parents of two leaves and decides if it is worth pruning them, returns the (un)pruned tree
def prune_rec(tree, valid_set, train_set):
    # if is parent of two leaves, consider pruning
    if isLeaf(tree.left) and isLeaf(tree.right):
        if valid_set.shape[0] == 0:
            return tree
        else:
            potential_label = most_common_label(train_set)
            accuracy_before = evaluate(valid_set, tree)
            accuracy_after = np.sum(valid_set.T[7] == potential_label) / valid_set.shape[0]

            # prune if the pruned version performs better on validation set
            if accuracy_after >= accuracy_before:
                tree.left = None
                tree.right = None

                tree.value = potential_label

            return tree
    else:
        split_attribute = tree.attribute
        split_value = tree.value

        # split left and right subsets for validation
        column_to_sort = valid_set.T[split_attribute]
        left_valid_subset = valid_set[column_to_sort < split_value]
        right_valid_subset = valid_set[column_to_sort >= split_value]

        # split left and right subsets for validation
        column_to_sort = train_set.T[split_attribute]
        left_train_subset = train_set[column_to_sort < split_value]
        right_train_subset = train_set[column_to_sort >= split_value]

        # recurse to the left an to the right if not leaves
        if not isLeaf(tree.left):
            tree.left = prune_rec(tree.left, left_valid_subset, left_train_subset)

        if not isLeaf(tree.right):
            tree.right = prune_rec(tree.right, right_valid_subset, right_train_subset)

        return tree

# returns the tree after pruning it as many times as possible 
def prune(tree, valid_set, train_set):
    accuracy_before = evaluate(valid_set, tree)
    is_improving = True

    while is_improving:
        pruned_tree = prune_rec(tree, valid_set, train_set)
        accuracy_after_pruning = evaluate(valid_set, pruned_tree)

        # if the accuracy is the same after executing prune_rec, it means that we didn't prune any leaf, thus, pruning won't improve accuracy anymore
        # the condition could be equivalently written as accuracy_before == accuracy_after_pruning
        if accuracy_before >= accuracy_after_pruning:
            is_improving = False
        else:
            accuracy_before = accuracy_after_pruning
            tree = pruned_tree

    return pruned_tree


################################ TREE PLOTTING #################################


def going_left(pos, v_step, parent):
    # function that returns the position of the left child for the parent node and the vertical step given
    return (pos[0]-np.abs(parent[0]-pos[0])/2, pos[1]-v_step)

def going_right(pos, v_step, parent):
    # function that returns the position of the right child for the parent node and the vertical step given
    return (pos[0]+np.abs(parent[0]-pos[0])/2, pos[1]-v_step)




def plotTree(ourTree, tr, depth_max, v_step, pos, parent):
    # recursive function that builds the tr subplot which contains all the structure of the tree

    val = ourTree.value
    attr = ourTree.attribute

    #preparing the text message at this node
    if ourTree.left == None: # n case of a leaf, we state the label found
        txt = "leaf :" + str(val)
    else: #the splitting condition
        txt = "X" + str(attr) + "<" + str(val)
    
    #creating the arrow to the parent, and the text box for the current node
    tr.annotate(txt,
            xy=parent, xycoords='data',
            xytext=pos, va='center',ha='center',size=6,
            arrowprops=dict(arrowstyle="<-"))

    #if we are not at a leaf, get the positions for the two children, and call the fonction again
    if ourTree.left != None:
        pos_left = going_left(pos, v_step, parent)
        pos_right = going_right(pos, v_step, parent)
        plotTree(ourTree.left, tr, depth_max, v_step, pos_left, pos)
        plotTree(ourTree.right, tr, depth_max, v_step, pos_right, pos)
    

"""
	viz(ourTree, depth_max, x_max, y_max)

Main function that initialises the figure, the subplot and the first node. 
Then it starts the recursion of the precedent function.
"""
def viz(ourTree, depth_max, x_max = 10, y_max = 10):
    #main function that initialises the figure, the subplot and the first node. 
    #then it starts the recursion of the precedent function.

    #initialising the pyplot object
    fig = plt.figure(figsize = (x_max, y_max))
    v_step = (y_max - 1) / depth_max
    tr = fig.add_subplot(111, xlim = (0, x_max), ylim = (0, y_max))
    starting_point = (x_max / 2, y_max - 1)

    #first node visualisation : apart from the rest because of the different first arrow
    val = ourTree.value
    attr = ourTree.attribute
    if ourTree.left == None:
        txt = "leaf :" + str(val)
    else:
        txt = "X" + str(attr) + "<" + str(val)
    tr.annotate(txt,
            xy = (x_max / 2, y_max), xycoords = 'data',
            xytext = starting_point,
            va = 'center', ha = 'center', size = 6,
            arrowprops = dict(arrowstyle = "<-"))
    
    #calling the building recursive function on the children of the first node
    if ourTree.left != None:
        plotTree(ourTree.left, tr, depth_max, v_step, going_left(starting_point, v_step, (0, 0)), starting_point)
        plotTree(ourTree.right, tr, depth_max, v_step, going_right(starting_point, v_step, (0, 0)), starting_point)

    #showing the result
    plt.show()



################################     DEMO     ##################################

"""
	main()

This function is used to put demonstrations of the functions that were coded. It is 
automatically launched if the script is run (but not if imported). And it can also be used
when the script is imported: import CW1; CW1.main()
"""
def main():
    np.set_printoptions(precision=3)

    # loading both datasets
    print("Loading clean dataset... ", end="", flush=True)
    clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt") # loads data as matrix from file	
    print("done. ")

    print("Loading noisy dataset... ", end="", flush=True)
    noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")
    print("done.")

    # perform 10-fold cross validation on clean
    print("Performing 10-fold cross validation on clean dataset... ", end="", flush=True)
    confusion_m_clean = cross_validation(clean_dataset, 10)
    print("done. ")

    # perform 10-fold cross validation on clean
    print("Performing 10-fold cross validation on noisy dataset... ", end="", flush=True)
    confusion_m_noisy = cross_validation(noisy_dataset, 10)
    print("done. ")

    # perform 10-fold cross validation on clean with pruning
    print("Performing 10-fold cross validation on clean dataset with pruning... ", end="", flush=True)
    confusion_m_clean_pruning = cross_cross_validation(clean_dataset, 10)
    print("done. ")

    # perform 10-fold cross validation on noisy with pruning
    print("Performing 10-fold cross validation on noisy dataset with pruning... ", end="", flush=True)
    confusion_m_noisy_pruning = cross_cross_validation(noisy_dataset, 10)
    print("done. ")

    print("\n\n-------- Results ---------\n\n")

    # print all results for clean dataset
    print("<<<< CLEAN DATASET >>>>\n")
    print("Average confusion matrix:\n", confusion_m_clean)
    recalls, precisions, f1_scores, accuracies = classes_metrics(confusion_m_clean)
    macro_metrics = macro_avg_metrics(confusion_m_clean)
    print("Recall: ", recalls)
    print("Precision: ", precisions)
    print("F1 score: ", f1_scores)
    print("Accuracy: ", accuracies)
    print("\nMacro averaged metrics:\n", macro_metrics)

    print("\n")

    # print all results for noisy dataset
    print("<<<< NOISY DATASET >>>>\n")
    print("Average confusion matrix:\n", confusion_m_noisy)
    recalls, precisions, f1_scores, accuracies = classes_metrics(confusion_m_noisy)
    macro_metrics = macro_avg_metrics(confusion_m_noisy)
    print("Recall: ", recalls)
    print("Precision: ", precisions)
    print("F1 score: ", f1_scores)
    print("Accuracy: ", accuracies)
    print("\nMacro averaged metrics:\n", macro_metrics)

    print("\n")

    # print all results for clean dataset with pruning
    print("<<<< CLEAN DATASET WITH PRUNING >>>>\n")
    print("Average confusion matrix:\n", confusion_m_clean_pruning)
    recalls, precisions, f1_scores, accuracies = classes_metrics(confusion_m_clean_pruning)
    macro_metrics = macro_avg_metrics(confusion_m_clean_pruning)
    print("Recall: ", recalls)
    print("Precision: ", precisions)
    print("F1 score: ", f1_scores)
    print("Accuracy: ", accuracies)
    print("\nMacro averaged metrics:\n", macro_metrics)

    print("\n")

    # print all results for noisy dataset with pruning
    print("<<<< NOISY DATASET WITH PRUNING >>>>\n")
    print("Average confusion matrix:\n", confusion_m_noisy_pruning)
    recalls, precisions, f1_scores, accuracies = classes_metrics(confusion_m_noisy_pruning)
    macro_metrics = macro_avg_metrics(confusion_m_noisy_pruning)
    print("Recall: ", recalls)
    print("Precision: ", precisions)
    print("F1 score: ", f1_scores)
    print("Accuracy: ", accuracies)
    print("\nMacro averaged metrics:\n", macro_metrics)


if __name__ == "__main__":
	main()