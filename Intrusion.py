#####################################  ACCELERATED PSO #####################################

def MAIN_ACC_PSO():
    import numpy as np
    import timeit
    
    from sklearn.datasets import load_iris
    from sklearn.datasets import fetch_kddcup99
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    import urllib 

############################### LOOADING OF THE DATASET ####################################


def ACC_PSO_LDA():

    import matplotlib.pyplot as plt
    import timeit
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    start_time1 = timeit.default_timer()
    X, X_new, y, target_names =optimization() # Optimization is not defined
   
    print("Dataset before applying the feature selection:")
    print(X)
    print("Dataset after applying the feature selection:")
    print(X_new)
    print(y)
    # X= np.array([[-1, -1],[-2,-1], [-3, -2], [1,1],[2,1],[3,2]])
    # y = np.array([1, 1, 1, 2, 2, 2])
    clf = LinearDiscriminantAnalysis();
    X_r2 = clf.fit(X_new, y).transform(X_new)

################################## PREDICTION OF THE SAMPLES ##############################

    # print(" The sample [0.3,0.8,0.9,0.5] belongs to the class:")
    # print(clf.predict([[0.3,0.8,0.9, 0.5]]))
    # print(" The sample [6,148,72,35] belongs to the class:")
    # print(clf.predict([[6,148,72,35]]))
    # print(" The sample [0, 0.5,0.6,0.8] belongs to the class:")
    # print(clf.predict([[0, 0.5,0.6,0.8]]))

    print(" The sample [0.3,0.8] belongs to the class:")
    print(clf.predict([[0.3,0.8,0.7]]))
    print(" The sample [6,148] belongs to the class:")
    print(clf.predict([[6,148,345]]))
    print(" The sample [0, 0.5] belongs to the class:")
    print(clf.predict([[0, 0.5,7.8]]))

    elapsed1 = timeit.default_timer() - start_time1
    print(" The time of execution:", elapsed1 )

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    #colors = ['navy', 'turquoise']
    lw = 2
    

    #for color, i, target_name in zip(colors, [0, 1, 2], target_names):
       #plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                   #label=target_name)
    #plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.title('PCA of IRIS dataset')
    
    plt.figure()
    for color, i, target_name in zip(colors, [0,1,2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 0], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('ACC_PSO_LDA of PIMA dataset')
    plt.show()


##################################### PERFORMANCE EVALUATION ################################

def PERFORMANCE_EVAL():
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import cycle
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # This is our classifier
    from sklearn import datasets
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier

#################################### Import some data to play around with ##################

    X_Pre, X_new, y, target_names = MAIN_ACC_PSO()

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2
    X = X_new
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Split into training and test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=random_state)
    print(" The Original dataset is:")
    print(X)
    print(" The trainning dataset(data) is :")
    print(X_train)
    print(" The testing dataset(data) is :")
    print(X_test)
    print(" The trainning dataset(target) is :")
    print(y_train)
    print(" The trainning dataset(target) is:")
    print(y_test)

   # Run classifier
    classifier = OneVsRestClassifier(LinearDiscriminantAnalysis())
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

#ACC_PSO_LDA()
PERFORMANCE_EVAL()


