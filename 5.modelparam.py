import pandas
#pip install xlrd==1.2.0
import matplotlib.pyplot as plt
from sklearn import metrics

def auc(actual, pred):
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

def roc_plot(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Position Rate')
    plt.ylabel('True Position Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

data = pandas.read_excel('./1.xlsx', sheet_name=7)

label, probability, prediction = data['label'], data['probability'], data['prediction'] 
print(auc(label, probability))

fpr, tpr, thresholds = metrics.roc_curve(label, probability, pos_label=1)
#print(fpr, tpr, thresholds)
roc_plot(fpr, tpr)

cm = metrics.confusion_matrix(label, prediction)
print(cm)

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

print(TP, FN, FP, TN)

sensitivity = TP / (TP + FN + .0)
specificity = TN / (TN + FP + .0)
precision = TP/ (TP + FP + .0)
accuracy = (TP + TN) / (TP + FN + FP + TN + .0)
F1 = 2 * sensitivity * precision / (sensitivity + precision)
print(sensitivity, specificity, precision, accuracy, F1)
print('301: 1.0 0.9380804953560371 0.8 0.9503722084367245 0.888888888888889')
print('PUMCH: 0.9861111111111112 0.9369024856596558 0.6826923076923077 0.9428571428571428 0.8068181818181818')
print('CHCAMS: 1.0 0.9680851063829787 0.9073359073359073 0.9756838905775076 0.951417004048583')
