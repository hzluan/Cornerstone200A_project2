import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from src.dataset import NLST

def main():
    nlst = NLST(**vars(NLST))
    nlst.setup()
    # test_loader = nlst.test_dataloader()
    test = nlst.test.dataset
    y_seq = []
    lung_rads = []
    i = 0
    for x in test:
        y_seq.append(x['y_seq'][0])
        lung_rads.append(x["lung_rads"])

# import model predictions for test
# calculate how many positives are captured by lung rad and model
# calculate how many negatives are captured by lung rad but not model
# calculate how many negatives are captured by model but not lung rad
#      lung rad 1       lung rad 0
# true  TP               FN
# false FP               TN
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y_seq)):
        if y_seq[i] == 1 and lung_rads[i] == 1:
            TP += 1
        elif y_seq[i] == 1 and lung_rads[i] == 0:
            FN += 1
        elif y_seq[i] == 0 and lung_rads[i] == 1:
            FP += 1
        elif y_seq[i] == 0 and lung_rads[i] == 0:
            TN += 1


    print('###################')
    print('Sensitivity is ' + str(TP / (TP + FN)))
    print('Specificity is ' + str(TN / (TN + FP)))
    print('PPV is ' + str(TP / (TP + FP)))
    print('NPV is ' + str(TN / (TN + FN)))
    print('C-Index is ' + str((TP / (TP + FN) + TN / (TN + FP)) / 2))
    print(TP)
    print(FN)
    print(FP)
    print(TN)

if __name__ == "__main__":
    main()
    