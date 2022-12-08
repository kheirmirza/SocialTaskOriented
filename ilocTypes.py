import csv

with open('ilocTypes', newline='') as f:
    reader = csv.reader(f)
    l = list(reader)
    print(list(set(l[0])))
    # print(list(set(list(first)[1:])))

# ['Request', 'Order', 'Command', 'Question', 'Prediction']