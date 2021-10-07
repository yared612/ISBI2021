import pandas as pd
data = pd.read_csv('/home/user/Downloads/RFMiD_Training_Labels.csv')
data_name = data.columns
one = []
for i in data_name:
    one_hot = pd.get_dummies(data[i])
    one.append(one_hot)
one.pop(0)
one.pop(0)
GT = pd.concat([one[0],one[1]],axis=1)
for j in range(2,len(one)):
    GT = pd.concat([GT,one[j]],axis=1)
    

# GT = pd.concat([one[0],one[1],one[2],one[3],one[4],one[5],one[6],one[7],one[8],one[9],one[10],one[11],one[12],one[13],one[14],one[15],one[16],one[17],one[18],one[19],one[20],one[21],one[22],one[23],one[24],one[25],one[26],one[27]],axis=1)