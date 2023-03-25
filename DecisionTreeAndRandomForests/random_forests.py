import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Decision tree libraries:
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

# libraries to visualize ythe decision tree:
# from IPython.display import Image  
# from sklearn.externals.six import StringIO  
# from sklearn.tree import export_graphviz
# import pydot 

# libraries to work with random forests:
from sklearn.ensemble import RandomForestClassifier


# getting the file with the dat and loading into the variable dataframe(df):
"""
Context: this data frame have information about children with kyphosis.
the 0 column are the indexes, the 1 are the info if the kyphosis is present on the 
patient of the respective index, the 2 is the age in months, the 3 is the number of affected vertebras, and the 4th 
are the number of affected vertebras.
"""
df = pd.read_csv('kyphosis.csv')
print('\n The information about the dataf frame is the following:\n', df.info())
# plt.show()
print(df.head())

# to visualize the data:
sns.pairplot(df, hue = 'Kyphosis')
sns.pairplot(df,hue='Kyphosis',palette='Set1')


# Now, dividing the data in a training group and a testing group
x = df.drop('Kyphosis',axis=1) # deleting the Kyphosis column 
y = df['Kyphosis']  # assigning the Kyphosis information into the y variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


# Now, training a single decision tree:
dtree = DecisionTreeClassifier() # this is a model of a decision tree

dtree.fit(x_train,y_train)
print(dtree)

# getting the predictions from the decision tree:
predictions = dtree.predict(x_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

# Visualizing the tree:
features = list(df.columns[1:])
features

# dot_data = StringIO()  
# export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

# graph = pydot.graph_from_dot_data(dot_data.getvalue())  
# Image(graph[0].create_png())  


# Now to random forests: rfc= random florest classifier
rfc = RandomForestClassifier(n_estimators=100) # 100 random trees 
rfc.fit(x_train, y_train) # the same parameters as the used in the decision tree

rfc_pred = rfc.predict(x_test)

print('Confusion matrix:/n',confusion_matrix(y_test,rfc_pred))
print('Classification report:/n',classification_report(y_test,rfc_pred))