import pandas
import numpy as np
import matplotlib.pyplot as plt
data = pandas.read_csv("data/titanic.csv")
data[:5]
# We make a 80/20% train/test split of the data
data_train = data[:int(0.8*len(data))]
data_test = data[int(0.8*len(data)):]

# print(data_train[:5])
# print(data_test[:5])

# The categorical-to-numerical function from chapter 2
# Changed to automatically add column names
def cat_to_num(data):
    categories = np.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s=%s" % (data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)
    
def prepare_data(data):
    """Takes a dataframe of raw data and returns ML model features
    """
    
    # Initially, we build a model only on the available numerical values
    features = data.drop(["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    
    # Setting missing age values to -1
    features["Age"] = data["Age"].fillna(-1)
    
    # Adding the sqrt of the fare feature
    features["sqrt_Fare"] = np.sqrt(data["Fare"])
    
    # Adding gender categorical value
    features = features.join( cat_to_num(data['Sex']) )
    
    # Adding Embarked categorical value
    # features = features.join( cat_to_num(data['Embarked']) )
    
    return features
    
# print(cat_to_num(data['Embarked']))
features = prepare_data(data_train)
# print(features)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features, data_train["Survived"])
# Make predictions
a = model.predict(prepare_data(data_test))
# print(a)
model.score(prepare_data(data_test), data_test["Survived"])
# plt.plot(prepare_data(data_test), a, 'o')
# x = np.linspace(10,40,5)
# plt.plot(x, x, '-');
# plt.show()
# from sklearn.svm import SVC
# model = SVC()
# a = model.fit(features, data_train["Survived"])
# # print(a)
# b =model.score(prepare_data(data_test), data_test["Survived"])
# print(b)