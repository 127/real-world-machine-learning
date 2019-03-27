import numpy as np
cat_data = np.array(['male', 'female', 'male', 'male', 'female', 'male', 'female', 'female'])
# print(cat_data)
def cat_to_num(data):
    categories = np.unique(data)
    # print(categories)
    features = []
    for cat in categories:
        binary = (data == cat)
        features.append(binary.astype("int"))
        # print(binary, cat, features)
    return features
    
a = cat_to_num(cat_data)
# print(a)

cabin_data = np.array(["C65", "", "E36", "C54", "B57 B59 B63 B66"])
def cabin_features(data):
    features = []
    for cabin in data:
        cabins = cabin.split(" ")
        n_cabins = len(cabins)
        # First char is the cabin_char
        try:
            cabin_char = cabins[0][0]
        except IndexError:
            cabin_char = "X"
            n_cabins = 0
        # The rest is the cabin number
        try:
            cabin_num = int(cabins[0][1:]) 
        except:
            cabin_num = -1
        # Add 3 features for each passanger
        features.append( [cabin_char, cabin_num, n_cabins] )
    return features
    
num_data = np.array([3, 10, 0.5, 43, 0.12, 8])

def normalize_feature(data, f_min=-1, f_max=1):
    d_min, d_max = min(data), max(data)
    factor = (f_max - f_min) / (d_max - d_min)
    normalized = f_min + data*factor
    return normalized, factor
x = normalize_feature(num_data)
print(x)