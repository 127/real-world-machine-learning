from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from keras import optimizers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['Расход топлива','Кол-во цилиндров','Объем двигателя','Л.с.','Вес',
                'Разгон до 100 км/ч', 'Год выпуска', 'Страна выпуска'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()
dataset = dataset.dropna()

origin = dataset.pop('Страна выпуска')
dataset['США'] = (origin == 1)*1.0
dataset['Европа'] = (origin == 2)*1.0
dataset['Япония'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["Расход топлива", "Кол-во цилиндров", "Объем двигателя", "Вес"]], diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop("Расход топлива")
train_stats = train_stats.transpose()
# print(train_stats)

train_labels = train_dataset.pop('Расход топлива')
test_labels = test_dataset.pop('Расход топлива')

# print(train_dataset, train_stats['mean'], train_stats['std']0)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  
  # print(dir(tf.train.experimental))

  # optimizer = tf.train.CheckpointableBase.Optimizer.RMSPropOptimizer(learning_rate=0.001)
  optimizer = keras.optimizers.RMSprop(lr=0.001)
  model.compile(loss='mse',
                optimizer=keras.optimizers.RMSprop(lr=0.001),
                metrics=['mae', 'mse'])
  return model
  
  
model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

# from pprint import pprint

# pprint(example_result)

# Выведем прогресс обучения в виде точек после каждой завершенной эпохи
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
  
  
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
# print(hist.tail())


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure(figsize=(8,12))
  
  plt.subplot(2,1,1)
  plt.xlabel('Эпоха')
  plt.ylabel('Среднее абсолютное отклонение')
  # pprint(hist)
  plt.plot(hist['epoch'], hist['mae'],
           label='Ошибка при обучении')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Ошибка при проверке')
  plt.ylim([0,5])
  plt.legend()
  
  plt.subplot(2,1,2)
  plt.xlabel('Эпоха')
  plt.ylabel('Среднеквадратическая ошибка')
  plt.plot(hist['epoch'], hist['mse'],
           label='Ошибка при обучении')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Ошибка при проверке')
  plt.ylim([0,20])
  plt.legend()
  # plt.show()

model = build_model()

# Параметр patience определяет количество эпох, которые можно пропустить без улучшений
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Среднее абсолютное отклонение на проверочных данных: {:5.2f} галлон на милю".format(mae))



test_predictions = model.predict(normed_test_data).flatten()

plt.clf()   # Очистим график
plt.scatter(test_labels, test_predictions)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()