import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

from os import listdir
from os.path import isfile, join

"""
data_path = './data/pivot/'
chunk = pd.read_csv(join(data_path, '001.csv'), index_col=0)
print(chunk.columns)

"""
# 0. Declare hyper parameter, and function for getting gdelt
day_gap = 0
num_classes = 7

# 5. split features vs targets
S = pd.read_csv("./data/train_df.csv", index_col=0)
print(S.shape)
target_col_name = 'curr_grd'

targets = S[target_col_name]
targets = targets.replace(0,1)
targets = targets - 1
# print(targets.unique())
# print("targets.shape:", targets.shape)

features = S.drop(labels=target_col_name, axis=1)
# I want to find date-independent model, so drop strt_dt and end_dt
features = S.drop(labels=['strt_dt','end_dt'], axis=1)
print("features.shape1:", features.shape)
dictOfCtry = { S['ctry'].unique()[i] : i for i in range(0, len(S['ctry'].unique()) ) }
features = features.replace(dictOfCtry)

# 6. Prepare a validation set + 10fold cross validation
fold_cv = 10
train_features = pd.DataFrame()
train_targets = pd.DataFrame()
val_features = pd.DataFrame()
val_targets = pd.DataFrame()

for k_idx in range(fold_cv):
    strt_pt = int(k_idx * len(features) / fold_cv)
    end__pt = int((k_idx+1) * len(features) / fold_cv)
    train_features = pd.concat([train_features,features[:strt_pt-1]])
    train_targets = pd.concat([train_targets,targets[:strt_pt-1]])
    val_features = pd.concat([val_features,features[strt_pt:end__pt-1]])
    val_targets = pd.concat([val_targets,targets[strt_pt:end__pt-1]])
    train_features = pd.concat([train_features,features[end__pt:]])
    train_targets = pd.concat([train_targets,targets[end__pt:]])

print(len(train_features),len(val_features))

from tensorflow.keras.utils import to_categorical
train_targets_cate = to_categorical(train_targets, num_classes)
val_targets_cate   = to_categorical(val_targets, num_classes)

print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))

# 7. Normalize the data using training set statistics
mean = np.mean(train_features, axis=0)
train_features -= mean
val_features -= mean
std = np.std(train_features, axis=0)
train_features /= std
val_features /= std

train_features = train_features.dropna(axis=1, how='all')
val_features   = val_features.dropna(axis=1, how='all')
print("features.shape2:", train_features.shape)
print("features.shape2:", val_features.shape)


# 8. Build a multi-classification model
from tensorflow import keras
print(train_features.shape[-1])

model = keras.Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(train_features.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax'),
    ]
)
model.summary()

# 9. Train the model with curr_grd argument
metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="categorical_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]

model.fit(
    train_features,
    train_targets_cate,
    batch_size=2048,
    epochs=30,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets_cate),
)
