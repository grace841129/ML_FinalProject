import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob  
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image                  
from tqdm import tqdm
from keras import regularizers
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout,Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 

img_width, img_height = 224, 224

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_labels = load_dataset('dogImages/train')
valid_files, valid_labels = load_dataset('dogImages/valid')
test_files, test_labels = load_dataset('dogImages/test')

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
dog_breeds = len(dog_names)

bottleneck_features = np.load('InceptionV3.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']

inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
inception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
inception_model.add(Dropout(0.4))
inception_model.add(Dense(dog_breeds, activation='softmax'))

inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
'''
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

inception_model.fit(train_InceptionV3, train_labels, 
          validation_data=(valid_InceptionV3, valid_labels),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
'''
inception_model.load_weights('saved_models/weights.best.InceptionV3805024.hdf5')

InceptionV3_predictions = [np.argmax(inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_labels, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

def  extract_InceptionV3 (tensor):  
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    
top_N = 3

def predict_breed(path):
    print('Predicting...')
    image_tensor = path_to_tensor(path)

    bottleneck_features = extract_InceptionV3(image_tensor)

    prediction = inception_model.predict(bottleneck_features)[0]

    breeds_predicted = [dog_names[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]

    return breeds_predicted, confidence_predicted

def make_prediction(path, multiple_breeds = False):
    breeds, confidence = predict_breed(path)
    img = mpimg.imread(path)
    plt.axis('off')

    plt.imshow(img)
    print('Breed:{}.'.format(breeds[0].replace("_", " ")))
    c=confidence[0]*100
    print('accuracy: %.4f%%'%c)
        
    if multiple_breeds == True:
        print('\n\nTop 3 predictions:')
        for i, j in zip(breeds, confidence):
            print('{} with a accuracy of {:.4f}'.format(i.replace("_", " "), j))

a=input("Enter the image:")
b=input("Is the dog mixed breed?[y/n]:")
if b == 'y':
    mul=True
else:
    mul=False
make_prediction(a,mul)
