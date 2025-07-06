{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d8c5de0e-f9e7-4d1c-bf5d-d1ed66209daf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.applications import ResNet50\n",
    "from tensorflow.keras.layers import Dense, Flatten, Dropout\n",
    "from tensorflow.keras.models import Sequential\n",
    "\n",
    "# تحميل نموذج ResNet50 مع استبعاد الطبقات العلوية\n",
    "base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "\n",
    "# إضافة طبقات للتصنيف النهائي\n",
    "model = Sequential([\n",
    "    base_model,\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.5),\n",
    "    Dense(1, activation='sigmoid')  # لتصنيف المرض (0: سليم, 1: مرض)\n",
    "])\n",
    "\n",
    "# تجميد الطبقات الأصلية لنموذج ResNet50\n",
    "base_model.trainable = False\n",
    "\n",
    "# تجميع النموذج\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# حفظ النموذج\n",
    "model.save('xray_model.h5')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "598009c8-86e1-4438-902e-cbc687acdd83",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TensorFlow version: 2.8.0\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "print(\"TensorFlow version:\", tf.__version__)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fe98cafb-2154-4843-baed-8903d0598189",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.applications import ResNet50\n",
    "from tensorflow.keras.layers import Dense, Flatten, Dropout\n",
    "from tensorflow.keras.models import Sequential\n",
    "\n",
    "# تحميل نموذج ResNet50 مع استبعاد الطبقات العلوية\n",
    "base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "\n",
    "# إضافة طبقات للتصنيف النهائي\n",
    "model = Sequential([\n",
    "    base_model,\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.5),\n",
    "    Dense(1, activation='sigmoid')  # لتصنيف المرض (0: سليم, 1: مرض)\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "afdbea05-656d-4b18-8580-f174f61ecebc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# تجميد الطبقات الأصلية لنموذج ResNet50\n",
    "base_model.trainable = False\n",
    "\n",
    "# تجميع النموذج\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "22781f50-d488-4e8b-90d1-28c62bcfd182",
   "metadata": {},
   "outputs": [],
   "source": [
    "# افتراض وجود بيانات للتدريب، مثل train_data و validation_data\n",
    "# model.fit(train_data, epochs=10, validation_data=validation_data)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fb31bcc7-55be-495b-80b3-7b6bfd801f84",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('xray_model.h5')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9145357c-9435-4791-ad1f-23cf9bfbde4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Num GPUs Available:  0\n"
     ]
    }
   ],
   "source": [
    "print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "3288992d-1c05-4377-99f5-08567e20dbc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 5216 images belonging to 2 classes.\n",
      "Found 16 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "# تهيئة بيانات التدريب\n",
    "train_datagen = ImageDataGenerator(rescale=1./255)\n",
    "train_data = train_datagen.flow_from_directory(\n",
    "    'C:/Users/a7sn4/Desktop/xray/train',  # المسار الصحيح لمجلد الصور\n",
    "    target_size=(224, 224),\n",
    "    batch_size=32,\n",
    "    class_mode='binary'  # أو 'categorical' إذا كانت التصنيفات متعددة\n",
    ")\n",
    "\n",
    "# تهيئة بيانات التحقق\n",
    "validation_datagen = ImageDataGenerator(rescale=1./255)\n",
    "validation_data = validation_datagen.flow_from_directory(\n",
    "    'C:/Users/a7sn4/Desktop/xray/validation',  # المسار الصحيح لمجلد التحقق\n",
    "    target_size=(224, 224),\n",
    "    batch_size=32,\n",
    "    class_mode='binary'  # أو 'categorical' إذا كانت التصنيفات متعددة\n",
    ")\n",
    "\n",
    "# الآن يمكنك استخدام train_data و validation_data لتدريب النموذج\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "534c35d2-82f3-44a2-9385-712079e4e438",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "163/163 [==============================] - 436s 3s/step - loss: 0.8275 - accuracy: 0.7304 - val_loss: 0.8592 - val_accuracy: 0.5625\n",
      "Epoch 2/10\n",
      "163/163 [==============================] - 423s 3s/step - loss: 0.4837 - accuracy: 0.7431 - val_loss: 0.8278 - val_accuracy: 0.5000\n",
      "Epoch 3/10\n",
      "163/163 [==============================] - 421s 3s/step - loss: 0.5182 - accuracy: 0.7418 - val_loss: 0.7586 - val_accuracy: 0.5000\n",
      "Epoch 4/10\n",
      "163/163 [==============================] - 431s 3s/step - loss: 0.4873 - accuracy: 0.7429 - val_loss: 0.9122 - val_accuracy: 0.5000\n",
      "Epoch 5/10\n",
      "163/163 [==============================] - 438s 3s/step - loss: 0.4894 - accuracy: 0.7429 - val_loss: 0.7607 - val_accuracy: 0.5000\n",
      "Epoch 6/10\n",
      "163/163 [==============================] - 453s 3s/step - loss: 0.4862 - accuracy: 0.7429 - val_loss: 0.6698 - val_accuracy: 0.5000\n",
      "Epoch 7/10\n",
      "163/163 [==============================] - 462s 3s/step - loss: 0.4792 - accuracy: 0.7429 - val_loss: 0.8275 - val_accuracy: 0.5000\n",
      "Epoch 8/10\n",
      "163/163 [==============================] - 450s 3s/step - loss: 0.4764 - accuracy: 0.7429 - val_loss: 0.9642 - val_accuracy: 0.5000\n",
      "Epoch 9/10\n",
      "163/163 [==============================] - 447s 3s/step - loss: 0.4759 - accuracy: 0.7429 - val_loss: 0.9530 - val_accuracy: 0.5000\n",
      "Epoch 10/10\n",
      "163/163 [==============================] - 437s 3s/step - loss: 0.4693 - accuracy: 0.7429 - val_loss: 0.7838 - val_accuracy: 0.5000\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x258cc6a7250>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(\n",
    "    train_data,\n",
    "    epochs=10,\n",
    "    validation_data=validation_data\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "fc2f9dfc-2b14-4324-994b-661196a6f4bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 1s 1s/step - loss: 0.7838 - accuracy: 0.5000\n",
      "Validation Loss: 0.7838309407234192\n",
      "Validation Accuracy: 0.5\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the model on the validation data\n",
    "validation_loss, validation_accuracy = model.evaluate(validation_data)\n",
    "print(f'Validation Loss: {validation_loss}')\n",
    "print(f'Validation Accuracy: {validation_accuracy}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b9f69ea-8540-47a5-8ab7-baf29408dcd0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
