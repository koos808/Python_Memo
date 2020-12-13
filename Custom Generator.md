## Custom Generator example code

### 1. Example Code 1
* Multi image generator
```
def generator_two_img(X1, X2, y, batch_size):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]
```

* Three image generator input
```
def generator_three_img(X1, X2, X3, y, batch_size):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
    genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
    genX3 = gen.flow(X3, y, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]
```

* Multiple Input Generator
```
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class MultipleInputGenerator(Sequence):
    """Wrapper of 2 ImageDataGenerator"""

    def __init__(self, X1, X2, Y, batch_size):
        # Keras generator
        self.generator = ImageDataGenerator(rotation_range=15, 
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True, 
                                            fill_mode='nearest')

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow(X1, Y, batch_size=batch_size)
        self.genX2 = self.generator.flow(X2, Y, batch_size=batch_size)

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)

        X_batch = [X1_batch, X2_batch]

        return X_batch, Y_batch
```

### 2. Example Code 2

```

```
