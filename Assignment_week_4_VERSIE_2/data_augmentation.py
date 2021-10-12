#Code for the data augmentation
#Group: 4, Course: 8DM50

import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import gryds

def show_data_aug_effect(images):
    """
    Function that generates example images with brightness augmentation
    :param images: data array, containing data of the training images
    :returns nothing but shows 3 by 3 subplots with random brightness augmentation
    """

    # load the image and convert to numpy array
    img = images[0]
    data = img_to_array(img)
    
    # expand dimension to one sample
    samples = np.expand_dims(data, 0)
    
    # create image data augmentation generator and prepare iterator
    datagen = ImageDataGenerator(brightness_range=[0.4,1.3])
    it = datagen.flow(samples, batch_size=1)
    
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()
    
def brightness_augmentation_data(images,segmentations,nr_new_images = 1,range_brightness=[0.4,1.3]):
    """
    Function that generates new images with brightness augmentation
    :param images: numpy array, containing input data of the training images
    :param segmentations: numpy array, containing segmentation data of the training images
    :param nr_new_images: integer, determines the number of generated images per one image, default is 1
    :param range_brightness: list, containing the minimum and maximum brightness values, default is [0.4,1.3]
    
    :return train_X_aug: numpy array, containing augmented data set
    :return train_segmentations_aug: numpy array, containing segmentations augmented data set
    :return train_X_tot: numpy array, containing complete data set (real + augmented)
    :return train_segmentations_aug_tot: numpy array, containing segmentations complete data set
    """
    
    datagen = ImageDataGenerator(brightness_range = range_brightness)
    X_aug = np.array([]) # array in which the new, generated images can be stored
    train_segmentations_aug = np.array([]) # array in which to store the 

    for i in range(len(images)): # For every image in the training data set, generate new images for the augmented dataset
        train_im = images[i]
        y_val = segmentations[i]
        y_val = np.expand_dims(y_val, 0)
        
        data = img_to_array(train_im)
        samples = np.expand_dims(data, 0)
        it = datagen.flow(samples, batch_size = nr_new_images) 
        for j in range(nr_new_images):
            image = it.next()
            
            # Save the generated image in an array
            if X_aug.size == 0: # If the array is still empty, create the first element (image) in the array
                X_aug = np.array(image)
            else:               # If the array already exists, add the new image to the array
                X_aug = np.concatenate((X_aug, image), axis = 0)
            
            # Save the corresponding y of that image in an array
            if train_segmentations_aug.size == 0:
                train_segmentations_aug = np.array(y_val)
            else: 
                train_segmentations_aug = np.concatenate((train_segmentations_aug, y_val), axis = 0)
       
    # Concatenate the real images and the augmentation images
    X_aug_tot = np.concatenate((X_aug,images))
    train_segmentations_aug_tot = np.concatenate((segmentations,train_segmentations_aug))
    
    return X_aug, train_segmentations_aug, X_aug_tot, train_segmentations_aug_tot

def bspline(image):
    disp_i = np.random.rand(3, 3) # Make a random 2D 3 x 3 grid
    disp_i -= 0.5 # Move the displacements to the -0.5 to 0.5 grid
    disp_i /= 10
    
    disp_j = np.random.rand(3, 3) # Make a random 2D 3 x 3 grid
    disp_j -= 0.5 # Move the displacements to the -0.5 to 0.5 grid
    disp_j /= 10 

    # Define a B-spline transformation object
    bspline_grid = gryds.BSplineTransformation([disp_i, disp_j])
    
    # Define an interpolator object for the image:
    interpolator = gryds.MultiChannelInterpolator(image, order=0, cval=[.1, .2, .3])
    
    # Transform the image using the B-spline transformation
    transformed_image = interpolator.transform(bspline_grid)
    
    return transformed_image

def show_data_aug_bspline_effect(images):
    """
    Function that generates example images with brightness and b spline geometric augmentation
    :param images: data array, containing data of the training images
    :returns nothing but shows 3 by 3 subplots with random brightness augmentation
    """

    # load the image and convert to numpy array
    img = images[0]
    data = img_to_array(img)
    
    # expand dimension to one sample
    samples = np.expand_dims(data, 0)
    
    # create image data augmentation generator and prepare iterator
    datagen = ImageDataGenerator(brightness_range=[0.4,1.3])
    it = datagen.flow(samples, batch_size=1)
    
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')    
        #print(image)
        image_t = bspline(image)
        image_t = image_t.astype('uint8')
        # plot raw pixel data
        pyplot.imshow(image_t)
    # show the figure
    pyplot.show()

def brightness_bspline_augmentation_data(images,segmentations,nr_new_images = 1,range_brightness=[0.4,1.3]):
    """
    Function that generates new images with brightness augmentation
    :param images: numpy array, containing input data of the training images
    :param segmentations: numpy array, containing segmentation data of the training images
    :param nr_new_images: integer, determines the number of generated images per one image, default is 2
    :param range_brightness: list, containing the minimum and maximum brightness values, default is [0.4,1.3]
    
    :return train_X_aug: numpy array, containing augmented data set
    :return train_segmentations_aug: numpy array, containing segmentations augmented data set
    :return train_X_tot: numpy array, containing complete data set (real + augmented)
    :return train_segmentations_aug_tot: numpy array, containing segmentations complete data set
    """
    
    datagen = ImageDataGenerator(brightness_range = range_brightness)
    X_aug = np.array([]) # array in which the new, generated images can be stored
    train_segmentations_aug = np.array([]) # array in which to store the 

    for i in range(len(images)): # For every image in the training data set, generate new images for the augmented dataset
        train_im = images[i]
        image_uint8 = train_im.astype('uint8')
        image_t = bspline(image_uint8)
        #image_t = image_t.astype('uint8')
        
        y_val = segmentations[i]
        y_val = np.expand_dims(y_val, 0)
        
        #data = img_to_array(train_im)
        data = img_to_array(image_t)
        
        samples = np.expand_dims(data, 0)
        it = datagen.flow(samples, batch_size = nr_new_images) 
        for j in range(nr_new_images):
            image = it.next()

            # Save the generated image in an array
            if X_aug.size == 0: # If the array is still empty, create the first element (image) in the array
                X_aug = np.array(image)
            else:               # If the array already exists, add the new image to the array
                X_aug = np.concatenate((X_aug, image), axis = 0)
            
            # Save the corresponding y of that image in an array
            if train_segmentations_aug.size == 0:
                train_segmentations_aug = np.array(y_val)
            else: 
                train_segmentations_aug = np.concatenate((train_segmentations_aug, y_val), axis = 0)
       
    # Concatenate the real images and the augmentation images
    X_aug_tot = np.concatenate((X_aug, images))
    train_segmentations_aug_tot = np.concatenate((segmentations,train_segmentations_aug))
    
    return X_aug, train_segmentations_aug, X_aug_tot, train_segmentations_aug_tot
