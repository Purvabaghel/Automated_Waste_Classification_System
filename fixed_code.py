import os
import numpy as np
import tensorflow as tf
import shutil
import random
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 224  # MobileNet input size
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 14  # Updated to match your actual dataset classes
LEARNING_RATE = 0.0001
TEST_SPLIT = 0.2  # 20% of data will be used for testing
RANDOM_SEED = 42  # For reproducibility

# Path to your dataset and where to save train/test splits
# Make sure this is the actual path to the directory containing your class folders
# For example, if your structure is C:/Users/HP/Documents/saket/garbage_dataset/plastic, etc.
# then ORIGINAL_DATASET_PATH should be "C:/Users/HP/Documents/saket/garbage_dataset"
ORIGINAL_DATASET_PATH = "C:/Users/HP/Documents/saket"  # Adjust this path!

# Other paths
PROCESSED_DATASET_PATH = "processed_dataset"  # Will create train and test dirs here
MODEL_SAVE_PATH = "model"  # Directory to save the trained model

# List of class names (this should match your actual class names)
# You will need to update this with your actual 14 class names
class_names = ["plastic", "metal", "paper", "cardboard", "glass", "organic", 
               "battery", "clothes", "shoes", "electronic", "biological", "other",
               "class13"]  # Add your two additional class names here

def create_train_test_split():
    """
    Creates train/test splits from the original dataset structure.
    Assumes the original structure has class folders directly:
    - dataset/
      - plastic/
      - metal/
      - etc.
    """
    # Create processed dataset directories if they don't exist
    os.makedirs(os.path.join(PROCESSED_DATASET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATASET_PATH, "test"), exist_ok=True)
    
    # Print directory contents to debug
    print(f"Contents of {ORIGINAL_DATASET_PATH}: {os.listdir(ORIGINAL_DATASET_PATH)}")
    
    all_files_count = 0
    
    # Process each class
    for class_name in os.listdir(ORIGINAL_DATASET_PATH):
        class_dir = os.path.join(ORIGINAL_DATASET_PATH, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            print(f"Skipping {class_name} as it's not a directory")
            continue
        
        # Create corresponding class folders in train and test
        os.makedirs(os.path.join(PROCESSED_DATASET_PATH, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATASET_PATH, "test", class_name), exist_ok=True)
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                       if os.path.isfile(os.path.join(class_dir, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        print(f"Found {len(image_files)} images in {class_name}")
        all_files_count += len(image_files)
        
        # Only proceed with split if there are files
        if len(image_files) > 0:
            # Split into train and test
            train_files, test_files = train_test_split(
                image_files, test_size=TEST_SPLIT, random_state=RANDOM_SEED
            )
            
            # Copy files to train folder
            for file in train_files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(PROCESSED_DATASET_PATH, "train", class_name, file)
                shutil.copy(src, dst)
            
            # Copy files to test folder
            for file in test_files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(PROCESSED_DATASET_PATH, "test", class_name, file)
                shutil.copy(src, dst)
            
            print(f"Processed {class_name}: {len(train_files)} train, {len(test_files)} test")
        else:
            print(f"Warning: No image files found in {class_dir}")
    
    print(f"Total files processed: {all_files_count}")
    
    if all_files_count == 0:
        raise ValueError("No image files found in any class directory. Please check your dataset path.")
        
    return PROCESSED_DATASET_PATH

def create_model():
    """Create and compile a MobileNetV2 model fine-tuned for garbage classification."""
    # Load the MobileNetV2 model without the top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators(dataset_path):
    """Prepare data generators for training and validation."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, test_generator

def train_model():
    """Train the MobileNetV2 model on the garbage classification dataset."""
    # First create the train/test split
    dataset_path = create_train_test_split()
    
    # Create directories for saving the model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Create the model
    model = create_model()
    
    # Prepare data generators
    train_generator, test_generator = prepare_data_generators(dataset_path)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model and class indices
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.h5'))
    
    # Save class indices for later use in the web interface
    class_indices = train_generator.class_indices
    np.save(os.path.join(MODEL_SAVE_PATH, 'class_indices.npy'), class_indices)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return model, class_indices, history

def unfreeze_and_finetune(model, dataset_path):
    """Unfreeze some of the top layers and continue training with lower learning rate."""
    # Unfreeze the top layers of the model
    for layer in model.layers[-30:]:  # Unfreeze the last 30 layers
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare data generators again
    train_generator, test_generator = prepare_data_generators(dataset_path)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'best_finetuned_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model with unfrozen layers
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,  # Fewer epochs for fine-tuning
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the fine-tuned model
    model.save(os.path.join(MODEL_SAVE_PATH, 'finetuned_model.h5'))
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Fine-tuned test accuracy: {test_accuracy:.4f}")
    
    return model, history

def create_model_for_web():
    """
    Create a version of the model suitable for TensorFlow.js conversion.
    This function prepares the model with the preprocessing built in.
    """
    # Load the saved model
    model = load_model(os.path.join(MODEL_SAVE_PATH, 'finetuned_model.h5'))
    
    # Load class indices
    class_indices = np.load(os.path.join(MODEL_SAVE_PATH, 'class_indices.npy'), allow_pickle=True).item()
    
    # Reverse the class indices to map from index to class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Save the mapping for use in the web interface
    with open(os.path.join(MODEL_SAVE_PATH, 'class_mapping.txt'), 'w') as f:
        for idx, class_name in idx_to_class.items():
            f.write(f"{idx},{class_name}\n")
    
    # Save the model in TensorFlow.js format (requires tensorflowjs package)
    # Install with: pip install tensorflowjs
    try:
        import tensorflowjs as tfjs
        tfjs_output_dir = os.path.join(MODEL_SAVE_PATH, 'tfjs_model')
        os.makedirs(tfjs_output_dir, exist_ok=True)
        tfjs.converters.save_keras_model(model, tfjs_output_dir)
        print(f"Model saved for TensorFlow.js in {tfjs_output_dir}")
    except ImportError:
        print("TensorFlow.js not installed. Install with: pip install tensorflowjs")
        print("Skipping TensorFlow.js model conversion.")

if __name__ == "__main__":
    # Train the model
    model, class_indices, history = train_model()
    
    # Fine-tune the model
    model, ft_history = unfreeze_and_finetune(model, PROCESSED_DATASET_PATH)
    
    # Prepare model for web deployment
    create_model_for_web()