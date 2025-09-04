import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def saliency(model, image_path):
    """Computes the saliency map of the predicted class as a feature attribution explanation.
    From Simonyan et al. (2013)

    :param model: predictive model
    :param image_path: path to the image to be explained
    :returns: saliency map of input image shape
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Load and process image
    image = Image.open(image_path)
    image = image.convert("RGB") 
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor.requires_grad = True
    output = model(image_tensor)
    c =  torch.argmax(output)
    score = output[0, c]
    score.backward()

    saliency = image_tensor.grad.abs().squeeze().max(0)[0].cpu()

    return saliency.squeeze()



def show_attribution_overlay(image_path, attribution_map, title):
    """Visualization of an attribution explanation overlayed to the image it's explaining

    :param image_path: path to the image to be explained
    :param attribution_map: explanation of size (224,224)
    :param title: title string
    """
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    x = np.arange(0., 100.0, dx)
    y = np.arange(0., 100.0, dy)
    X, Y = np.meshgrid(x, y)
    
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    
    #img, label = dataset[sample_idx]
    img = Image.open(image_path)
    img = img.convert("RGB")

    #just for displaying, without normalization
    unnormalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
        std = [ 1., 1., 1. ]),
        ])

    img = unnormalize_transform(img)
    img = img.permute(1,2,0) # RGB channel to the back
    im1 = plt.imshow(img, interpolation='nearest',
                 extent=extent)

    im2 = plt.imshow(attribution_map, cmap=plt.cm.viridis, alpha=.8, interpolation='bilinear',
                    extent=extent)
    plt.axis('off')

def preprocessing(data_scoring, target_column='loan_status', test_size=0.2, random_state=12):

    # Create a copy to avoid modifying original data
    data = data_scoring.copy()
    
    # Step 1: Handle missing values
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)  # Fixed: drop=True to avoid creating 'index' column
    # Step 2: Remove outliers
    # Remove people over 80 years old
    initial_count = len(data)
    data = data.drop(data[data['person_age'] > 80].index, axis=0)
    data.reset_index(drop=True, inplace=True)
    
    # Additional outlier removal
    # Remove unrealistic income values (less than $1000 or greater than $1M)
    data = data[(data['person_income'] >= 1000) & (data['person_income'] <= 1000000)]
    
    # Remove unrealistic loan amounts
    data = data[(data['loan_amnt'] >= 500) & (data['loan_amnt'] <= 100000)]
    
    # Remove unrealistic interest rates
    data = data[(data['loan_int_rate'] >= 0) & (data['loan_int_rate'] <= 35)]
    
    data.reset_index(drop=True, inplace=True)

    
    # Create loan-to-income ratio (with safety check)
    #data['loan_to_income_ratio'] = data['loan_amnt'] / data['person_income']
    
    # Fixed employment length ratio (more meaningful calculation)
    # Employment length per $1000 of loan amount
    #data['emp_length_per_loan_1k'] = data['person_emp_length'] / (data['loan_amnt'] / 1000)
    
    # Interest rate per $1000 of loan amount
    #data['int_rate_per_loan_1k'] = data['loan_int_rate'] / (data['loan_amnt'] / 1000)
    
    # Additional useful features
    #data['income_to_age_ratio'] = data['person_income'] / data['person_age']
    #data['loan_to_emp_length'] = data['loan_amnt'] / (data['person_emp_length'] + 1)  # +1 to avoid division by zero

    
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    print(f"Target distribution (%):\n{y.value_counts(normalize=True) * 100}")
    
    # Step 5: Train-test split (BEFORE any encoding to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test