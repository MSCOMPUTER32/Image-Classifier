import argparse  
import json  
import torch 
import torch.nn as nn
from PIL import Image  
import numpy as np 
from matplotlib import pyplot as plt
from torchvision import transforms, models

def load_model(checkpoint_path):
    """load a model from a checkpoint file"""
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']

    if architecture == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 2048),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 102),  
        )
    elif architecture == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 102),
        )
    else:
        raise ValueError("Unsupported architecture. Choose either 'vgg16' or 'resnet50'.")

    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model

def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns an Numpy array
    '''

    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    checkpoint_path = 'best_model_vgg16.pth' 
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    class_to_idx = checkpoint['class_to_idx']

    image_tensor = process_image(image_path)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)


    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_labels = torch.topk(probabilities, topk)

        top_probs = top_probs.cpu().numpy().tolist()
        top_labels = top_labels.cpu().numpy().tolist()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]

    return top_probs, top_labels

def display_prediction(image_path, probs, labels, cat_to_name):
    ''' Display an image along with the top K classes
    '''
    image = Image.open(image_path)


    flower_names = [cat_to_name[str(label)] for label in labels]

    fig, (ax1, ax2) = plt.subplots(figsize=(8,8), ncols=1, nrows=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax2.barh(np.arange(len(probs)), probs, align='center')
    ax2.set_yticks(np.arange(len(probs)))
    ax2.set_yticklabels(flower_names)  
    ax2.set_xlabel('Probabilities')
    ax2.set_title('Class Probability')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image', type=str, help='path to the image')
    parser.add_argument('checkpoint', type=str, help='path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the file with flower names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')

    args = parser.parse_args()

    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
        model = load_model(args.checkpoint)
        model.to(device)
        print(f"Using device: {device}")
    except RuntimeError as e:
        print("Error: GPU is not available. Using CPU instead.")
        device = torch.device('cpu')
        model = load_model(args.checkpoint)
        model.to(device)
    
    probs, labels = predict(args.image, model, args.top_k)

    classes = [str(label) for label in labels]
    flower_names = [cat_to_name.get(class_index, "Unknown") for class_index in classes] if cat_to_name else classes

    print ("Probabilities: ", probs)
    print ("Flower names: ", flower_names)

    if cat_to_name:
        display_prediction(args.image, probs, labels, cat_to_name)

if __name__ == '__main__':
    main()