import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
# Load the DenseNet model
def load_model():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    try:
        model.load_state_dict(torch.load('densenet_model10.pth', map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.to(device)
    model.eval()
    return model, device

# Process and predict each frame
def process_video(video_path, model, device):
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip = round(fps / target_fps)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Skip frames to achieve the target FPS
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        # Preprocess the frame
        input_tensor = transform(frame).unsqueeze(0).to(device)
        # Predict using the model
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            accident = 'No Accident' if predicted.item() == 1 else 'Accident'
        if accident == 'Accident':
            print(f'Frame {frame_count} - {accident}')
        frame_count += 1
    cap.release()

# Main function to load the model and process the video
if __name__ == "__main__":
    model, device = load_model()
    video_path = 'car-crash.mov'
    process_video(video_path, model, device)
