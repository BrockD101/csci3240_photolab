#!/usr/bin/env python3
import argparse
import cv2
import os
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
image_path = args.image

for f in os.listdir('src/models'):
    if f.endswith('.pth'):
        deepfill_weights_path = os.path.join('src/models', f)
print("Creating rcnn model")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
rcnn = rcnn.eval()

print('Creating deepfill model')
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

model = ObjectRemove(segmentModel=rcnn,
                        rcnn_transforms=transforms, 
                        inpaintModel=deepfill, 
                        image_path=image_path )

output = model.run()

def apply_transformations(temp_val, gray_val, avg_val, gauss_val, edge_val):
    global inpainted_image

    # Temperature adjustment (increase brightness)
    temp_adjusted = cv2.convertScaleAbs(inpainted_image, alpha=1.0, beta=temp_val)  # Beta acts as temperature control
    
    # Grayscale Conversion (if enabled)
    if gray_val:
        processed_image = cv2.cvtColor(temp_adjusted, cv2.COLOR_BGR2GRAY)
    else:
        processed_image = temp_adjusted
    
    # Apply Averaging Filter
    avg_filtered = cv2.blur(processed_image, (avg_val, avg_val)) if avg_val > 0 else processed_image

    # Ensure Gaussian kernel size is odd
    gauss_val = gauss_val if gauss_val % 2 == 1 else gauss_val + 1
    
    # Apply Gaussian Filter
    gauss_filtered = cv2.GaussianBlur(avg_filtered, (gauss_val, gauss_val), 0) if gauss_val > 0 else avg_filtered

    # Edge Detection using Canny
    edges = cv2.Canny(gauss_filtered, edge_val, edge_val) if edge_val > 0 else gauss_filtered
    
    return edges, temp_adjusted

def update_image(val):
    global processed_image

    # Extract slider values
    temp_val = cv2.getTrackbarPos('Temperature', 'Inpainted Image')
    gray_val = cv2.getTrackbarPos('Grayscale', 'Inpainted Image')
    avg_val = cv2.getTrackbarPos('Avg Filter Size', 'Inpainted Image')
    gauss_val = cv2.getTrackbarPos('Gaussian Filter Size', 'Inpainted Image')
    edge_val = cv2.getTrackbarPos('Edge Threshold', 'Inpainted Image')
    
    # Apply transformations and update the processed image
    edges, temp_adjusted = apply_transformations(temp_val, gray_val, avg_val, gauss_val, edge_val)
    processed_image = edges  

    if edges is not None:
        combined = np.hstack((inpainted_image, cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)))
        cv2.imshow('Inpainted Image', combined)

def save_image(filename="final_inpainted.jpg"):
    """ Save the processed (right-side) image to the file system """
    cv2.imwrite(filename, processed_image)
    print(f"Image saved as {filename}")

def main():
    global inpainted_image
    inpainted_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) 

    cv2.namedWindow('Inpainted Image', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Temperature', 'Inpainted Image', 30, 100, update_image)
    cv2.createTrackbar('Grayscale', 'Inpainted Image', 0, 1, update_image)
    cv2.createTrackbar('Avg Filter Size', 'Inpainted Image', 5, 15, update_image)
    cv2.createTrackbar('Gaussian Filter Size', 'Inpainted Image', 5, 15, update_image)
    cv2.createTrackbar('Edge Threshold', 'Inpainted Image', 100, 200, update_image)

    cv2.imshow('Inpainted Image', inpainted_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' is pressed exit the loop
        if key == ord('q'):
            break
        # If 'c' is pressed save the final image 
        elif key == ord('c'):
            # Ensure edges are calculated before saving
            save_image(filename="final_inpainted.jpg")
        
        update_image(None)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
