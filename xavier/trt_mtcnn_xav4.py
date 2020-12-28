"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""
from facenet_pytorch import InceptionResnetV1
import operator
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle
import math
from time import perf_counter
from shutil import copyfile
from time import perf_counter
from torch2trt import TRTModule

workers = 0 if os.name == 'nt' else 4
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
import time
import argparse
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms

import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn


WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        for j in range(5):
            cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return img


def loop_and_detect(cam, mtcnn, minsize):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            img = show_faces(img, dets, landmarks)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)



def getKNeighbors(X, y, image_to_test, k):
    """
    
    This function calculate the distance between an encoding face test_image and the encoding face database. 
    Then it select the k closest encoding face. 
    
    :param X: list of all the encoding face database (output of facetensordatabase() function)
    :param y: list of the name corresponding to these encoding face (output of facetensordatabase() function)
    :param image_to_test: encoding face to test against the database
    :param k: numbers of closest neighbours we want to use for the KNN
    
    :return: returns 1 variables: A list containing tuple organised as follow: (face_encoding tensor, distance, name)
    
    """
    X=torch.stack(X)
    #print(image_to_test.shape)
    dist=((image_to_test - X)**2)
    dist=dist.sum(axis=1)
    dist=torch.sqrt(dist)
    
    value, indices = torch.sort(dist)
    close_neighbors=[]
    for x in range (k):
        close_neighbors.append((X[indices[x]], value[x], y[indices[x]]))
    return close_neighbors

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    face = F.to_tensor(np.float32(face))

    return face

def extract(img, batch_boxes):
        # Determine if a batch or single image was passed
        batch_mode = True
        if (
                not isinstance(img, (list, tuple)) and
                not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
                not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Process all bounding boxes
        faces = []
        for im, box_im in zip(img, batch_boxes):
            if box_im is None:
                faces.append(None)
                continue

            faces_im = []
            for i, box in enumerate(box_im):

                face = extract_face(im, box, image_size=160, margin=0)
                face = fixed_image_standardization(face)
                faces_im.append(face)
            if len(faces_im)==0:
                return None
            else:
                faces_im = torch.stack(faces_im)

                faces.append(faces_im)

        if not batch_mode:
            faces = faces[0]

        return faces


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def getResponse(close_neighbors, distance_threshold, distance_min,):
    """
    
    This function is used to classify the unknown encoding face to a corresponding member of the database when the distance is 
    close enough.
    
    :param X: output of getKNeighbors() function
    :param distance_threshold: if the distance between the unknown encoding face and the closest neighbour is >0.6 than 
                      the unknown encoding face is classify as unknown
                      
    :param distance_min: if the distance between the unknown encoding face and the closest neighbour is <0.2 than 
                      the unknown encoding face is directly classify as this person
                      
    :return: returns 2 variables: * sorted_Vo: A list containing the name getting the majority of vote.
                                  * sorted_Vo: A list of Bol saying if the is a match or not with database
                                              True=yes
                                              False=No
    
    """
    classVotes = {}
    classVotes["Unknown Person"] = 0
    if distance_min > close_neighbors[0][1]:
        are_matches = True        
        return close_neighbors[0][2], are_matches
    else:  
        for x in close_neighbors:
            if distance_min <= x[1] <= distance_threshold:
                if x[2] in classVotes:
                    classVotes[x[2]] += 1
                else:
                    classVotes[x[2]] = 1
            else:
                classVotes["Unknown Person"] +=1

        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        if len(sortedVotes)>1:
            #print(sortedVotes)
            are_matches = (sortedVotes[0][0] != "Unknown Person") and (sortedVotes[0][1] > sortedVotes[1][1])
            #print(are_matches)
        else:
            #print(sortedVotes)
            are_matches = (sortedVotes[0][0] != "Unknown Person")
            #print(are_matches)
        return sortedVotes[0][0], are_matches

def identify (X_img_path, mtcnn, resnet, distance_min=0.3, embeddings_db=None, names=None, distance_threshold=0.7, k=3, num=0, device = None):
    """
    Recognizes faces in given image using a KNN classifier and the dispatch them to file corresponding to the faces it recognized

    :param X_img_path: path to image to be recognized

    :param mtcnn: mtcnn for faces identification in the picture

    :param resnet: model to get the encode face embeddings
    
    :param distance_min: if the distance between the unknown encoding face and the closest neighbour is <0.2 than 
                      the unknown encoding face is directly classify as this person

    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
        of mis-classifying an unknown person as a known one.

    :param k: numbers of closest neighbours we want to use for the KNN

    :param num: help to differentiate the pictures_names


    :return: name identify
    """

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    t1_start = perf_counter()
    img = cv2.imread(X_img_path)
    
    # Load image file and find face locations
    dets, landmarks = mtcnn.detect(img, minsize=40)
    t2 = perf_counter()
    x_aligned = extract(img, dets)
    t3 = perf_counter()
    #print(x_aligned)

    if x_aligned is not None:
        print("num_faces_detect: ", len(x_aligned))
        result = x_aligned.to(device)
        #print(result)
        predict = resnet(result).detach().cpu()
        #print(predict)
        t5 = perf_counter()
        # See if the face is a match for the known face(s)
        print ('num_faces: ', len(predict))
        for face in predict:
            #print(face)
            neighbors = getKNeighbors(embeddings_db, names, face, k)
            predi, are_matches = getResponse(neighbors, distance_threshold, distance_min)
            t7 = perf_counter()
            name = "Unknown"

            if are_matches:
                name = predi     

            print(name)
    t1_end = perf_counter()
    return print(f"done {X_img_path} Elapsed total time: {t1_end-t1_start}")#" mtccn: {t2-t1_start} getfaces: {t3-t2} resnet: {t5-t3} predi: {t7-t5}")

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def main():
    args = parse_args()
    #cam = Camera(args)
    #if not cam.isOpened():
    #    raise SystemExit('ERROR: failed to open camera!')

    #define models and gpu device
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = TrtMtcnn()
    resnet = TRTModule()

    resnet.load_state_dict(torch.load('/resnet.trt'))

    #Loading database and name
    with open('/facenet_pytorch/data/2020_12_17_tensors_faces_db_embeedings.pkl', 'rb') as f:
        embeddings_db = pickle.load(f)

    with open('/facenet_pytorch/data/2020_12_17_Tensorsnames_residents.pkl', 'rb') as f:
        names = pickle.load(f)

    for file in os.listdir('/facenet_pytorch/data/pictures_blog_update/'):
        full_path = '/facenet_pytorch/data/pictures_blog_update/' + file
        identify (full_path, mtcnn, resnet, embeddings_db=embeddings_db, names=names, distance_min=0.3, distance_threshold=0.7, k=3, num=0, device=device)

    #open_window(
    #    WINDOW_NAME, 'Camera TensorRT MTCNN Demo for Jetson Nano',
    #    cam.img_width, cam.img_height)
    #loop_and_detect(cam, mtcnn, args.minsize)

    #cam.release()
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
