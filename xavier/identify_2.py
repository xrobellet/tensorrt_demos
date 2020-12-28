#SETUP

from facenet_pytorch import MTCNN, InceptionResnetV1
import operator
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle
import math
from unidecode import unidecode
from PIL import Image
from shutil import copyfile
from time import perf_counter 

workers = 0 if os.name == 'nt' else 4
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

#some useful functions

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


def dispatch_pictures(X_img_path, mtcnn, resnet, distance_min=0.3, distance_threshold=0.6, k=3, num=0):
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


    :return: create different folder correponding to the name it recognized and copy the pictures into it:
      <person1>/
      │   ├── <person1>_num.jpg
      │   ├── <person1>_num.jpg
      │   ├── ...
      ├── <person2>/
      │   ├── <person2>_num.jpg
      │   └── <person2>_num.jpg
    """

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    t1_start = perf_counter()
    img = Image.open(X_img_path) 


    # Load image file and find face locations
    x_aligned = mtcnn(img)
    #print(x_aligned)

    if x_aligned is not None:
        print("num_faces_detect: ", len(x_aligned))

        result = x_aligned.to(device)
        #print(result)
        predict = resnet(result).detach().cpu()
        #print(predict)

        # See if the face is a match for the known face(s)

        for face in predict:
            #print(face)
            neighbors = getKNeighbors(embeddings_db, names, face, k)
            predi, are_matches = getResponse(neighbors, distance_threshold, distance_min)

            name = "Unknown"

            if are_matches:
                name = predi      

                if os.path.isdir("./test_blog/"+ name.replace(" ", "_")):
                    copyfile (X_img_path, "./test_blog/"+ name.replace(" ", "_")+"/"+name.replace(" ", "_")+"_"+ str(num) +".jpg")
                    num= num + 1
                else:
                    os.mkdir("./test_blog/"+name.replace(" ", "_"))
                    copyfile (X_img_path, "./test_blog/"+ name.replace(" ", "_")+"/"+name.replace(" ", "_")+"_"+ str(num) +".jpg")
                    num = num + 1
    t1_end = perf_counter()


    return print("done", X_img_path, " Elapsed time: ", t1_end-t1_start)

def identify (X_img_path, mtcnn, resnet, distance_min=0.3, distance_threshold=0.6, k=3, num=0):
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
    img = Image.open(X_img_path)
    image_size = img.size
    width = image_size[0]
    height = image_size[1]
    resized_im = img.resize((round(img.size[0]*0.8), round(img.size[1]*0.8)))
    t2 = perf_counter()
    # Load image file and find face locations
    x_aligned = mtcnn(resized_im)
    #print(x_aligned)
    t3 = perf_counter()
    if x_aligned is not None:
        print("num_faces_detect: ", len(x_aligned))

        result = x_aligned.to(device)
        #print(result)
        t4 = perf_counter()
        predict = resnet(result).detach().cpu()
        t5 = perf_counter()
        #print(predict)

        # See if the face is a match for the known face(s)
        print ('num_faces: ', len(predict))

        for face in predict:
            #print(face)
            neighbors = getKNeighbors(embeddings_db, names, face, k)
            predi, are_matches = getResponse(neighbors, distance_threshold, distance_min)

            name = "Unknown"

            if are_matches:
                name = unidecode(predi)      

            print(name)
    t6 = perf_counter()
    return print(f"done, {X_img_path}  Elapsed time_total:  {t6-t1_start} time image {t2-t1_start} timetodevice {t4-t3} time mtcnn {t3-t2} time resnet {t5-t3}")

def mtcnn_split(X_img_path, mtcnn):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    t1_start = perf_counter()
    img = Image.open(X_img_path)
    image_size = img.size
    width = image_size[0]
    height = image_size[1]
    resized_im = img.resize((round(img.size[0]*1), round(img.size[1]*1)))
    # Load image file and find face locations
    x_aligned = mtcnn(resized_im)
    t2 = perf_counter()
    print(f"time mtcnn {t2-t1_start}")
    return x_aligned

def resnet_split(x_aligned, resnet, distance_min=0.3, distance_threshold=0.6, k=3, device=None):
    t3 = perf_counter()
    if x_aligned is not None:
        print("num_faces_detect: ", len(x_aligned))

        result = x_aligned.to(device)
        #print(result)
        predict = resnet(result).detach().cpu()
        t5 = perf_counter()
        #print(predict)

        # See if the face is a match for the known face(s)
        print ('num_faces: ', len(predict))

        for face in predict:
            #print(face)
            neighbors = getKNeighbors(embeddings_db, names, face, k)
            predi, are_matches = getResponse(neighbors, distance_threshold, distance_min)

            name = "Unknown"

            if are_matches:
                name = unidecode(predi)      

            print(name)
    t6 = perf_counter()
    return print(f"time resnet+classification = {t6-t3}")

#Loading database and name

with open('./2020_12_17_tensors_faces_db_embeedings.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

with open('./2020_12_17_Tensorsnames_residents.pkl', 'rb') as f:
    names = pickle.load(f)

with open('./face_list.pkl', 'rb') as f:
    face_list = pickle.load(f)

#define models and gpu device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#define the model:
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
for faces in face_list:
    resnet_split (faces, resnet, distance_min=0.3, distance_threshold=0.7, k=3, device=device)

#resnet_split (faces, resnet, distance_min=0.3, distance_threshold=0.7, k=3, device=device)

#num=0
#for file in os.listdir('./pictures_blog_update/'):
#    full_path = './pictures_blog_update/' + file
#    identify(full_path, mtcnn, resnet, distance_min=0.3, distance_threshold=0.7, k=3, num=num)
#print('finish')
