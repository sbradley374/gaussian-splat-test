import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from read_write_model import *
from borrowed import *
from gsmodel import *



torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path of dataset")
    args = parser.parse_args()

    #in their code they use GSplatDataset but i don't think we need that
    gs = read_points_bin_as_gau(Path(args.path, "sparse/0/points3D.bin"))

    training_params, adam_params = get_training_params(gs) 
    cameras, images = get_cameras_and_images(args.path)
    #points = needs to be the actual points
     
    # path = "/home/liu/bag/gaussian-splatting/tandt/train"
    # gs_set = GSplatDataset(path, resize_rate=1)
    # gs = np.load("data/final.npy")

    #we should do it a different way than this but we do need to get the paramaters
    

    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15) #we can fine tune these

    epochs = 20 #we can fine tune this
    n = len(images)

    twcs = torch.stack([x.twc for x in cameras])
    cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
    sence_size = float(torch.max(cam_dist)) * 1.1

    model = GSModel(sence_size, len(images) * epochs)

    for epoch in range(epochs):
        #randomly shuffly indices for which photo we are looking at
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs: #go through each photo (like SGD with batch size = 1)
            cam = cameras[i]
            image_gt = images[i]

            image = model(*training_params.values(), cam) #get the image generated by rasterizing current splats onto place where image is taken
            loss = gau_loss(image, image_gt) 

            
            loss.backward()  

            model.update_density_info() 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            model.update_pws_lr(optimizer)
            avg_loss += loss.item()

        #not necessary but useful print statement:
        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))

        #check if its time to do the special things (we can play arround with these and how they are done) 
        with torch.no_grad():
            if (epoch > 1 and epoch <= 50):
                if (epoch % 5 == 0):
                    print("updating gaussian density...")
                    model.update_gaussian_density(training_params, optimizer)
                if (epoch % 15 == 0):
                    print("reseting gaussian aplha...")
                    model.reset_alpha(training_params, optimizer)

    #this doesn't need to be a function in our implementation:
    save_training_params('data/final.ply', training_params)
 

