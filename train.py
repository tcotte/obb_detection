import math
import os
import time

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.nn import CrossEntropyLoss, BCELoss
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from losses import OrientedGIoULoss
from object_detector import ObjectDetector
from torch_dataset import OrientedBoundingBoxes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 1
# specify the loss weights
LABELS = 1.0
BBOX = 0.6
PROB = 0.4
UNFROZEN_LAYERS = 5
PRETRAINED_BACKBONE = True
# IMGSZ = (128, 128) # height, width
# IMGSZ = args.imgsz # height, width

def collate_fn(batch):
    return tuple(zip(*batch))


CLASSES = ["Rec"]
bbox_format = 'albumentations'

list_train_transformation = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(always_apply=True),
        ToTensorV2()]


train_transform = A.Compose(
    list_train_transformation
)

test_transform = A.Compose([
    A.Normalize(always_apply=True),
    ToTensorV2()],
)

train_dataset = OrientedBoundingBoxes(image_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds0\img",
                          mask_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds0\ann",
                          classes=["rec"],
                          transforms=train_transform)

val_dataset = OrientedBoundingBoxes(image_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds1\img",
                          mask_paths=r"C:\Users\tristan_cotte\PycharmProjects\InstanceSegmentationOBB\dataset\ds1\ann",
                          classes=["rec"],
                          transforms=test_transform)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=os.cpu_count(), pin_memory=PIN_MEMORY, collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(),
                                                pin_memory=PIN_MEMORY, collate_fn=collate_fn)


if __name__ == "__main__":

    print("[INFO] total training samples: {}...".format(len(train_dataset)))
    print("[INFO] total test samples: {}...".format(len(val_dataset)))
    # calculate steps per epoch for training and validation set
    train_steps = math.ceil(len(train_dataset) / BATCH_SIZE)
    val_steps = math.ceil(len(val_dataset) / BATCH_SIZE)
    # create data loaders
    # trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                          shuffle=True, num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)
    # testLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
    #                         num_workers=os.cpu_count(), pin_memory=PIN_MEMORY)

    # Network
    # load the ResNet network
    if PRETRAINED_BACKBONE:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    backbone = resnet18(weights=weights)
    # freeze some ResNet layers, so they will *not* be updated during the training process
    params = backbone.state_dict()
    list_layers = list(params.keys())
    for name, param in backbone.named_parameters():
        if name in list_layers[-UNFROZEN_LAYERS:]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # create our custom object detector model and flash it to the current
    # device
    objectDetector = ObjectDetector(backbone, len(CLASSES), confidence=0.5)

    # else:
    #     objectDetector = torch.load(args.weights, map_location=torch.device('cpu'))


    objectDetector = objectDetector.to(DEVICE)
    # define our loss functions

    classLossFunc = CrossEntropyLoss()
    probLossFunc = BCELoss()

    bboxLossFunc = OrientedGIoULoss()
    # initialize the optimizer, compile the model, and show the model
    # summary
    opt = optim.Adam(objectDetector.parameters(), lr=INIT_LR)

    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
         "val_class_acc": [], "train_iou": [], "val_iou": []}

    # if wb_visu:
    #     w_b = WeightandBiaises(project_name="circle_detection", run_id=model_name, interval_display=50, cfg={
    #         "epochs": NUM_EPOCHS,
    #         "batch_size": BATCH_SIZE,
    #         "learning_rate": INIT_LR,
    #         "optimizer": repr(opt).split(" ")[0],
    #         "unfrozen_layers": UNFROZEN_LAYERS,
    #         "backbone_architecture": repr(backbone).split("(")[0],
    #         "pretrained_bacbone": PRETRAINED_BACKBONE,
    #         "dataset": "colored_circles",
    #         "weight_loss_bbox_regression": BBOX,
    #         "weight_loss_prob_x": PROB,
    #         "image_size": IMGSZ,
    #         "translation_aug": args.translation
    #     })
    # else:
    #     w_b = None

    # loop over epochs

    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # set the model in training mode
        objectDetector.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        bbox_train_loss = 0
        bbox_test_loss = 0

        obj_train_loss = 0
        obj_test_loss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        obj_train_acc = 0
        obj_test_acc = 0

        train_iou = 0
        val_iou = 0

        # 'image': image, 'masks': torch.stack(masks), "obb_point_based": torch.Tensor(obb).type(torch.uint16),
        # "obb_angle_based": torch.Tensor((*center, *lengths, angle)).type(torch.float32)

        # loop over the training set
        for (images, labels, masks, _, obb_angle_based) in training_loader:
            # send the input to the device
            # labels = torch.Tensor(labels)
            # bboxes = torch.stack(bboxes, dim=1)

            # bboxes = bboxes.to(torch.float32)
            # bboxes = torch.squeeze(bboxes, 1)

            (images, labels, obb_angle_based) = (images.to(DEVICE),
                                        labels.to(DEVICE), obb_angle_based.to(DEVICE))


            # perform a forward pass and calculate the training loss
            opt.zero_grad()
            predictions = objectDetector(images)
            coord_pred = predictions[0]*probs
            bboxes *= probs
            bbox_loss = bboxLossFunc(coord_pred.float(), bboxes)
            objectness_loss = probLossFunc(predictions[2], probs.float())
            totalLoss = BBOX * bbox_loss + objectness_loss * PROB

            totalLoss = totalLoss.to(torch.float)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights

            totalLoss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            # train_iou += batch_iou(a=coord_pred.detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(bboxes)
            totalTrainLoss += totalLoss
            bbox_train_loss += bbox_loss
            obj_train_loss += objectness_loss
            obj_train_acc += torch.sum(torch.where(predictions[2] > 0.5, 1, 0) == probs).item() / len(bboxes)


        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            objectDetector.eval()
            # loop over the validation set
            for (images, labels, bboxes, probs, _) in validation_loader:
                # send the input to the device
                labels = torch.Tensor(labels)
                bboxes = torch.squeeze(bboxes, 1)

                # bboxes = torch.stack(bboxes, dim=1)
                (images, labels, bboxes, probs) = (images.to(DEVICE),
                                            labels.to(DEVICE), bboxes.to(DEVICE), probs.to(DEVICE))
                # make the predictions and calculate the validation loss
                predictions = objectDetector(images)

                coord_pred = predictions[0] * probs
                bboxes *= probs

                bbox_loss = bboxLossFunc(coord_pred.float(), bboxes.to(torch.float32))
                # classLoss = classLossFunc(predictions[1], labels)
                objectness_loss = probLossFunc(predictions[2], probs.float())
                totalLoss = BBOX * bbox_loss + objectness_loss * PROB
                totalValLoss += totalLoss
                bbox_test_loss += bbox_loss
                obj_test_loss += objectness_loss
                # calculate the number of correct predictions
                # val_iou += batch_iou(a=coord_pred.detach().cpu().numpy(), b=bboxes.cpu().numpy()).sum() / len(
                #     bboxes)
                obj_test_acc += torch.sum(torch.where(predictions[2] > 0.5, 1, 0) == probs).item() / len(bboxes)

                # if w_b is not None:
                #     w_b.plot_one_batch(predictions[0], images, [None]*len(predictions[0]), e)

        # calculate the average training and validation loss
        avg_train_loss = totalTrainLoss / train_steps
        avg_val_loss = totalValLoss / val_steps


        # calculate the training and validation accuracy
        # update our training history
        H["total_train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["total_val_loss"].append(avg_val_loss.cpu().detach().numpy())
        H["train_iou"].append(train_iou / train_steps)
        H["val_iou"].append(val_iou / val_steps)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.8f}".format(
            avg_train_loss, train_iou / train_steps))
        print("Val loss: {:.6f}, Val accuracy: {:.8f}".format(
            avg_val_loss, val_iou / val_steps))
        endTime = time.time()