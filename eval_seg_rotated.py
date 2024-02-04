import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg

import warnings
warnings.filterwarnings("ignore")

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

def rotate_mat(theta):
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return torch.from_numpy(rot).float()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model()
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # for angle in [0, 30, 45, 75, 90, 120, 180]:
    #     print("Angle = ", angle)
    #     theta = angle/180 * np.pi
    #     rot = rotate_mat(theta)
    #     rotated_test = test_data @ rot

    #     # ------ TO DO: Make Prediction ------
    #     pred_out = model(rotated_test)
    #     pred_label = torch.argmax(pred_out, dim=-1)

    #     test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    #     print ("test accuracy: {}".format(test_accuracy))

    #     good_seg, bad_seg = 0, 0 

    #     # Visualize Segmentation Result (Pred VS Ground Truth)
    #     # for i in range(pred_label.shape[0]):
    #     for i in [1, 2, 6, 7, 351]:
    #         seg_acc = pred_label[i, :].eq(test_label.data[i, :]).cpu().sum().item()/pred_label.shape[1]
    #         print(i, " Acc - ", seg_acc)
    #         viz_seg(rotated_test[i], test_label[i], "{}/rotate_gt_seg_{}_{}.gif".format(args.output_dir, angle, i), args.device)
    #         viz_seg(rotated_test[i], pred_label[i], "{}/rotate_pred_seg_{}_{}.gif".format(args.output_dir, angle, i), args.device)

    num_points = args.num_points
    # [10000, 7500, 5000, 2000, 1000]
    for num_points in [50, 20]:
        # Sample Points per Object
        ind = np.random.choice(10000, num_points, replace=False)
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        # print("Test_data - ", test_data.shape)
        test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
        
        # ------ TO DO: Make Prediction ------
        pred_out = model(test_data)
        pred_label = torch.argmax(pred_out, dim=-1)

        # Compute Accuracy
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
        print("Nsample - ", num_points)
        print("Total correct - ", pred_label.eq(test_label.data).cpu().sum().item(), " total = ", pred_label.shape[0])
        print ("test accuracy: {}".format(test_accuracy))

        count = 0
        for i in [1, 2, 6, 7, 351]:
            # if pred_label[i] != test_label[i] == 1 and count<5:
            seg_acc = pred_label[i, :].eq(test_label.data[i, :]).cpu().sum().item()/pred_label.shape[1]
            print(i, " Acc - ", seg_acc)
            # print(pred_label[i], test_label[i])
            viz_seg(test_data[i], test_label[i], "{}/nsample_gt_seg_{}_{}.gif".format(args.output_dir, num_points, i), args.device)
            viz_seg(test_data[i], pred_label[i], "{}/nsample_pred_seg_{}_{}.gif".format(args.output_dir, num_points, i), args.device)

            count+=1 



import matplotlib.pyplot as plt
import numpy as np

n = [10000, 5000, 1000, 500, 100, 50, 20]
x = np.log(n)
y = [88.26, 88.2, 87.3, 86, 79.96, 37.68, 14.25]
plt.scatter(x, y)
plt.plot(x, y, color='red')
plt.xticks(x, n)
plt.xlabel('Number of Points')
plt.ylabel('Avg Segmentation Accuracy')
plt.title('PointNet invariance to number of points for segmentation')

plt.savefig('./data/seg_nsample.png')
plt.show()