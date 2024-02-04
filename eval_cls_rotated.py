from cgi import test
from tokenize import Double
import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_pointCloud

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="cls", help='The name of the experiment')

    return parser

def rotate_mat(theta):
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return torch.from_numpy(rot).float()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model()
    # model = model.cuda()
    
    print("load checkpoint - ", args.load_checkpoint)
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))
    
    # # rotate = np.array([[0, -1, 0],  [1, 0, 0], [0, 0, 1]])

    # # test_data = test_data @ rotate

    
    # for angle in [0, 30, 45, 75, 90, 120, 180]:
    #     print("ANgle = ", angle)
    #     theta = angle/180 * np.pi
    #     rot = rotate_mat(theta)
    #     rotated_test = test_data @ rot
    #     # ------ TO DO: Make Prediction ------
    #     pred_out = model(rotated_test)
    #     pred_label = torch.argmax(pred_out, dim=-1)

    #     # Compute Accuracy
    #     test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])

    #     print("Total correct - ", pred_label.eq(test_label.data).cpu().sum().item(), " total = ", pred_label.shape[0])
    #     print ("test accuracy: {}".format(test_accuracy))

    #     count = 0
    #     for i in [1, 3, 620, 625, 720, 721]:
    #         # if pred_label[i] != test_label[i] == 1 and count<5:
    #         print(i)
    #         print(pred_label[i], test_label[i])
    #         viz_pointCloud(rotated_test[i], "{}/rotate_{}_{}_{}_{}_{}.gif".format(args.output_dir, args.exp_name, angle, i, test_label[i], pred_label[i]), args.device)
    #         # viz_pointCloud(test_data[i], "{}/pc_w_{}_{}_{}.gif".format(args.output_dir, args.exp_name, i, pred_label[i]), args.device)
    #         count+=1 


    num_points = args.num_points
    # [10000, 7500, 5000, 2000, 1000]
    for num_points in [50, 20]:
        # Sample Points per Object
        ind = np.random.choice(10000, num_points, replace=False)
        test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        print("Test_data - ", test_data.shape)
        test_label = torch.from_numpy(np.load(args.test_label))
        
        # ------ TO DO: Make Prediction ------
        pred_out = model(test_data)
        pred_label = torch.argmax(pred_out, dim=-1)

        # Compute Accuracy
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])

        print("Total correct - ", pred_label.eq(test_label.data).cpu().sum().item(), " total = ", pred_label.shape[0])
        print ("test accuracy: {}".format(test_accuracy))

        count = 0
        for i in [1, 3, 620, 625, 720, 721]:
            # if pred_label[i] != test_label[i] == 1 and count<5:
            print(i)
            print(pred_label[i], test_label[i])
            viz_pointCloud(test_data[i], "{}/nsample_cls_{}_{}_{}_{}.gif".format(args.output_dir, num_points, i, test_label[i], pred_label[i]), args.device)
            # viz_pointCloud(test_data[i], "{}/pc_w_{}_{}_{}.gif".format(args.output_dir, args.exp_name, i, pred_label[i]), args.device)
            count+=1 

