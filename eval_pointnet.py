import numpy as np
import argparse

import torch
from pointnet_plus_model import get_model
from utils import create_dir
from data_loader import get_data_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--test_data', type=str, default='./data/pointnet++/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/pointnet++/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--task', type=str, default="pointnet++", help='The task: pointnet++')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    with torch.no_grad():
        model = get_model(3)
        model = model.cuda()
        
        print("load checkpoint - ", args.load_checkpoint)
        # Load Model Checkpoint
        model_path = './checkpoints/pointnet++/{}.pt'.format(args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        model.eval()
        print ("successfully loaded checkpoint from {}".format(model_path))


        # # Sample Points per Object
        # ind = np.random.choice(10000,args.num_points, replace=False)
        # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
        # test_label = torch.from_numpy(np.load(args.test_label))

        correct_obj = 0
        num_obj = 0

        test_dataloader = get_data_loader(args=args, train=False)

        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # ------ TO DO: Make Predictions ------
            # with torch.no_grad():
            pred_out = model(point_clouds)
            pred_labels = torch.argmax(pred_out, dim=-1)
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

    test_accuracy = correct_obj / num_obj

    # # ------ TO DO: Make Prediction ------
    # pred_out = model(test_data)
    # pred_label = torch.argmax(pred_out, dim=-1)

    # Compute Accuracy
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

