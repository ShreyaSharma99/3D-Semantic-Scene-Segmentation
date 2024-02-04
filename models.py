import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, N=10000, num_classes=3):
        super(cls_model, self).__init__()
        # pass

        self.pointNet1 = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 1024), nn.ReLU(),
            )

        # self.maxPool = nn.MaxPool1d(N)

        self.pointNet2 = nn.Sequential(
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, num_classes)
                # , nn.Softmax()
            )

              

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # pass
        B, N, _ = points.shape
        # print("points shape - ", points.shape)

        # maxpool_layer = nn.MaxPool1d(N, stride=2),
        output = self.pointNet1(points)
        # output = torch.transpose(output, 1, 2)
        # print("output shape - ", output.shape)
        
        

        output = torch.max(output, axis=1)[0]
        # self.maxPool(output).squeeze()
        # print("output now - ", output.shape)

        output = self.pointNet2(output)
        # print("output final - ", output[0, :])
        # print("Output shape final - ", output.shape)
        return output
        
        



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # pass
        self.pointNet1 = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU()
            )

        self.pointNet2 = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 1024), nn.ReLU(),
            )

        self.pointNet3 = nn.Sequential(
                nn.Linear(1088, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, num_seg_classes)
            )


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # pass
        local_feat = self.pointNet1(points)
        # B x N x 64

        output = self.pointNet2(local_feat)
        global_feat = torch.max(output, axis=1)[0]
        # B x 1024
        # print("global_feat - ", global_feat.shape)
        

        global_feat_repeated = global_feat.view(global_feat.shape[0], 1, global_feat.shape[1]).repeat(1, local_feat.shape[1], 1)
        # B x N x 1024
        # print("global_feat_repeated - ", global_feat_repeated.shape)

        concet_feat = torch.cat((local_feat, global_feat_repeated), 2)
        # print("concet_feat - ", concet_feat.shape)

        output = self.pointNet3(concet_feat)
        # print("output final - ", output[0, :])
        # print("Output shape final - ", output.shape)
        return output


# DGCNN --------------------------------------------------------------------------------------------------------------

def knn(x, k):
    inner_product = -2*torch.matmul(torch.transpose(x, 2, 1), x)
    x2 = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -x2 - inner_product - torch.transpose(x2, 2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device='cuda').view(-1, 1, 1)*num_points

    idx = idx + idx_base

    # TODO: check idx size and see if any reshaping is needed
    # TODO: check x size and see if any reshaping is needed
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = torch.transpose(x, 2, 1).contiguous() 

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # B x N x K x D
    # TODO: convert x = B x N x 1 x D to shape x = B x N x k x D (hint: repeating the elements in that dimension)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, output_channels=40):
        super(DGCNN, self).__init__()
        # self.args = args
        self.k = 20
        self.emb_dims = 1024
        self.dropout = 0.5
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        # print("Iput shape  - ", x.shape)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  
        # TODO: conv
        # TODO: max -> x1   
        x = self.conv1(x)                      
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x = get_graph_feature(x1, k=self.k)  
        # TODO: conv
        # TODO: max -> x2   
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k) 
        # TODO: conv
        # TODO: max -> x3    
        x = self.conv3(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x3, k=self.k)   
        # TODO: conv
        # TODO: max -> x4
        x = self.conv4(x)                       
        x4 = x.max(dim=-1, keepdim=False)[0]    

        x = torch.cat((x1, x2, x3, x4), dim=1)  # TODO: concat all x1 to x4
        # TODO: conv
        # TODO: pooling
        # TODO: ReLU / Leaky ReLU
        x = self.conv5(x)                       
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           
        x = torch.cat((x1, x2), 1)              

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) 
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) 
        x = self.dp2(x)
        x = self.linear3(x)                                            
        
        # print("output shape - ", x.shape)
        return x
