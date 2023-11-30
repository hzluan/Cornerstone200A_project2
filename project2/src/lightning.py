import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from torchvision.models import resnet18
from torchvision.models.video import r3d_18
from torchvision.models.swin_transformer import swin_v2_b
import numpy as np
from src.cindex import concordance_index

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=1e-4, use_attention=False):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        ######################################
        self.use_attention = use_attention
        self.loss = nn.CrossEntropyLoss()
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def AttentionLoss(self, alpha, A):
        # noremalise by things that have annpotations
        # view so dimension is batch, 1
        is_legit = torch.sum(A, (1, 2, 3)) > 0
        # Maxpool A 
        A = F.max_pool3d(A, kernel_size=7)
        likelihood = torch.sum(alpha*A, (1, 2, 3))
        total_loss = -torch.log(likelihood.detach().cpu() + 10e-9)
        avg_loss = torch.sum(torch.einsum('ij, ik -> ', is_legit.cpu().type(torch.LongTensor), total_loss.cpu().type(torch.LongTensor)))/(torch.sum(is_legit)+10e-9)
        return avg_loss
    
    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y, annotation_mask = batch["x"], batch["y_seq"][:,0], batch['mask']
        return x, y.to(torch.long).view(-1), annotation_mask

    def training_step(self, batch, batch_idx):
        x, y, annotation_mask = self.get_xy(batch)

        ## TODO: get predictions from your model and store them as y_hat
        ###################################################
        y_hat, alpha = self.forward(x)

        if self.use_attention:
            attention_loss = self.AttentionLoss(alpha, annotation_mask)
            attn_weight = 1e-5
            loss = attn_weight*attention_loss + self.loss(y_hat, y)
        else:
            attention_loss = 0
            loss = self.loss(y_hat, y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_attention_loss', attention_loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, annotation_mask = self.get_xy(batch)
        #################################################
        y_hat, alpha = self.forward(x)

        if self.use_attention:
            attention_loss = self.AttentionLoss(alpha, annotation_mask)
            loss = self.loss(y_hat, y)
        else:
            attention_loss = 0
            loss = self.loss(y_hat, y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        self.log('val_attention_loss', attention_loss, sync_dist=True,  prog_bar=True)


        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y, annotation_mask = self.get_xy(batch)
        ###############################
        y_hat, alpha = self.forward(x)

        if self.use_attention:
            attention_loss = self.AttentionLoss(alpha, annotation_mask)
            loss = self.loss(y_hat, y)
        else:
            attention_loss = 0
            loss = self.loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_attention_loss', attention_loss, sync_dist=True,  prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        
        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        ########################
        self.opt = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return self.opt

class MLP(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr = 1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        #######################################
        layers = []


        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        #######################################
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.network(x), None

class CNN(Classifer):
    def __init__(self, input_dim=28*28*3, input_chan=3, out_chan=128, num_layers=6, kernel_size=3, stride=1, num_classes=9, use_bn=False, model_tune=False, **kwargs):
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters()

        self.out_chan = out_chan
        self.use_bn = use_bn
        self.kernel_size = kernel_size
        self.stride = stride

        blocks = []
        input_H = 28
        output_H = 28

        # TODO: Implement CNN
        # use different number of layers, kernel size, strides
        for _ in range(num_layers):
            layers = []
            layers.append(nn.Conv2d(input_chan, out_chan, kernel_size=kernel_size, padding=1))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_chan))
            layers.append(nn.ReLU())
            input_chan = out_chan
            output_H = (input_H + 2*1 - kernel_size) // stride + 1
            input_H = output_H

            blocks.append(nn.Sequential(*layers))

        if model_tune==True:
            blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # find output dimension after maxpool layer
            output_H = (output_H + 2*1 - 3) // 2 + 1
        # squeeze the spatial dimensions
        blocks.append(nn.Flatten())
        output_dim = output_H * output_H * out_chan
        blocks.append(nn.Linear(output_dim, num_classes))

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x), None

class ResNet18(Classifer):
    def __init__(self, num_classes=9, init_lr = 1e-3, pretrained=False,model_tune=False,**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.pretrained = pretrained
        if self.pretrained:
            self.resnet = resnet18(weights='DEFAULT')
        else:
            self.resnet = resnet18()
        ####################
        # experimenting with resnet18
        if model_tune==True:
            self.resnet.maxpool = nn.avgpool2d(kernel_size=3, stride=2, padding=1)
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x = self.resnet(x)
        return x, None

class CNN3D(Classifer):
    def __init__(self, input_dim=256*256*200*3, input_chan=1, out_chan=32, num_layers=3, kernel_size=3, stride = 1, num_classes=9, use_bn=False, use_attention=False, attention_mask=None, **kwargs):
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters()

        self.out_chan = out_chan
        self.use_bn = use_bn
        self.kernel_size = kernel_size
        self.stride = stride

        self.attention_mask = attention_mask

        layers = []
        input_H = 256
        output_H = 256
        input_D = 200
        output_D = 200

        for _ in range(num_layers):
            layers.append(nn.Conv3d(input_chan, out_chan, kernel_size=kernel_size, padding=1))
            if self.use_bn:
                layers.append(nn.BatchNorm3d(out_chan))
            layers.append(nn.ReLU())
            input_chan = out_chan
            output_H = (input_H + 2*1 - kernel_size[0]) // stride + 1
            output_D = (input_D + 2*1 - kernel_size[0]) // stride + 1
            input_H = output_H
            input_D = output_D

        # squeeze the spatial dimensions
        layers.append(nn.Flatten())
        # find dimensions of output
        output_dim = output_H * output_H * out_chan * output_D

        self.network = nn.Sequential(*layers)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Conv3d(in_channels=input_chan, out_channels=1,
                                    kernel_size=1, stride=1)
        
        self.fc = nn.Linear(output_dim, num_classes)
    
    def forward(self, x):
        B, C, D, H, W = x.size()
        h = self.network(x)
        alpha = None
        if self.use_attention:
            alpha = self.attention(x)
            alpha = F.softmax(alpha.view(B, -1), -1).view(B, 1, D, H, W)
            h = alpha*h
            # add maxpooling layer and concatenate
        h_logit = self.fc(h)
        return h_logit, alpha

#############################
# for 3d resnet18
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet183D(Classifer):

    def __init__(self, block=BasicBlock3D, layers=[2,2,2,2], num_classes=2, pretrained=False, use_attention=False, init_lr=1e-4, random_init=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()
        self.inplanes = 64
        self.pretrained = pretrained
        self.random_init = random_init
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.maxpool2 = nn.MaxPool3d((3, 3, 3))

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Conv3d(in_channels=1, out_channels=1,
                                    kernel_size=1, stride=1)
            self.attn_maxpool = nn.MaxPool3d((7, 7, 7))
            self.fc_attn = nn.Linear(36288, 128)
            self.fc_max = nn.Linear(512, 28)
            self.fc = nn.Linear(352, num_classes)
        else:
            self.fc = nn.Linear(4096, num_classes)

        if self.pretrained:
            pretrained_model = resnet18(pretrained=True)
            # model_weight_path = '../models/resnet18-5c106cde.pth'
            # pretrained_weights = torch.load(model_weight_path)
            # pretrained_model = resnet18()
            # pretrained_model.load_state_dict(pretrained_weights)
            # self.conv1.weight.data = self.repeat_weights(pretrained_model.conv1.weight.data, 200)
            if self.random_init:
                for name, module in pretrained_model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        reference_weights = pretrained_model.state_dict()[name + '.weight']
                        module.apply(lambda m: self.init_weights(m, reference_weights))
            self._initialize_3d_from_2d(pretrained_model)

    def init_weights(self, m, reference_weights):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight = nn.Parameter(torch.randn_like(m.weight) * reference_weights.std() + reference_weights.mean())
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def repeat_weights(self, w2d, num_repeats):
        c_out, c_in, h, w = w2d.size()
        w3d = w2d.unsqueeze(4).repeat(1, 1, 1, 1, num_repeats) 
        return w3d
    
    def _initialize_3d_from_2d(self, pretrained_model):
        pretrained_layers = [
            pretrained_model.layer1, 
            pretrained_model.layer2, 
            pretrained_model.layer3, 
            pretrained_model.layer4
        ]
        model_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        for pre_layer, model_layer in zip(pretrained_layers, model_layers):
            for pre_block, model_block in zip(pre_layer, model_layer):
                model_block.conv1.weight.data = self.repeat_weights(pre_block.conv1.weight.data, 3)
                model_block.conv2.weight.data = self.repeat_weights(pre_block.conv2.weight.data, 3)
                model_block.bn1.weight.data = pre_block.bn1.weight.data.clone()
                model_block.bn1.bias.data = pre_block.bn1.bias.data.clone()
                model_block.bn2.weight.data = pre_block.bn2.weight.data.clone()
                model_block.bn2.bias.data = pre_block.bn2.bias.data.clone()
    

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, D, H, W = x.size()
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 6, 512, 8, 8, 7
        max_pooling = self.maxpool2(x) # B, 512, 2, 2, 2
        max_pooling = max_pooling.view(B, -1, max_pooling.size()[1]) # 6, 8, 512
        alpha = None
        if self.use_attention:
            alpha = self.attention(residual)
            alpha = F.softmax(alpha.view(B, -1), -1).view(B, 1, D, H, W)
            alpha = alpha*residual
            # add maxpooling layer and concatenate
            alpha = self.attn_maxpool(alpha) # B, 1, 36, 36, 28
            attn_pooling = self.fc_attn(alpha.view(B, -1)) # B, 
            max_pooling = self.fc_max(max_pooling) # B, 8, 28
            x = torch.concat([attn_pooling, max_pooling.view(B,-1)], dim=1)
            x = self.fc(x)
            return x, alpha
        else:
            max_pooling = max_pooling.view(B, -1)
            x = self.fc(max_pooling)
            return x,  alpha

class R3D(Classifer):
    def __init__(self, num_classes=9, init_lr = 1e-3, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.pretrained = pretrained
        if self.pretrained:
            self.resnet3d = r3d_18(weights='DEFAULT')
        else:
            self.resnet3d = r3d_18()
        
        original_first_layer = self.resnet3d.stem[0]

        new_first_layer = torch.nn.Conv3d(1, 
                                        original_first_layer.out_channels, 
                                        kernel_size=original_first_layer.kernel_size, 
                                        stride=original_first_layer.stride, 
                                        padding=original_first_layer.padding, 
                                        bias=False)

        with torch.no_grad():
            new_first_layer.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

        setattr(self.resnet3d.stem, '0', new_first_layer)

        num_ftrs = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet3d(x)
        return x, None
 
class SwinTransformer(Classifer):
    def __init__(self, num_classes=9, init_lr = 1e-3, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.pretrained = pretrained
        if self.pretrained:
            self.swin = swin_v2_b(weights='DEFAULT')
        else:
            self.swin = swin_v2_b()

        num_ftrs = self.swin.head.in_features
        self.swin.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x = self.swin(x)
        return x

NLST_CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}
class RiskModel(Classifer):
    def __init__(self, input_num_chan=1, num_classes=2, init_lr = 1e-3, max_followup=6, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = 512

        ## Maximum number of followups to predict (set to 6 for full risk prediction task)
        self.max_followup = max_followup

        # TODO: Initalize components of your model here
        raise NotImplementedError("Not implemented yet")



    def forward(self, x):
        raise NotImplementedError("Not implemented yet")

    def get_xy(self, batch):
        """
            x: (B, C, D, W, H) -  Tensor of CT volume
            y_seq: (B, T) - Tensor of cancer outcomes. a vector of [0,0,1,1,1, 1] means the patient got between years 2-3, so
            had cancer within 3 years, within 4, within 5, and within 6 years.
            y_mask: (B, T) - Tensor of mask indicating future time points are observed and not censored. For example, if y_seq = [0,0,0,0,0,0], then y_mask = [1,1,0,0,0,0], we only know that the patient did not have cancer within 2 years, but we don't know if they had cancer within 3 years or not.
            mask: (B, D, W, H) - Tensor of mask indicating which voxels are inside an annotated cancer region (1) or not (0).
                TODO: You can add more inputs here if you want to use them from the NLST dataloader.
                Hint: You may want to change the mask definition to suit your localization method

        """
        return batch['x'], batch['y_seq'][:, :self.max_followup], batch['y_mask'][:, :self.max_followup], batch['mask']

    def step(self, batch, batch_idx, stage, outputs):
        x, y_seq, y_mask, region_annotation_mask = self.get_xy(batch)

        # TODO: Get risk scores from your model
        y_hat = None ## (B, T) shape tensor of risk scores.
        # TODO: Compute your loss (with or without localization)
        loss = None

        raise NotImplementedError("Not implemented yet")
        
        # TODO: Log any metrics you want to wandb
        metric_value = -1
        metric_name = "dummy_metric"
        self.log('{}_{}'.format(stage, metric_name), metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        # TODO: Store the predictions and labels for use at the end of the epoch for AUC and C-Index computation.
        outputs.append({
            "y_hat": y_hat, # Logits for all risk scores
            "y_mask": y_mask, # Tensor of when the patient was observed
            "y_seq": y_seq, # Tensor of when the patient had cancer
            "y": batch["y"], # If patient has cancer within 6 years
            "time_at_event": batch["time_at_event"] # Censor time
        })

        return loss
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.training_outputs)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_outputs)

    def on_epoch_end(self, stage, outputs):
        y_hat = F.sigmoid(torch.cat([o["y_hat"] for o in outputs]))
        y_seq = torch.cat([o["y_seq"] for o in outputs])
        y_mask = torch.cat([o["y_mask"] for o in outputs])

        for i in range(self.max_followup):
            '''
                Filter samples for either valid negative (observed followup) at time i
                or known pos within range i (including if cancer at prev time and censoring before current time)
            '''
            valid_probs = y_hat[:, i][(y_mask[:, i] == 1) | (y_seq[:,i] == 1)]
            valid_labels = y_seq[:, i][(y_mask[:, i] == 1)| (y_seq[:,i] == 1)]
            self.log("{}_{}year_auc".format(stage, i+1), self.auc(valid_probs, valid_labels.view(-1)), sync_dist=True, prog_bar=True)

        y = torch.cat([o["y"] for o in outputs])
        time_at_event = torch.cat([o["time_at_event"] for o in outputs])

        if y.sum() > 0 and self.max_followup == 6:
            c_index = concordance_index(time_at_event.cpu().numpy(), y_hat.detach().cpu().numpy(), y.cpu().numpy(), NLST_CENSORING_DIST)
        else:
            c_index = 0
        self.log("{}_c_index".format(stage), c_index, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.on_epoch_end("train", self.training_outputs)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        self.on_epoch_end("val", self.validation_outputs)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        self.on_epoch_end("test", self.test_outputs)
        self.test_outputs = []

