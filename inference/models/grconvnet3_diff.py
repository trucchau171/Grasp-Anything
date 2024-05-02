import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.models.diffusion_utils import *
import open_clip


class GenerativeResnet_Diff(GraspModel):

    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False, prob=0.0, latent_dim=512):
        super(GenerativeResnet_Diff, self).__init__()
        self.latent_dim = latent_dim

        clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip = clip.float()        
        self.clip.eval()
        self.clip_dim = 512
        for name, param in self.clip.named_parameters():
                param.requires_grad = False

        self.proj_text = nn.Sequential(nn.Linear(self.clip_dim * 2, 128), nn.ReLU())

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        self.attn_layer = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

         # Setup timestep embedding layer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Setup U-net-like input and output process
        # self.input_process = InputProcess(self.data_rep, self.xyz_dim, self.extract_dim).to(self.device)
        # self.output_process = OutputProcess(self.data_rep, self.xyz_dim, self.extract_dim, self.pcd_points).to(self.device)


    def encode_clip_text(self, prompt):
        text_tokens = self.tokenizer(prompt).cuda()
        text_features = self.clip.encode_text(text_tokens)
        return text_features

    def forward(self, x_in, timesteps, given_x, prompt):
        """
        x: noisy signal - torch.Tensor.shape([bs,..])
        timesteps: torch.Tensor.shape([bs,])
        """
        # Embed features from time
        emb_ts = self.embed_timestep(timesteps)
        emb_ts = emb_ts.permute(1, 0, 2)
        emb_ts = emb_ts.squeeze()

        emb_text = self.encode_clip_text(prompt)

        if len(emb_ts.shape) == 1:
            emb_ts = emb_ts.unsqueeze(0)
        # print(emb_text.shape)
        # print(emb_ts.shape)
        emb_cond = torch.cat((emb_ts, emb_text), dim=-1)
        emb_cond = self.proj_text(emb_cond)
        emb_cond = emb_cond.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(given_x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        grid = x.shape[2]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
       
        attn_output, attn_output_weights = self.attn_layer(x, emb_cond, emb_cond)
        if len(attn_output.shape) == 4:
            attn_output = attn_output.squeeze()
        attn_output = attn_output.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], grid, grid)  # shape = [*, width, grid ** 2]
    
        x = F.relu(self.bn4(self.conv4(attn_output)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        # add noise to the output
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)
        
        
        output = torch.cat((pos_output, cos_output, sin_output, width_output), dim=1)

        x_in += output


        # shape conv6 [8, 32, 225, 225], pos_output [8, 1, 224, 224], cos_output [8, 1, 224, 224], sin_output [8, 1, 224, 224], width_output [8, 1, 224, 224]

        return x_in
    
