import torch
import torch.nn as nn
from torchsummary import summary


def _make_layers(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1


class R2Block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2, max_pool=True):
        super(R2Block, self).__init__()
        self.pool = max_pool
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.pool:
            x = self.max_pool(x)
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _make_layers(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool = False,
        recurrent: bool = True
    ):
        super().__init__()
        self.attention = attention
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if attention:
            self.attn = Attention(out_channels, out_channels, out_channels//2)
        if recurrent:
            self.conv = R2Block(in_channels, out_channels, max_pool=False)
        else:
            self.conv = _make_layers(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        if self.attention:
            x2 = self.attn(x1, x2)
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attention: bool = True,
        recurrent: bool = True
    ):
        super().__init__()
        self.attention = attention

        if recurrent:
            self.conv_in = R2Block(in_channels, 64, max_pool=False)
            self.down1 = R2Block(64, 128)
            self.down2 = R2Block(128, 256)
            self.down3 = R2Block(256, 512)
            self.down4 = R2Block(512, 1024)
        else:
            self.conv_in = _make_layers(in_channels, 64)
            self.down1 = DownSample(64, 128)
            self.down2 = DownSample(128, 256)
            self.down3 = DownSample(256, 512)
            self.down4 = DownSample(512, 1024)

        self.up1 = UpSample(1024, 512, attention=attention, recurrent=recurrent)
        self.up2 = UpSample(512, 256, attention=attention, recurrent=recurrent)
        self.up3 = UpSample(256, 128, attention=attention, recurrent=recurrent)
        self.up4 = UpSample(128, 64, attention=attention, recurrent=recurrent)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    model = UNet(in_channels=3, attention=True)
    print(summary(model, input_size=(3, 64, 64), device="cpu"))