import torch
import torch.nn as nn
import model.Basic as ops
import torch.nn.functional as F


class DRLN(nn.Module):
    def __init__(self, scale, multi_scale=True):
        super(DRLN, self).__init__()

        self.scale = scale
        channel = 64

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        # feature extraction
        self.head = nn.Conv2d(3, channel, 3, 1, 1)

        # DRLMs
        self.b1 = ops.DRLM(channel, channel)
        self.b2 = ops.DRLM(channel, channel)
        self.b3 = ops.DRLM(channel, channel)
        self.b4 = ops.DRLM(channel, channel)
        self.b5 = ops.DRLM(channel, channel)
        self.b6 = ops.DRLM(channel, channel)
        self.b7 = ops.DRLM(channel, channel)
        self.b8 = ops.DRLM(channel, channel)
        self.b9 = ops.DRLM(channel, channel)
        self.b10 = ops.DRLM(channel, channel)
        self.b11 = ops.DRLM(channel, channel)
        self.b12 = ops.DRLM(channel, channel)
        self.b13 = ops.DRLM(channel, channel)
        self.b14 = ops.DRLM(channel, channel)
        self.b15 = ops.DRLM(channel, channel)
        self.b16 = ops.DRLM(channel, channel)
        self.b17 = ops.DRLM(channel, channel)
        self.b18 = ops.DRLM(channel, channel)
        self.b19 = ops.DRLM(channel, channel)
        self.b20 = ops.DRLM(channel, channel)

        # convs
        self.c1 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c2 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c3 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c4 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c5 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c6 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c7 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c8 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c9 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c10 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c11 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c12 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c13 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c14 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c15 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c16 = ops.BasicBlockReLU(channel*5, channel, 3, 1, 1)
        self.c17 = ops.BasicBlockReLU(channel*2, channel, 3, 1, 1)
        self.c18 = ops.BasicBlockReLU(channel*3, channel, 3, 1, 1)
        self.c19 = ops.BasicBlockReLU(channel*4, channel, 3, 1, 1)
        self.c20 = ops.BasicBlockReLU(channel*5, channel, 3, 1, 1)

        # upsamle and reconstruction
        self.upsample = ops.UpsampleBlock(channel, self.scale,
                                          multi_scale=multi_scale)
        self.tail = nn.Conv2d(channel, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x

        # 1st short skip connected (SSC)
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        # 2nd short skip connected (SSC)
        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)

        b5 = self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        # 3rd short skip connected (SSC)
        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        # 4th short skip connected (SSC)
        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)

        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3

        # 5th short skip connected (SSC)
        b13 = self.b13(a4)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)

        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4

        # 6th short skip connected (SSC)
        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)

        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        a6 = o20 + a5

        # long skip connection
        b_out = a6 + x

        # upsampling
        out = self.upsample(b_out, scale=self.scale)

        # reconstruction
        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out

    def change_scale(self, scale):
        self.scale = scale

if __name__ == '__main__':
    model = DRLN(scale=3).to('cuda')
    x = torch.zeros(1, 3, 64, 64).to('cuda')
    with torch.no_grad():
        y = model(x)
        print(y.shape)

        model.change_scale(2)
        y = model(x)
        print(y.shape)

        model.change_scale(4)
        y = model(x)
        print(y.shape)
