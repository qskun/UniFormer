import importlib
import torch
import torch.nn as nn

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self.backbone = self._build_backbone(net_cfg["backbone"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

    def _build_backbone(self, enc_cfg):
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x, need_fp=False):
        h, w = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x)
        
        if need_fp:
            outs = self.decoder((torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c2, nn.Dropout2d(0.5)(c2))),
                                 torch.cat((c3, nn.Dropout2d(0.5)(c3))), torch.cat((c4, nn.Dropout2d(0.5)(c4)))), h, w)
            outs, outs_fp = outs.chunk(2)
            return outs, outs_fp

        outs = self.decoder((c1, c2, c3, c4), h, w)
        return outs
