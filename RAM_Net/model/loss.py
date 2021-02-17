import torch.nn.functional as F
import torch
from kornia.filters.sobel import spatial_gradient, sobel


def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
    log_diff = y_input - y_target
    is_nan = torch.isnan(log_diff)
    return weight * ((log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2))


def scale_invariant_log_loss(y_input, y_target, n_lambda = 1.0):
    log_diff = torch.log(y_input)-torch.log(y_target)
    is_nan = torch.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(n_lambda*(log_diff[~is_nan].mean())**2)


def mse_loss(y_input, y_target):
    return F.mse_loss(y_input[~torch.isnan(y_target)], y_target[~torch.isnan(y_target)])


class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]
        print('Done')

    def forward(self, prediction, target, preview = False):
        # helper to remove potential nan in labels
        def nan_helper(y):
            return torch.isnan(y), lambda z: z.nonzero()[0]
        
        loss_value = 0
        loss_value_2 = 0
        diff = prediction - target
        _,_,H,W = target.shape
        upsample = torch.nn.Upsample(size=(2*H,2*W), mode='bicubic', align_corners=True)
        record = []

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                loss_value += torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
                # * batch size * 2 (because kornia spatial product has two outputs).
                # replaces the following line to be able to deal with nan's.
                # loss_value += torch.abs(delta_diff).mean(dim=(3,4)).sum()

        if preview:
            return record
        else:
            return (loss_value/self.num_scales)


multi_scale_grad_loss_fn = MultiScaleGradient()


def multi_scale_grad_loss(prediction, target, preview = False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)

