import torchvision
from torch.autograd import Variable
import torch

def nms(data):



    return ll


if __name__ == '__main__':

    dummy_input = Variable(torch.randn(2, 10647, 85))
    dummy_input = dummy_input.view(-1, 85)
    boxes = dummy_input[..., 0:4].squeeze()
    scores = dummy_input[..., 4]
    result = torchvision.ops.nms(boxes, scores, 0.4)
    ll = torch.index_select(dummy_input, 0, result)

    print(result.shape)