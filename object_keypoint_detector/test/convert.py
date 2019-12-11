import torch
import torchfile
from hourglass import CreateModel

model = CreateModel()

model_lua = torchfile.load('../models/model_100c.t7')

# conv

layer_conv = ['model.layer1',
'model.layer4.layer2.layer1',
'model.layer8.layer2.layer1',
'model.layer9.layer3.layer2.layer1',
'model.layer9.layer8.layer3.layer2.layer1',
'model.layer9.layer8.layer8.layer3.layer2.layer1',
'model.layer9.layer8.layer8.layer8.layer3.layer2.layer1',
'model.layer9.layer8.layer8.layer8.layer8.layer2.layer1',
'model.layer10.layer1',
'model.layer11.layer1',
'model.layer12',
'model.layer13',
'model.layer14',
'model.layer15.layer1.layer2.layer1',
'model.layer15.layer3.layer2.layer1',
'model.layer15.layer5.layer2.layer1',
'model.layer15.layer8.layer3.layer2.layer1',
'model.layer15.layer8.layer8.layer3.layer2.layer1',
'model.layer15.layer8.layer8.layer8.layer3.layer2.layer1',
'model.layer15.layer8.layer8.layer8.layer8.layer2.layer1',
'model.layer16.layer1',
'model.layer17.layer1',
'model.layer18']

module_conv = [[0],
[3, 0, 1, 0],
[5, 0, 2, 0, 1, 0],
[5, 0, 3, 0, 0, 2, 0, 1, 0],
[5, 0, 3, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 0],
[5, 0, 4, 0],
[5, 0, 5, 0],
[6, 1, 1],
[7, 1, 1],
[6, 0, 1],
[8, 1, 2, 0, 0, 0, 0, 1, 0],
[8, 1, 2, 0, 0, 2, 0, 1, 0],
[8, 1, 2, 0, 1, 1, 0, 1, 0],
[8, 1, 2, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 1, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 0],
[8, 1, 3, 0],
[8, 1, 4, 0],
[8, 1, 5]]


for layer, module in zip(layer_conv, module_conv):
    module_string = 'model_lua'
    for module_id in module:
        module_string = '%s[\'modules\'][%d]' % (module_string, module_id)

    exec_string = '%s.weight.data = torch.FloatTensor(%s[\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.bias.data = torch.FloatTensor(%s[\'bias\'])' %(layer, module_string); exec(exec_string)


# batchnorm

layer_batchnorm = ['model.layer2',
'model.layer10.layer2',
'model.layer11.layer2',
'model.layer16.layer2',
'model.layer17.layer2']


module_batchnorm = [[1],
[5, 0, 4, 1],
[5, 0, 5, 1],
[8, 1, 3, 1],
[8, 1, 4, 1]]


for layer, module in zip(layer_batchnorm, module_batchnorm):
    module_string = 'model_lua'
    for module_id in module:
        module_string = '%s[\'modules\'][%d]' % (module_string, module_id)

    exec_string = '%s.weight.data = torch.FloatTensor(%s[\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.bias.data = torch.FloatTensor(%s[\'bias\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.running_mean.data = torch.FloatTensor(%s[\'running_mean\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.running_var.data = torch.FloatTensor(%s[\'running_var\'])' %(layer, module_string); exec(exec_string)


# Residual

layer_residual = ['model.layer4.layer1',
'model.layer6.layer1',
'model.layer7.layer1',
'model.layer8.layer1',
'model.layer9.layer1.layer1',
'model.layer9.layer2.layer1',
'model.layer9.layer3.layer1',
'model.layer9.layer5.layer1',
'model.layer9.layer6.layer1',
'model.layer9.layer7.layer1',
'model.layer9.layer8.layer1.layer1',
'model.layer9.layer8.layer2.layer1',
'model.layer9.layer8.layer3.layer1',
'model.layer9.layer8.layer5.layer1',
'model.layer9.layer8.layer6.layer1',
'model.layer9.layer8.layer7.layer1',
'model.layer9.layer8.layer8.layer1.layer1',
'model.layer9.layer8.layer8.layer2.layer1',
'model.layer9.layer8.layer8.layer3.layer1',
'model.layer9.layer8.layer8.layer5.layer1',
'model.layer9.layer8.layer8.layer6.layer1',
'model.layer9.layer8.layer8.layer7.layer1',
'model.layer9.layer8.layer8.layer8.layer1.layer1',
'model.layer9.layer8.layer8.layer8.layer2.layer1',
'model.layer9.layer8.layer8.layer8.layer3.layer1',
'model.layer9.layer8.layer8.layer8.layer5.layer1',
'model.layer9.layer8.layer8.layer8.layer6.layer1',
'model.layer9.layer8.layer8.layer8.layer7.layer1',
'model.layer9.layer8.layer8.layer8.layer8.layer1',
'model.layer9.layer8.layer8.layer8.layer9.layer1',
'model.layer9.layer8.layer8.layer9.layer1',
'model.layer9.layer8.layer9.layer1',
'model.layer9.layer9.layer1',
'model.layer15.layer1.layer1',
'model.layer15.layer2.layer1',
'model.layer15.layer3.layer1',
'model.layer15.layer5.layer1',
'model.layer15.layer6.layer1',
'model.layer15.layer7.layer1',
'model.layer15.layer8.layer1.layer1',
'model.layer15.layer8.layer2.layer1',
'model.layer15.layer8.layer3.layer1',
'model.layer15.layer8.layer5.layer1',
'model.layer15.layer8.layer6.layer1',
'model.layer15.layer8.layer7.layer1',
'model.layer15.layer8.layer8.layer1.layer1',
'model.layer15.layer8.layer8.layer2.layer1',
'model.layer15.layer8.layer8.layer3.layer1',
'model.layer15.layer8.layer8.layer5.layer1',
'model.layer15.layer8.layer8.layer6.layer1',
'model.layer15.layer8.layer8.layer7.layer1',
'model.layer15.layer8.layer8.layer8.layer1.layer1',
'model.layer15.layer8.layer8.layer8.layer2.layer1',
'model.layer15.layer8.layer8.layer8.layer3.layer1',
'model.layer15.layer8.layer8.layer8.layer5.layer1',
'model.layer15.layer8.layer8.layer8.layer6.layer1',
'model.layer15.layer8.layer8.layer8.layer7.layer1',
'model.layer15.layer8.layer8.layer8.layer8.layer1',
'model.layer15.layer8.layer8.layer8.layer9.layer1',
'model.layer15.layer8.layer8.layer9.layer1',
'model.layer15.layer8.layer9.layer1',
'model.layer15.layer9.layer1']

module_residual = [[3, 0, 0],
[5, 0, 0, 0, 0],
[5, 0, 1, 0, 0],
[5, 0, 2, 0, 0],
[5, 0, 3, 0, 0, 0, 0, 0],
[5, 0, 3, 0, 0, 1, 0, 0],
[5, 0, 3, 0, 0, 2, 0, 0],
[5, 0, 3, 0, 1, 1, 0, 0],
[5, 0, 3, 0, 1, 2, 0, 0],
[5, 0, 3, 0, 1, 3, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 0, 0, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 0, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 0, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 3, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 0, 0, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 0, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 0, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 1, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 2, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 5, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 4, 0, 1, 5, 0, 0],
[5, 0, 3, 0, 1, 4, 0, 1, 5, 0, 0],
[5, 0, 3, 0, 1, 5, 0, 0],
[8, 1, 2, 0, 0, 0, 0, 0],
[8, 1, 2, 0, 0, 1, 0, 0],
[8, 1, 2, 0, 0, 2, 0, 0],
[8, 1, 2, 0, 1, 1, 0, 0],
[8, 1, 2, 0, 1, 2, 0, 0],
[8, 1, 2, 0, 1, 3, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 0, 0, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 0, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 0, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 3, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 0, 0, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 0, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 0, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 1, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 2, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 3, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 4, 0, 1, 5, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 4, 0, 1, 5, 0, 0],
[8, 1, 2, 0, 1, 4, 0, 1, 5, 0, 0],
[8, 1, 2, 0, 1, 5, 0, 0]]


for layer, module in zip(layer_residual, module_residual):
    module_string = 'model_lua'
    for module_id in module:
        module_string = '%s[\'modules\'][%d]' % (module_string, module_id)
    exec_string = '%s.layer1.weight.data = torch.FloatTensor(%s[\'modules\'][0][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer1.bias.data = torch.FloatTensor(%s[\'modules\'][0][\'bias\'])' %(layer, module_string); exec(exec_string)

    exec_string = '%s.layer1.running_mean.data = torch.FloatTensor(%s[\'modules\'][0][\'running_mean\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer1.running_var.data = torch.FloatTensor(%s[\'modules\'][0][\'running_var\'])' %(layer, module_string); exec(exec_string)

    exec_string = '%s.layer3.weight.data = torch.FloatTensor(%s[\'modules\'][2][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer3.bias.data = torch.FloatTensor(%s[\'modules\'][2][\'bias\'])' %(layer, module_string); exec(exec_string)

    exec_string = '%s.layer4.weight.data = torch.FloatTensor(%s[\'modules\'][3][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer4.bias.data = torch.FloatTensor(%s[\'modules\'][3][\'bias\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer4.running_mean.data = torch.FloatTensor(%s[\'modules\'][3][\'running_mean\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer4.running_var.data = torch.FloatTensor(%s[\'modules\'][3][\'running_var\'])' %(layer, module_string); exec(exec_string)  

    exec_string = '%s.layer6.weight.data = torch.FloatTensor(%s[\'modules\'][5][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer6.bias.data = torch.FloatTensor(%s[\'modules\'][5][\'bias\'])' %(layer, module_string); exec(exec_string)

    exec_string = '%s.layer7.weight.data = torch.FloatTensor(%s[\'modules\'][6][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer7.bias.data = torch.FloatTensor(%s[\'modules\'][6][\'bias\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer7.running_mean.data = torch.FloatTensor(%s[\'modules\'][6][\'running_mean\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer7.running_var.data = torch.FloatTensor(%s[\'modules\'][6][\'running_var\'])' %(layer, module_string); exec(exec_string)

    exec_string = '%s.layer9.weight.data = torch.FloatTensor(%s[\'modules\'][8][\'weight\'])' %(layer, module_string); exec(exec_string)
    exec_string = '%s.layer9.bias.data = torch.FloatTensor(%s[\'modules\'][8][\'bias\'])' %(layer, module_string); exec(exec_string)


model.cuda()
model.eval()
import h5py
import numpy as np
f = h5py.File('valid_17.h5', 'r')
a = np.array(f['input'])
out = model.forward(torch.from_numpy((a)).float().cuda())
print(out[1].max().item())
