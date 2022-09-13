import torch

def create_model(opt):
    if opt.model == 'VirtualStain_I2I':
        from .VirtualStain_I2I_model import VirtualStain_I2IModel, InferenceModel
        if opt.isTrain:
            model = VirtualStain_I2IModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
