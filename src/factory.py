# -*- coding: utf-8 -*-

"""Factory method for easily getting models by name."""

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161',
           'inception_v3', 'resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152', 'vgg11', 'vgg11_bn', 'vgg13',
           'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']


def get_model(name, **kwargs):
    """Get a model by name."""
    if name not in __all__:
        raise NotImplementedError

    if name.startswith('dense'):
        exec("from densenet import {}".format(name))
    elif name.startswith('incep'):
        exec("from inception import {}".format(name))
    elif name.startswith('res'):
        exec("from resnet import {}".format(name))
    elif name.startswith('vgg'):
        exec("from vgg import {}".format(name))
    else:
        raise NotImplementedError

    params = ""
    for key in kwargs.keys():
        params += (key + '=' + str(kwargs[key]) + ', ')
    model = eval(name + '(' + params[:-2] + ')')

    return model
