import torch
import torch.nn as nn

BIAS_factor = 0.0
BIAS = 0.0
LAYERS=4

F_And = None
F_Or = None
F_Implies = None
F_Equiv = None
F_Not = None
F_Forall = None
F_Exists = None


def set_tnorm(tnorm):
    assert tnorm in ['min','luk','prod','mean','']
    global F_And,F_Or,F_Implies,F_Not,F_Equiv,F_Forall
    if tnorm == "min":
        def F_And(wffs):
            return torch.mean(wffs, dim=-1, keepdim=True)

        def F_Or(wffs):
            result,_ = torch.max(wffs, dim=-1, keepdim=True)
            return result

        def F_Implies(wff1, wff2):
            return torch.max((wff1 <= wff2).type(torch.FloatTensor), wff2)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return torch.max((wff1 == wff2).type(torch.FloatTensor), torch.min(wff1, wff2))

    if tnorm == "prod":
        def F_And(wffs):
            return torch.prod(wffs, dim=-1, keepdim=True)

        def F_Or(wffs):
            return 1 - torch.prod(1-wffs, dim=-1, keepdim=True)

        def F_Implies(wff1, wff2):
            le_wff1_wff2 = (wff1 <= wff2).type(torch.FloatTensor)
            gt_wff1_wff2 = (wff1 > wff2).type(torch.FloatTensor)
            return le_wff1_wff2 + gt_wff1_wff2*wff2/wff1 if wff1[0] == 0 else torch.tensor([1.0])

        def F_Not(wff):
            # according to standard goedel logic is
            return 1-wff

        def F_Equiv(wff1,wff2):
            return torch.min(wff1/wff2, wff2/wff1)

    if tnorm == "mean":
        def F_And(wffs):
            return torch.mean(wffs, dim=-1, keepdim=True)

        def F_Or(wffs):
            result,_ = torch.max(wffs, dim=-1, keepdim=True)
            return result

        def F_Implies(wff1, wff2):
            return torch.clamp(2*wff2-wff1, 0, 1)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return 1 - torch.abs(wff1-wff2)

    if tnorm == "luk":
        def F_And(wffs):
            result = torch.sum(wffs, dim=-1, keepdim=True) + 1 - list(wffs.size())[-1]
            return torch.max(torch.zeros_like(result, requires_grad=True), result)

        def F_Or(wffs):
            result = torch.sum(wffs, dim=-1, keepdim=True)
            return torch.min(result, torch.ones_like(result, requires_grad=True))

        def F_Implies(wff1, wff2):
            result = 1 - wff1 + wff2
            return torch.min(torch.ones_like(result, requires_grad=True), result)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return 1 - torch.abs(wff1-wff2)


def multi_axes_op(op, input, axes, keepdim=False):
    '''
    Performs `torch.max` over multiple dimensions of `input`
    '''
    if isinstance(axes, int):
        axes = [axes]
    else:
        axes = sorted(axes)
    result = input
    for axis in reversed(axes):
        if op == 'mean':
            result = torch.mean(result, axis, keepdim)
        elif op == 'min':
            result,_ = torch.min(result, axis, keepdim)
        elif op == 'max':
            result,_ = torch.max(result, axis, keepdim)
    return result


def set_universal_aggreg(aggreg):
    assert aggreg in ['hmean','min','mean']
    global F_Forall
    if aggreg == "hmean":
        def F_Forall(axis,wff):
            # return 1 / torch.mean(1/(wff+1e-10), dim=axis)
            return 1 / multi_axes_op('mean', 1/(wff+1e-10), axes=axis)

    if aggreg == "min":
        def F_Forall(axis,wff):
            # return torch.min(wff, dim=axis)
            return multi_axes_op('min', wff, axes=axis)

    if aggreg == "mean":
        def F_Forall(axis,wff):
            # return torch.mean(wff, dim=axis)
            return multi_axes_op('mean', wff, axes=axis)


def set_existential_aggregator(aggreg):
    assert aggreg in ['max']
    global F_Exists
    if aggreg == "max":
        def F_Exists(axis, wff):
            # return torch.max(wff, dim=axis)[0]
            return multi_axes_op('max', wff, axes=axis)


set_tnorm("luk")
set_universal_aggreg("hmean")
set_existential_aggregator("max")


def And(*wffs):
    if len(wffs) == 0:
        result = torch.tensor(1.0, requires_grad=True)
        result.doms = []
    else:
        cross_wffs,_ = cross_args(wffs)
        result = F_And(cross_wffs)
        result.doms = cross_wffs.doms
    return result


def Or(*wffs):
    if len(wffs) == 0:
        result = torch.tensor(0.0, requires_grad=True)
        result.doms = []
    else:
        cross_wffs,_ = cross_args(wffs)
        result = F_Or(cross_wffs)
        result.doms = cross_wffs.doms
    return result


def Implies(wff1, wff2):
    _, cross_wffs = cross_2args(wff1, wff2)
    result = F_Implies(cross_wffs[0], cross_wffs[1])
    result.doms = cross_wffs[0].doms
    return result


def Not(wff):
    result = F_Not(wff)
    result.doms = wff.doms
    return result


def Equiv(wff1,wff2):
    _, cross_wffs = cross_2args(wff1, wff2)
    result = F_Equiv(cross_wffs[0], cross_wffs[1])
    result.doms = cross_wffs[0].doms
    return result


def Forall(vars,wff):
    if type(vars) is not tuple:
        vars = (vars,)
    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
    not_empty_vars = torch.prod(torch.tensor([var.numel() for var in vars])).type(torch.ByteTensor)
    ones = torch.ones((1,)*(len(result_doms)+1), requires_grad=True)
    result = F_Forall(quantif_axis, wff) if not_empty_vars else ones

    result.doms = result_doms
    return result


def Exists(vars,wff):
    if type(vars) is not tuple:
        vars = (vars,)
    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
    not_empty_vars = torch.prod(torch.tensor([var.numel() for var in vars])).type(torch.ByteTensor)
    zeros = torch.zeros((1,) * (len(result_doms) + 1), requires_grad=True)
    result = F_Exists(quantif_axis, wff) if not_empty_vars else zeros
    result.doms = result_doms
    return result


def variable(label, number_of_features_or_feed):
    if isinstance(number_of_features_or_feed, torch.Tensor):
        result = number_of_features_or_feed.clone()
    else:
        result = torch.tensor(number_of_features_or_feed)
    result.doms = [label]
    return result


def constant(label, value=None, min_value=None, max_value=None):
    label = "ltn_constant_" + label
    if value is not None:
        result = torch.tensor([value])
    else:
        result = torch.empty(1, len(min_value)).uniform(min_value, max_value)
    result.label = label
    result.doms = []
    return result


class Function(nn.Module):
    def __init__(self, label, input_shape_spec, output_shape_spec=1,fun_definition=None):
        super(Function, self).__init__()
        self.label = label
        if type(input_shape_spec) is list:
            self.number_of_features = sum([int(v.shape[1]) for v in input_shape_spec])
        elif type(input_shape_spec) is torch.Tensor:
            self.number_of_features = int(input_shape_spec.shape[1])
        else:
            self.number_of_features = input_shape_spec
        self.output_shape_spec = output_shape_spec
        if fun_definition is None:
            self.W = torch.nn.Parameter(torch.rand([self.number_of_features + 1, self.output_shape_spec]))
            def apply_fun(*args):
                tensor_args = torch.cat(args, axis=1)
                self.X = torch.nn.Parameter(torch.cat([torch.ones(tensor_args.size([0], 1))]))
                result = torch.matmul(self.X, self.W)
                return result
            self.pars = [self.W]
        else:
            def apply_fun(*args):
                return fun_definition(*args)
            self.pars = []
        self.apply_fun = apply_fun

    def forward(self, *args):
        crossed_args, list_of_args_in_crossed_args = cross_args(args)
        result = self.apply_fun(*list_of_args_in_crossed_args)
        if crossed_args.doms != []:
            result = torch.reshape(result, list(crossed_args.size())[:-1] + list(result.size())[-1:])
        else:
            result = torch.reshape(result, (self.output_shape_spec,))
        result.doms = crossed_args.doms
        return result

    def reset_parameters(self):
        if self.pars:
            self.W = torch.nn.Parameter(torch.rand([self.number_of_features + 1, self.output_shape_spec]))
            self.pars = [self.W]


class Predicate(nn.Module):
    def __init__(self, label, number_of_features_or_vars, pred_definition=None, layers=2):
        super(Predicate, self).__init__()
        self.label = label
        self.layers = layers
        self.pred_definition = pred_definition
        global BIAS
        if type(number_of_features_or_vars) is list:
            self.number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
        elif type(number_of_features_or_vars) is torch.Tensor:
            self.number_of_features = int(number_of_features_or_vars.shape[1])
        else:
            self.number_of_features = number_of_features_or_vars
        if self.pred_definition is None:
            self.W = torch.nn.Parameter(torch.rand(self.layers, self.number_of_features + 1, self.number_of_features + 1,
                                requires_grad=True))
            self.V = torch.nn.Parameter(torch.rand(self.layers, self.number_of_features + 1, requires_grad=True))
            self.b = torch.nn.Parameter(torch.ones(1, self.layers, requires_grad=True))
            self.u = torch.nn.Parameter(torch.ones(self.layers, 1, requires_grad=True))
            def apply_pred(*args):
                tensor_args = torch.cat(args, dim=1)
                X = torch.cat((torch.ones(tensor_args.size()[0], 1), tensor_args), 1)
                XW = torch.matmul(X.unsqueeze(0).repeat(self.layers, 1, 1), self.W)
                XWX = torch.squeeze(torch.matmul(X.unsqueeze(1), XW.permute(1, 2, 0)), 1)
                XV = torch.matmul(X, torch.t(self.V))
                gX = torch.matmul(torch.tanh(XWX + XV + self.b), self.u)
                return torch.sigmoid(gX)
            self.pars = [self.W, self.V, self.b, self.u]
        else:
            def apply_pred(*args):
                return self.pred_definition(*args)
            self.pars = []
        self.apply_pred = apply_pred

    def forward(self, *args):
        global BIAS
        crossed_args, list_of_args_in_crossed_args = cross_args(args)
        result = self.apply_pred(*list_of_args_in_crossed_args)
        if crossed_args.doms != []:
            result = torch.reshape(result, list(list(crossed_args.size())[:-1]) + [1])
        else:
            result = torch.reshape(result, (1,))
        result.doms = crossed_args.doms
        BIAS = (BIAS + .5 - torch.mean(result)) / 2 * BIAS_factor
        return result

    def reset_parameters(self):
        if self.pars:
            self.W = torch.nn.Parameter(
                torch.rand(self.layers, self.number_of_features + 1, self.number_of_features + 1,
                           requires_grad=True))
            self.V = torch.nn.Parameter(torch.rand(self.layers, self.number_of_features + 1, requires_grad=True))
            self.b = torch.nn.Parameter(torch.ones(1, self.layers, requires_grad=True))
            self.u = torch.nn.Parameter(torch.ones(self.layers, 1, requires_grad=True))
            self.pars = [self.W, self.V, self.b, self.u]


def cross_args(args):
    result = args[0]
    for arg in args[1:]:
        result,_ = cross_2args(result, arg)
    result_flat = torch.reshape(result, (torch.prod(torch.tensor(result.size()[:-1])), result.size()[-1]))
    result_args = torch.split(result_flat, [arg.size()[-1] for arg in args], 1)
    return result, result_args


def cross_2args(X,Y):
    if X.doms == [] and Y.doms == []:
        result = torch.cat([X,Y], dim=-1)
        result.doms = []
        return result,[X,Y]
    X_Y = set(X.doms) - set(Y.doms)
    Y_X = set(Y.doms) - set(X.doms)
    eX = X
    eX_doms = [x for x in X.doms]
    for y in Y_X:
        eX = eX.unsqueeze(0)
        eX_doms = [y] + eX_doms
    eY = Y
    eY_doms = [y for y in Y.doms]
    for x in X_Y:
        eY = eY.unsqueeze(-2)
        eY_doms.append(x)
    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))
    eY = eY.permute(perm_eY + [len(perm_eY)])
    mult_eX = [1]*(len(eX_doms)+1)
    mult_eY = [1]*(len(eY_doms)+1)
    for i in range(len(mult_eX)-1):
        mult_eX[i] = max(1, (eY.size()[i] // eX.size()[i]))
        mult_eY[i] = max(1, (eX.size()[i] // eY.size()[i]))
    result1 = eX.repeat(mult_eX)
    result2 = eY.repeat(mult_eY)
    result = torch.cat([result1, result2], dim=-1)
    result1.doms = eX_doms
    result2.doms = eX_doms
    result.doms = eX_doms
    return result,[result1,result2]