import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    """
    Arguments:
    ----------
    logits: (N, num_classes)
    targets: (N)
    """
    # print(logits.shape, targets.shape)
    # print(torch.abs(x-y))
    log_prob = -1.0 * F.log_softmax(logits, 1)
    loss = log_prob.gather(2, targets.unsqueeze(2))
    loss = loss.mean()
    return loss

def split_loss(outputs, targets):
    """
    Arguments:
    ----------
    outputs: (rpn_outputs, cpn_outputs)
    targets: (rpn_targets, cpn_targets)
    """
    
    lambda3 = 0.1
    lambda4 = 0.25

    rpn_outputs, cpn_outputs = outputs
    rpn_targets, cpn_targets = targets

    r3, r4, r5 = rpn_outputs
    c3, c4, c5 = cpn_outputs
    

    # TODO: update cross entropy loss acc. to the paper
    crit = torch.nn.BCELoss()
    rpn_targets = rpn_targets.float()
    cpn_targets = cpn_targets.float()
    
    # print(rpn_targets.shape, r3.shape, r3.squeeze(1).shape)
    rl3 = crit(r3.squeeze(1), rpn_targets)
    rl4 = crit(r4.squeeze(1), rpn_targets)
    rl5 = crit(r5.squeeze(1), rpn_targets)

    cl3 = crit(c3.squeeze(1), cpn_targets)
    cl4 = crit(c4.squeeze(1), cpn_targets)
    cl5 = crit(c5.squeeze(1), cpn_targets)

    rpn_loss = rl5 + (lambda4 * rl4) + (lambda3 * rl3)
    cpn_loss = cl5 + (lambda4 * cl4) + (lambda3 * cl3)
    
    loss = rpn_loss + cpn_loss

    return loss, rpn_loss, cpn_loss


def merge_loss(outputs, targets, weights=[0.25, 0.75]):
    D1_probs, D2_probs, R1_probs, R2_probs = outputs

    D_targets, R_targets = targets
    D_targets = D_targets.unsqueeze(0)
    R_targets = R_targets.unsqueeze(0)

    # crit = torch.nn.BCELoss()
    # dl2 = crit(D1_probs.squeeze(1), D_targets)
    # dl3 = crit(D2_probs.squeeze(1), D_targets)
    
    # rl2 = crit(R1_probs.squeeze(1), R_targets)
    # rl3 = crit(R2_probs.squeeze(1), R_targets)

    dl2 = weighted_binary_cross_entropy(D1_probs.squeeze(1), D_targets, weights=weights)
    dl3 = weighted_binary_cross_entropy(D2_probs.squeeze(1), D_targets, weights=weights)
    
    rl2 = weighted_binary_cross_entropy(R1_probs.squeeze(1), R_targets, weights=weights)
    rl3 = weighted_binary_cross_entropy(R2_probs.squeeze(1), R_targets, weights=weights)

    if torch.isnan(dl2):
        dl2.data = torch.tensor(1e-8).data
        print("WARNING: dl2_loss is nan. Resetting it to 0.0\n")

    if torch.isnan(dl3): 
        dl3.data = torch.tensor(1e-8).data
        print("WARNING: dl3_loss is nan. Resetting it to 0.0\n")

    if torch.isnan(rl2): 
        rl2.data = torch.tensor(1e-8).data
        print("WARNING: rl2_loss is nan. Resetting it to 1e-8\n")
    
    if torch.isnan(rl3): 
        rl3.data = torch.tensor(1e-8).data
        print("WARNING: rl3_loss is nan. Resetting it to 1e-8\n")

    lambda2 = 0.25
    down_loss = dl3 + (lambda2 * dl2)
    right_loss = rl3 + (lambda2 * rl2)

    loss = down_loss + right_loss

    return loss, down_loss, right_loss

def get_logits(sig_probs):
    """
    Arguments:
    ----------
    sig_probs: output sigmoid probs from model
    """

    pos = sig_probs.squeeze(dim=0).view(sig_probs.shape[0],sig_probs.shape[2],1)
    neg = torch.sub(1, sig_probs.squeeze(dim=0)).view(sig_probs.shape[0],sig_probs.shape[2],1)
    logits = torch.cat((pos,neg),2)
    return logits


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


