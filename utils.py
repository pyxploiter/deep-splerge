import torch
import torch.nn.functional as F

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

def splerge_loss(outputs, targets):
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
    
    # r3_logits = get_logits(r3)
    # r4_logits = get_logits(r4)
    # r5_logits = get_logits(r5)
    
    # c3_logits = get_logits(c3)
    # c4_logits = get_logits(c4)
    # c5_logits = get_logits(c5)

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

    # rl3 = cross_entropy_loss(r3_logits, rpn_targets)
    # rl4 = cross_entropy_loss(r4_logits, rpn_targets)
    # rl5 = cross_entropy_loss(r5_logits, rpn_targets)

    # cl3 = cross_entropy_loss(c3_logits, cpn_targets)
    # cl4 = cross_entropy_loss(c4_logits, cpn_targets)
    # cl5 = cross_entropy_loss(c5_logits, cpn_targets)

    rpn_loss = rl5 + (lambda4 * rl4) + (lambda3 * rl3)
    cpn_loss = cl5 + (lambda4 * cl4) + (lambda3 * cl3)
    
    loss = rpn_loss + cpn_loss

    return loss, rpn_loss, cpn_loss

def collate_fn(batch):
    return tuple(zip(*batch))
