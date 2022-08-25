def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    ''''
    # warmup_epoch = -1
    if iteration <= pow(10, 4):
        lr = initial_lr
    elif iteration > 1.5*pow(10, 4):
        lr = initial_lr / 100
    else:
        lr = initial_lr / 10
    '''

    if iteration <= 4*pow(10, 2):
        lr = 1 * 1e-3
    elif iteration > 1.2 * pow(10, 3):
        lr = 5 * 1e-4
    else:
        lr = 1 * 1e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr