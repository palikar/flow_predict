#!/usr/bin/env python



def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('--learning rate = %.7f' % lr)
