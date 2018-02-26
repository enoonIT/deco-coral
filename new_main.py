from __future__ import division
import argparse

import time

import itertools
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils import model_zoo
from torch.autograd import Variable

import old_models
import models
import utils
from data_loader import get_train_test_loader, get_office31_dataloader
from logger import Logger

CUDA = True if torch.cuda.is_available() else False
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9


def train(model, optimizer, epoch, _lambda, deco_lambda=1e-3):
    model.train()

    result = []

    # Expected size : xs -> (batch_size, 3, 300, 300), ys -> (batch_size)
    # source, target = list(enumerate(source_loader)), list(enumerate(target_loader))
    train_steps = len(source_loader)

    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, itertools.cycle(target_loader))):
        source_data, source_label = source_batch
        target_data, _ = target_batch
        if CUDA:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()
        out1, out2, deco_norm = model(source_data, target_data)

        classification_loss = torch.nn.functional.cross_entropy(out1, source_label)
        coral_loss = old_models.CORAL(out1, out2)

        sum_loss = _lambda * coral_loss + classification_loss + deco_lambda * deco_norm
        sum_loss.backward()

        optimizer.step()

        result.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': _lambda,
            'coral_loss': coral_loss.data[0],
            'classification_loss': classification_loss.data[0],
            'total_loss': sum_loss.data[0],
            'deco_norm': deco_norm
        })

        if batch_idx % 3 == 0:
            print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
                  'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}. ' \
                  'Image norm:{:.4f}, Deco norm {:.4f}'.format(
                epoch,
                batch_idx + 1,
                train_steps,
                _lambda,
                classification_loss.data[0],
                coral_loss.data[0],
                sum_loss.data[0],
                source_data.norm().data[0] / source_data.shape[0] + target_data.norm().data[0] / target_data.shape[0],
                deco_norm.data[0]
            ))

    return result


def test(model, dataset_loader, e, mode='source'):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataset_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        out1, out2, _ = model(data, data)

        out = out1 if mode == 'source' else out2

        # sum up batch loss
        test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).data[0]

        # get the index of the max log-probability
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataset_loader.dataset)

    return {
        'epoch': e,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * correct / len(dataset_loader.dataset)
    }


# load AlexNet pre-trained model
def load_pretrained(model):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()

    # filter out unmatch dict and delete last fc bias, weight
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # del pretrained_dict['classifier.6.bias']
    # del pretrained_dict['classifier.6.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Resume from checkpoint file')
    parser.add_argument('--log_subdir', default="")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--extra', help="appended to log name", default="")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lambda_val', default=0.6, type=float)
    parser.add_argument('--deco_weight', default=0.001, type=float)
    parser.add_argument('--deco_lambda', default=1e-3, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    LEARNING_RATE = args.lr
    BATCH_SIZE = [args.batch_size, args.batch_size]
    EPOCHS = args.epochs
    STEP_DOWN = int(EPOCHS * 0.4)
    GAMMA = 0.2

    source_loader = get_office31_dataloader(case='amazon', batch_size=BATCH_SIZE[0])
    target_loader = get_office31_dataloader(case='webcam', batch_size=BATCH_SIZE[1])

    model = models.DeepColorizationCORAL(31, args.deco_weight)
    lambda_val = args.lambda_val
    extra = args.extra
    if lambda_val is not None:
        extra += "lambda_%g" % lambda_val
    else:
        extra += "growing_lambda"
    name = "bs%d_lr%g_e%d_%s_%d" % (BATCH_SIZE[0], LEARNING_RATE, EPOCHS, extra, int(time.time()) % 100)
    logger = Logger("logs/%s%s" % (args.log_subdir, name))
    # support different learning rate according to CORAL paper
    # i.e. 10 times learning rate for the last two fc layers.
    optimizer = torch.optim.SGD([
        {'params': model.deco.parameters(), 'lr': LEARNING_RATE},
        {'params': model.source_fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.target_fc.parameters(), 'lr': LEARNING_RATE}
    ], lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=STEP_DOWN, gamma=GAMMA)

    if CUDA:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
    if args.load is not None:
        utils.load_net(model, args.load)
    else:
        load_pretrained(model.sharedNet)

    training_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []
    print(scheduler.get_lr())
    start = time.time()
    for e in range(0, EPOCHS):
        scheduler.step()
        print(scheduler.get_lr())
        if lambda_val is not None:
            _lambda = lambda_val
        else:
            _lambda = (e + 1) / EPOCHS

        res = train(model, optimizer, e + 1, _lambda, args.deco_lambda)
        print('###EPOCH {}: Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
            e + 1,
            sum(row['classification_loss'] / row['total_steps'] for row in res),
            sum(row['coral_loss'] / row['total_steps'] for row in res),
            sum(row['total_loss'] / row['total_steps'] for row in res),
        ))

        training_statistic.append(res)

        test_source = test(model, source_loader, e)
        test_target = test(model, target_loader, e, mode='target')
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)

        print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            test_source['average_loss'],
            test_source['correct'],
            test_source['total'],
            test_source['accuracy'],
        ))
        print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            test_target['average_loss'],
            test_target['correct'],
            test_target['total'],
            test_target['accuracy'],
        ))
        logger.scalar_summary("loss/coral", sum(row['coral_loss'] / row['total_steps'] for row in res), e + 1)
        logger.scalar_summary("loss/source", sum(row['classification_loss'] / row['total_steps'] for row in res), e + 1)
        logger.scalar_summary("loss/deco_norm", sum(row['deco_norm'] / row['total_steps'] for row in res), e + 1)
        logger.scalar_summary("acc/target", test_target['accuracy'], e + 1)
        logger.scalar_summary("acc/source", test_source['accuracy'], e + 1)
        logger.scalar_summary("lr", scheduler.get_lr()[0], e + 1)
        logger.scalar_summary("weights/lambda", _lambda, e + 1)
    print("It took %g seconds" % (time.time() - start))
    utils.save(training_statistic, 'training_statistic.pkl')
    utils.save(testing_s_statistic, 'testing_s_statistic.pkl')
    utils.save(testing_t_statistic, 'testing_t_statistic.pkl')
    utils.save_net(model, 'models/checkpoint_%s.tar' % name)
