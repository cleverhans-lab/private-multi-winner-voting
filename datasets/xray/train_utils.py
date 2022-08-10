import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join

import numpy as np
import torch
import sklearn.metrics
import sklearn, sklearn.model_selection
from datasets.xray.dataset_pathologies import default_pathologies

from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


# from tqdm.auto import tqdm


def train(model, train_loader, valid_loader, args):
    print("The args:")
    pprint.pprint(args)
    args.model_from_cost_to_timing = args.architecture
    args.name = args.dataset

    dataset_name = args.dataset + "-" + args.model_from_cost_to_timing + "-" + args.name

    args.cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if args.cuda else 'cpu'
    if not torch.cuda.is_available() and args.cuda:
        device = 'cpu'
        print(
            "WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    args.output_dir = 'cxpert_output'
    print(args.output_dir)

    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setting the seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # # Dataset
    # gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,
    #                                                 test_size=0.2,
    #                                                 random_state=args.seed)
    # train_inds, test_inds = next(
    #     gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    # train_dataset = SubsetDataset(dataset, train_inds)
    # valid_dataset = SubsetDataset(dataset, test_inds)

    # Dataloader
    # args.shuffle = True
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=args.shuffle,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=args.cuda)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=args.shuffle,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=args.cuda)
    # print(model)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5,
                             amsgrad=True)
    print(optim)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(
        join(args.output_dir,
             f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(args.output_dir, f'{dataset_name}-e')):-len(
                '.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in
                        np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'),
                  'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)

    for epoch in range(start_epoch, args.num_epochs):

        avg_loss = train_epoch(args=args,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion)

        auc_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)[0]

        if np.mean(auc_valid) > best_metric:
            best_metric = np.mean(auc_valid)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(args.output_dir, f'{dataset_name}-best.pt'))
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric
        }

        metrics.append(stat)

        with open(join(args.output_dir, f'{dataset_name}-metrics.pkl'),
                  'wb') as f:
            pickle.dump(metrics, f)

        torch.save(model,
                   join(args.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    return metrics, best_metric, weights_for_best_validauc


def train_epoch(args, epoch, model, device, train_loader, optimizer, criterion,
                limit=None):
    model.train_features_epoch()

    if args.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)

    losses = []
    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):

        if limit and (batch_idx > limit):
            print("breaking out")
            break

        optimizer.zero_grad()

        # images = samples["img"].float().to(device)
        # targets = samples["lab"].to(device)

        images, targets = samples
        images = images.float().to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = torch.zeros(1).to(device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:, task]
            task_target = targets[:, task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                if args.taskweights:
                    loss += weights[task] * task_loss
                else:
                    loss += task_loss

        # here regularize the weight matrix when label_concat is used
        if args.label_concat_reg:
            if not args.label_concat:
                raise Exception("cfg.label_concat must be true")
            weight = model.classifier.weight
            num_labels = len(default_pathologies)
            num_datasets = weight.shape[0] // num_labels
            weight_stacked = weight.reshape(num_datasets, num_labels, -1)
            label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
            for task in range(num_labels):
                dists = torch.pdist(weight_stacked[:, task], p=2).mean()
                loss += label_concat_reg_lambda * dists

        loss = loss.sum()

        if args.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()

        if args.weightreg:
            loss += model.classifier.weight.abs().sum()

        loss.backward()

        losses.append(loss.detach().cpu().numpy())
        t.set_description(
            f'Epoch {epoch + 1} - Train - Loss = {np.mean(losses):4.4f}')

        optimizer.step()

    return np.mean(losses)


def valid_test_epoch(name, epoch, model, device, data_loader, criterion,
                     limit=None):
    model.eval()

    avg_loss = []
    task_outputs = {}
    task_targets = {}
    for task in range(data_loader.dataset.labels.shape[1]):
        task_outputs[task] = []
        task_targets[task] = []

    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break

            # images = samples["img"].to(device)
            # targets = samples["lab"].to(device)

            images, targets = samples
            images = images.float().to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:, task]
                task_target = targets[:, task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(),
                                      task_target.double())

                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()

            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(
                f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')

        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])

        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task])) > 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task],
                                                         task_outputs[task])
                # print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets
