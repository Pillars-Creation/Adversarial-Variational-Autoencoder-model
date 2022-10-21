import os
import copy
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from data import MovieLens1MColdStartDataLoader, MovieLens25MColdStartDataLoader, TaobaoADColdStartDataLoader
from model import FactorizationMachineModel, WideAndDeep, DeepFactorizationMachineModel, AdaptiveFactorizationNetwork, \
    ProductNeuralNetworkModel
from model import AttentionalFactorizationMachineModel, DeepCrossNetworkModel, MWUF, MetaE, CVAR
from model.wd import WideAndDeep
from model.cvaegan_update import CVAEGAN
from model.gan import GAN

import pdb


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_model_path', default='./pretrain_backbones')
    parser.add_argument('--datahub_path', default='./datahub/')
    parser.add_argument('--dataset_name', default='movielens1M',
                        help='required to be one of [movielens1M, movielens25M, taobaoAD]')
    # parser.add_argument('--dataset_name', default='movielens1M', help='required to be one of [movielens1M, taobaoAd]')
    # parser.add_argument('--dataset_path', default='./datahub/movielens1M/emb_warm_split_preprocess_ml-1M.pkl')
    # parser.add_argument('--dataset_path',
    #                     default='./datahub/taobaoAd/cold_start/emb_warm_split_preprocess_taobao-ad.pkl')
    parser.add_argument('--warmup_model', default='gan',
                        help="required to be one of [base, mwuf, metaE, cvar, cvaegan, gan]")
    parser.add_argument('--save_pretrain_model', type=bool, default=True, )
    parser.add_argument('--is_dropoutnet', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--model_name', default='ipnn',
                        help='backbone name, we implemented [fm, wd, deepfm, afn, ipnn, opnn, afm, dcn]')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--cvar_epochs', type=int, default=1)
    parser.add_argument('--cvar_iters', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args


def get_loaders(name, datahub_path, device, bsz, shuffle):
    path = os.path.join(datahub_path, name, "{}_data.pkl".format(name))
    if name == 'movielens1M':
        dataloaders = MovieLens1MColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    elif name == 'movielens25M':
        dataloaders = MovieLens25MColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    elif name == 'taobaoAD':
        dataloaders = TaobaoADColdStartDataLoader(name, path, device, bsz=bsz, shuffle=shuffle)
    else:
        raise ValueError('unkown dataset name: {}'.format(name))
    return dataloaders


def get_model(name, dl):
    if name == 'fm':
        return FactorizationMachineModel(dl.description, 16)
    elif name == 'wd':
        return WideAndDeep(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'deepfm':
        return DeepFactorizationMachineModel(dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(dl.description, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400),
                                            dropout=0)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16,), dropout=0, method='inner')
    elif name == 'opnn':
        return ProductNeuralNetworkModel(dl.description, embed_dim=16, mlp_dims=(16,), dropout=0, method='outer')
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(dl.description, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'dcn':
        return DeepCrossNetworkModel(dl.description, embed_dim=16, num_layers=3, mlp_dims=[16, 16], dropout=0.2)
    return


def test(model, data_loader, device):
    model.eval()
    labels, scores, predicts = list(), list(), list()
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for _, (features, label) in enumerate(data_loader):
            features = {key: value.to(device) for key, value in features.items()}
            label = label.to(device)
            y = model(features)
            labels.extend(label.tolist())
            scores.extend(y.tolist())
    scores_arr = np.array(scores)
    return roc_auc_score(labels, scores), f1_score(labels,
                                                   (scores_arr > np.mean(scores_arr)).astype(np.float32).tolist())


def dropoutNet_train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10,
                     val_data_loader=None):
    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                 lr=lr, weight_decay=weight_decay)
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader)
        for i, (features, label) in enumerate(data_loader):
            if random.random() < 0.1:
                bsz = label.shape[0]
                item_emb = model.emb_layer['item_id']
                origin_item_emb = item_emb(features['item_id']).squeeze(1)
                mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                    .repeat(bsz, 1)
                y = model.forward_with_item_id_emb(mean_item_emb, features)
            else:
                y = model(features)
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print("    iters {}/{} loss: {:.4f}".format(i + 1, total_iters + 1, total_loss / log_interval),
                      end='\r')
                total_loss = 0

        print("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss / total_iters), " " * 20)
    return


def train(model, data_loader, device, epoch, lr, weight_decay, save_path, log_interval=10, val_data_loader=None):
    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), \
                                 lr=lr, weight_decay=weight_decay)

    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader)
        for i, (features, label) in enumerate(data_loader):
            y = model(features)
            item_ids = features[model.item_id_name]
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
#             if i < 1000:
#                 emb = model.emb_layer[model.item_id_name]
#                 emb2 = model.emb_layer[model.item_id_name](item_ids)
#                 model.immature_emb[model.item_id_name] = model.emb_layer[model.item_id_name]
#                 # item_id_emb = self.emb_layer[model.item_id_name](features[model.item_id_name])
            if (i + 1) % log_interval == 0:
                print("    Iter {}/{} loss: {:.4f}".format(i + 1, total_iters + 1, total_loss / log_interval), end='\r')
                total_loss = 0
    return


def pretrain(dataset_name,
             datahub_name,
             bsz,
             shuffle,
             model_name,
             epoch,
             lr,
             weight_decay,
             device,
             save_dir,
             is_dropoutnet=False):
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("get loader before")
    dataloaders = get_loaders(dataset_name, datahub_name, device, bsz, shuffle == 1)
    print("get model before")
    model = get_model(model_name, dataloaders).to(device)
    print("get model after")
    save_path = os.path.join(save_dir, 'model.pth')
    print("=" * 20, 'pretrain {}'.format(model_name), "=" * 20)
    # init parameters
    model.init()
    # pretrain
    if is_dropoutnet:
        dropoutNet_train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path,
                         val_data_loader=dataloaders['test'])
    else:
        train(model, dataloaders['train_base'], device, epoch, lr, weight_decay, save_path,
              val_data_loader=dataloaders['test'])
    print("=" * 20, 'pretrain {}'.format(model_name), "=" * 20)
    return model, dataloaders


def base(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    print("*" * 20, "base", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    save_path = os.path.join(save_dir, 'model.pth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # data set list
    auc_list = []
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    for i, train_s in enumerate(dataset_list):
        auc, f1 = test(model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[base model] evaluate on [test dataset] auc: {:.4f}, F1 socre: {:.4f}".format(auc, f1))
        if i < 3:
            model.only_optimize_itemid()
            train(model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*" * 20, "base", "*" * 20)
    return auc_list


def metaE(model,
          dataloaders,
          model_name,
          epoch,
          lr,
          weight_decay,
          device,
          save_dir):
    print("*" * 20, "metaE", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    metaE_model = MetaE(model, warm_features=dataloaders.item_features, device=device).to(device)
    # fetch data
    metaE_dataloaders = [dataloaders[name] for name in ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']]
    # train meta embedding generator
    metaE_model.train()
    criterion = torch.nn.BCELoss()
    metaE_model.optimize_metaE()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, metaE_model.parameters()), \
                                 lr=lr, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        dataloader_a = metaE_dataloaders[epoch_i]
        dataloader_b = metaE_dataloaders[(epoch_i + 1) % 4]
        epoch_loss = 0.0
        total_iter_num = len(dataloader_a)
        iter_dataloader_b = iter(dataloader_b)
        for i, (features_a, label_a) in enumerate(dataloader_a):
            features_b, label_b = next(iter_dataloader_b)
            loss_a, target_b = metaE_model(features_a, label_a, features_b, criterion)
            loss_b = criterion(target_b, label_b.float())
            loss = 0.1 * loss_a + 0.9 * loss_b
            metaE_model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if (i + 1) % 10 == 0:
                print("    iters {}/{}, loss: {:.4f}, loss_a: {:.4f}, loss_b: {:.4f}".format(i + 1, int(total_iter_num),
                                                                                             loss, loss_a, loss_b),
                      end='\r')
        print("Epoch {}/{} loss: {:.4f}".format(epoch_i, epoch, epoch_loss / total_iter_num), " " * 100)
    # replace item id embedding with warmed itemid embedding
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = metaE_model.model.emb_layer[metaE_model.item_id_name].weight.data
        warm_item_id_emb = metaE_model.warm_item_id(features)
        indexes = features[metaE_model.item_id_name].squeeze()
        origin_item_id_emb[indexes,] = warm_item_id_emb
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(metaE_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[metaE] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            metaE_model.model.only_optimize_itemid()
            train(metaE_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*" * 20, "metaE", "*" * 20)
    return auc_list


def mwuf(model,
         dataloaders,
         model_name,
         epoch,
         lr,
         weight_decay,
         device,
         save_dir):
    print("*" * 20, "mwuf", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train mwuf
    mwuf_model = MWUF(model,
                      item_features=dataloaders.item_features,
                      train_loader=train_base,
                      device=device).to(device)

    mwuf_model.init_meta()
    mwuf_model.train()
    criterion = torch.nn.BCELoss()
    mwuf_model.optimize_new_item_emb()
    optimizer1 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                  lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_meta()
    optimizer2 = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mwuf_model.parameters()), \
                                  lr=lr, weight_decay=weight_decay)
    mwuf_model.optimize_all()
    total_iters = len(train_base)
    loss_1, loss_2 = 0.0, 0.0
    for i, (features, label) in enumerate(train_base):
        # if i + 1 > total_iters * 0.3:
        #     break
        y_cold = mwuf_model.cold_forward(features)
        cold_loss = criterion(y_cold, label.float())
        mwuf_model.zero_grad()
        cold_loss.backward()
        optimizer1.step()
        y_warm = mwuf_model.forward(features)
        warm_loss = criterion(y_warm, label.float())
        mwuf_model.zero_grad()
        warm_loss.backward()
        optimizer2.step()
        loss_1 += cold_loss
        loss_2 += warm_loss
        if (i + 1) % 10 == 0:
            print("    iters {}/{}  warm loss: {:.4f}" \
                  .format(i + 1, int(total_iters), \
                          warm_loss.item()), end='\r')
    print("final average warmup loss: cold-loss: {:.4f}, warm-loss: {:.4f}"
          .format(loss_1 / total_iters, loss_2 / total_iters))
    # use trained meta scale and shift to initialize embedding of new items
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = mwuf_model.model.emb_layer[mwuf_model.item_id_name].weight.data
        warm_item_id_emb = mwuf_model.warm_item_id(features)
        indexes = features[mwuf_model.item_id_name].squeeze()
        origin_item_id_emb[indexes,] = warm_item_id_emb
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(mwuf_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[mwuf] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            mwuf_model.model.only_optimize_itemid()
            train(mwuf_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    print("*" * 20, "mwuf", "*" * 20)
    return auc_list


def cvar(model,
         dataloaders,
         model_name,
         epoch,
         cvar_epochs,
         cvar_iters,
         lr,
         weight_decay,
         device,
         save_dir):
    print("*" * 20, "cvar", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train cvar
    warm_model = CVAR(model,
                      warm_features=dataloaders.item_features,
                      train_loader=train_base,
                      device=device).to(device)
    warm_model.init_cvar()

    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        warm_model.optimize_cvar()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, warm_model.parameters()), \
                                     lr=lr, weight_decay=weight_decay)
        batch_num = len(dataloader)
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                a, b, c, d = 0.0, 0.0, 0.0, 0.0
                for _ in range(iters):
                    target, recon_term, reg_term = warm_model(features)
                    main_loss = criterion(target, label.float())
                    loss = main_loss + recon_term + 1e-4 * reg_term
                    warm_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    a, b, c, d = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item()
                a, b, c, d = a / iters, b / iters, c / iters, d / iters
                if logger and (i + 1) % 10 == 0:
                    print("    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}" \
                          .format(i + 1, batch_num, a, b, c, d), end='\r')
        # warm-up item id embedding
        train_a = dataloaders['train_warm_a']
        for (features, label) in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data

            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes,] = warm_item_id_emb

    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[cvar] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    print("*" * 20, "cvar", "*" * 20)
    return auc_list


def cvaegan(model,
            dataloaders,
            model_name,
            epoch,
            cvar_epochs,
            cvar_iters,
            lr,
            weight_decay,
            device,
            save_dir):
    print("*" * 20, "cvaegan", "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    # train cvar
    warm_model = CVAEGAN(model,
                         warm_features=dataloaders.item_features,
                         train_loader=train_base,
                         device=device).to(device)
    warm_model.init_cvaegan()

    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        warm_model.optimizer_cvaegan()

        optimizer_main = torch.optim.Adam(params=[kv[1] for kv in warm_model.named_parameters() if
                                                  kv[1].requires_grad and 'discriminator' not in kv[0]], \
                                          lr=lr, weight_decay=weight_decay)  # except D

        optimizer_d = torch.optim.Adam(params=warm_model.discriminator.parameters(), \
                                       lr=lr, weight_decay=weight_decay)

        batch_num = len(dataloader)
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                a, b, c, d, e = 0.0, 0.0, 0.0, 0.0, 0.0
                for _ in range(iters):
                    bsz = label.shape[0]
                    adv_criterion = torch.nn.BCELoss().to(device)
                    real_label = torch.ones((bsz, 1)).to(device)
                    fake_label = torch.zeros((bsz, 1)).to(device)

                    ## train ae and model

                    target, recon_term, reg_term, warm_id_emb = warm_model(features)

                    # recon_loss, reg_loss, target, warm_id_emb = warm_model(features)
                    pred_adv = warm_model.discriminator(warm_id_emb)
                    adv_loss_G = adv_criterion(pred_adv, real_label)  # make D think it is real
                    main_loss = criterion(target, label.float())

                    pred_adv = warm_model.discriminator(warm_id_emb)
                    adv_loss_G = adv_criterion(pred_adv, real_label)  # make D think it is real

                    # get matching loss
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()

                    for j in range(2):  # len(warm_model.discriminator) - 1
                        real_input = warm_model.discriminator[j](real_input)

                    warm_id_emb_input = warm_id_emb
                    for j in range(2):
                        warm_id_emb_input = warm_model.discriminator[j](warm_id_emb_input)

                    LG_2 = torch.mean(torch.square(warm_id_emb_input - real_input))
                    LGD = torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))

                    loss = main_loss + recon_term + 1e-4 * reg_term + 0.01 * adv_loss_G  # + 0.1 * LG_2 + 0.1 * LGD#+ 0.01 * adv_loss_G

                    warm_model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_main.step()

                    ## train discriminator
                    for inner_iter in range(3):
                        _, _, _, warm_id_emb = warm_model(features)

                        item_ids = features[warm_model.item_id_name]
                        item_id_emb = warm_model.origin_item_emb(item_ids).squeeze()
                        real_item_emb = item_id_emb

                        real_pred = warm_model.discriminator(real_item_emb)
                        fake_pred = warm_model.discriminator(warm_id_emb)  # geneated is fake

                        # print(real_pred.requires_grad)
                        adv_loss = (adv_criterion(fake_pred, fake_label) + adv_criterion(real_pred, real_label)) / 2

                        optimizer_d.zero_grad()
                        adv_loss.backward(retain_graph=True)
                        optimizer_d.step()

                    a, b, c, d, e = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item(), e + adv_loss_G
                a, b, c, d, e = a / iters, b / iters, c / iters, d / iters, e / iters
                if logger and (i + 1) % 10 == 0:
                    print(
                        "    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, adv_loss_G: {:.4f}" \
                        .format(i + 1, batch_num, a, b, c, d, e), end='\r')
                    print("loss = main_loss + recon_term + 1e-4 * reg_term + 0.1 * LG_2 + 0.1 * LGD")
        # warm-up item id embedding
        train_a = dataloaders['train_warm_a']
        for (features, label) in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data

            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes,] = warm_item_id_emb

    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []

    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[cvaegan] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    print("*" * 20, "cvaegan", "*" * 20)
    return auc_list


def cvaegan_c(model,
              dataloaders,
              model_name,
              epoch,
              cvar_epochs,
              cvar_iters,
              lr,
              weight_decay,
              device,
              save_dir, dataset_name):
    print("*" * 20, "cvaegan_c" + model_name, "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    if dataset_name == 'movielens1M':
        c_feature = 'genres'
        c_class_size = 19
    elif dataset_name == 'taobaoAD':
        c_feature = 'cate_id'
        c_class_size = 5000
    elif dataset_name == 'movielens25M':
        c_feature = 'genres'
        c_class_size = 20

    # train cvar
    warm_model = CVAEGAN(model,
                         warm_features=dataloaders.item_features,
                         train_loader=train_base,
                         device=device, c_feature=c_feature, c_class_size=c_class_size).to(device)
    warm_model.init_cvaegan()

    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        CE_loss = torch.nn.CrossEntropyLoss()
        cce_loss = tf.keras.losses.CategoricalCrossentropy()

        warm_model.optimizer_cvaegan()

        optimizer_main = torch.optim.Adam(params=[kv[1] for kv in warm_model.named_parameters() if
                                                  kv[1].requires_grad and 'discriminator' not in kv[0]], \
                                          lr=lr, weight_decay=weight_decay)  # except D

        optimizer_d = torch.optim.Adam(params=[p for p in warm_model.discriminator.parameters()] + [p for p in
                                                                                                    warm_model.condition_discriminator.parameters()], \
                                       lr=lr, weight_decay=weight_decay)

        batch_num = len(dataloader)
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                #                 pdb.set_trace()
                a, b, c, d, e, f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for _ in range(iters):
                    bsz = label.shape[0]
                    adv_criterion = torch.nn.BCELoss().to(device)
                    real_label = torch.ones((bsz, 1)).to(device)
                    fake_label = torch.zeros((bsz, 1)).to(device)

                    ## train ae and model
                    target, recon_term, reg_term, warm_id_emb = warm_model(features)

                    pred_adv = warm_model.discriminator(warm_id_emb)
                    loss_D = adv_criterion(pred_adv, real_label)  # make D think it is real
                    main_loss = criterion(target, label.float())

                    # get Conditional loss LC
                    pred_class = warm_model.condition_discriminator(warm_id_emb).to(device)
                    class_label = features[c_feature][:, [0]]
                    #                     class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                    #                     print (class_label)
                    #                     print(pred_class)
                    loss_C = CE_loss(pred_class, class_label.squeeze())

                    # get matching loss LGC
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()

                    matching_layer = len(warm_model.condition_discriminator) - 1
                    for j in range(matching_layer):
                        real_input = warm_model.condition_discriminator[j](real_input)

                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.condition_discriminator[j](warm_id_emb_input)

                    loss_matching_GC = 0.5 * torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))
                    #                     loss_matching_GC = 0
                    #                     for j in range(18):
                    #                         iclass_idx = torch.equal(class_label, j*torch.ones_like(class_label))
                    #                         loss_matching_GC += torch.square(torch.mean(warm_id_emb_input[iclass_idx]) - torch.mean(real_input[iclass_idx]))
                    loss_L2_C = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get matching loss LGD
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()
                    for j in range(matching_layer):
                        real_input = warm_model.discriminator[j](real_input)

                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.discriminator[j](warm_id_emb_input)
                    loss_matching_GD = 0.5 * torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))

                    loss_L2_D = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get l2 reconstruction loss
                    target, recon_term, reg_term, warm_id_emb = warm_model(features)
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()

                    L2_loss = loss_L2_D + loss_L2_C + 0.5 * (torch.mean(torch.square(warm_id_emb - real_input)))

                    loss = main_loss + recon_term + 1e-4 * reg_term + 0.1 * L2_loss \
                           + 0.1 * loss_matching_GC + 0.1 * loss_matching_GD

                    warm_model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_main.step()

                    ## train discriminator
                    for inner_iter in range(3):
                        _, _, _, warm_id_emb = warm_model(features)

                        item_ids = features[warm_model.item_id_name]
                        item_id_emb = warm_model.origin_item_emb(item_ids).squeeze()
                        real_item_emb = item_id_emb

                        real_pred = warm_model.discriminator(real_item_emb)
                        fake_pred = warm_model.discriminator(warm_id_emb)  # geneated is fake
                        class_label = features[c_feature][:, [0]]
                        #                         class_label = torch.floor(features[c_feature]*c_class_size%c_class_size).long().to(device)
                        #                         class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                        pred_class = warm_model.condition_discriminator(warm_id_emb)
                        condition_loss = CE_loss(pred_class, class_label.squeeze())

                        # print(real_pred.requires_grad)
                        adv_loss = (adv_criterion(fake_pred, fake_label) + adv_criterion(real_pred, real_label)) / 2
                        adv_loss += condition_loss

                        optimizer_d.zero_grad()
                        adv_loss.backward(retain_graph=True)
                        optimizer_d.step()

                    a, b, c, d, e, f = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item() \
                        , e + adv_loss.item(), f + condition_loss.item()
                a, b, c, d, e, f = a / iters, b / iters, c / iters, d / iters, e / iters, f / iters
                if logger and (i + 1) % 10 == 0:
                    print("    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, adv_loss_G: {:.4f}\
                         , condition_loss: {:.4f}".format(i + 1, batch_num, a, b, c, d, e, f), end='\r')
        # warm-up item id embedding
        train_a = dataloaders['train_warm_a']
        for (features, label) in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data

            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes,] = warm_item_id_emb

    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []

    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[cvaegan] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    print("*" * 20, "cvaegan_c_" + model_name, "*" * 20)
    return auc_list

def gan(model,
              dataloaders,
              model_name,
              epoch,
              cvar_epochs,
              cvar_iters,
              lr,
              weight_decay,
              device,
              save_dir, dataset_name):
    print("*" * 20, "gan_c_" + model_name, "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    if dataset_name == 'movielens1M':
        c_feature = 'genres'
        c_class_size = 19
    elif dataset_name == 'taobaoAD':
        c_feature = 'cate_id'
        c_class_size = 5000
    elif dataset_name == 'movielens25M':
        c_feature = 'genres'
        c_class_size = 20

    # train gan
    warm_model = GAN(model,
                         warm_features=dataloaders.item_features,
                         train_loader=train_base,
                         device=device, c_feature=c_feature, c_class_size=c_class_size).to(device)
    warm_model.init_cvaegan()

    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        CE_loss = torch.nn.CrossEntropyLoss()

        warm_model.optimizer_cvaegan()

        optimizer_main = torch.optim.Adam(params=[kv[1] for kv in warm_model.named_parameters() if
                                                  kv[1].requires_grad and 'discriminator' not in kv[0]], \
                                          lr=lr, weight_decay=weight_decay)  # except D

        optimizer_d = torch.optim.Adam(params=[p for p in warm_model.discriminator.parameters()] + [p for p in
                                                                                                    warm_model.condition_discriminator.parameters()], \
                                       lr=lr, weight_decay=weight_decay)

        batch_num = len(dataloader)
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                #                 pdb.set_trace()
                a, b, c, d, e, f = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for _ in range(iters):
                    bsz = label.shape[0]
                    adv_criterion = torch.nn.BCELoss().to(device)
                    real_label = torch.ones((bsz, 1)).to(device)
                    fake_label = torch.zeros((bsz, 1)).to(device)

                    ## train ae and model
                    target, recon_term, reg_term, warm_id_emb = warm_model(features)

                    pred_adv = warm_model.discriminator(warm_id_emb)
                    loss_D = adv_criterion(pred_adv, real_label)  # make D think it is real
                    main_loss = criterion(target, label.float())

                    # get Conditional loss LC
                    pred_class = warm_model.condition_discriminator(warm_id_emb).to(device)
                    class_label = features[c_feature][:, [0]]
                    #                     class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                    #                     print (class_label)
                    #                     print(pred_class)
                    loss_C = CE_loss(pred_class, class_label.squeeze())

                    # get matching loss LGC
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()

                    matching_layer = len(warm_model.condition_discriminator) - 1
                    for j in range(matching_layer):
                        real_input = warm_model.condition_discriminator[j](real_input)

                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.condition_discriminator[j](warm_id_emb_input)

                    loss_matching_GC = 0.5 * torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))
                    #                     loss_matching_GC = 0
                    #                     for j in range(18):
                    #                         iclass_idx = torch.equal(class_label, j*torch.ones_like(class_label))
                    #                         loss_matching_GC += torch.square(torch.mean(warm_id_emb_input[iclass_idx]) - torch.mean(real_input[iclass_idx]))
                    loss_L2_C = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get matching loss LGD
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()
                    for j in range(matching_layer):
                        real_input = warm_model.discriminator[j](real_input)

                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.discriminator[j](warm_id_emb_input)
                    loss_matching_GD = 0.5 * torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))

                    loss_L2_D = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get l2 reconstruction loss
                    target, recon_term, reg_term, warm_id_emb = warm_model(features)
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()

                    L2_loss = loss_L2_D + loss_L2_C + 0.5 * (torch.mean(torch.square(warm_id_emb - real_input)))

                    loss = main_loss + recon_term + 1e-4 * reg_term + 0.1 * L2_loss \
                           + 0.1 * loss_matching_GC + 0.1 * loss_matching_GD

                    warm_model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_main.step()

                    ## train discriminator
                    for inner_iter in range(3):
                        _, _, _, warm_id_emb = warm_model(features)

                        item_ids = features[warm_model.item_id_name]
                        item_id_emb = warm_model.origin_item_emb(item_ids).squeeze()
                        real_item_emb = item_id_emb

                        real_pred = warm_model.discriminator(real_item_emb)
                        fake_pred = warm_model.discriminator(warm_id_emb)  # geneated is fake
                        class_label = features[c_feature][:, [0]]
                        #                         class_label = torch.floor(features[c_feature]*c_class_size%c_class_size).long().to(device)
                        #                         class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                        pred_class = warm_model.condition_discriminator(warm_id_emb)
                        condition_loss = CE_loss(pred_class, class_label.squeeze())

                        # print(real_pred.requires_grad)
                        adv_loss = (adv_criterion(fake_pred, fake_label) + adv_criterion(real_pred, real_label)) / 2
                        adv_loss += condition_loss

                        optimizer_d.zero_grad()
                        adv_loss.backward(retain_graph=True)
                        optimizer_d.step()

                    a, b, c, d, e, f = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item() \
                        , e + adv_loss.item(), f + condition_loss.item()
                a, b, c, d, e, f = a / iters, b / iters, c / iters, d / iters, e / iters, f / iters
                if logger and (i + 1) % 10 == 0:
                    print("    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, adv_loss_G: {:.4f}\
                         , condition_loss: {:.4f}".format(i + 1, batch_num, a, b, c, d, e, f), end='\r')
        # warm-up item id embedding
        train_a = dataloaders['train_warm_a']
        for (features, label) in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data

            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes,] = warm_item_id_emb

    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    # test by steps
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list = []

    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc.item())
        print("[gan] evaluate on [test dataset] auc: {:.4f}, F1 score: {:.4f}".format(auc, f1))
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    print("*" * 20, "gan_c_" + model_name, "*" * 20)
    return auc_list



def run(model, dataloaders, args, model_name, warm):
    if warm == 'base':
        auc_list = base(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device,
                        args.save_dir)
    elif warm == 'mwuf':
        auc_list = mwuf(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device,
                        args.save_dir)
    elif warm == 'metaE':
        auc_list = metaE(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device,
                         args.save_dir)
    elif warm == 'cvar':
        auc_list = cvar(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr,
                        args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvaegan':
        auc_list = cvaegan(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr,
                           args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvaegan_c':
        auc_list = cvaegan_c(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr,
                             args.weight_decay, args.device, args.save_dir, args.dataset_name)
    elif warm == 'gan':
        auc_list = gan(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr,
                             args.weight_decay, args.device, args.save_dir, args.dataset_name)

    return auc_list


if __name__ == '__main__':
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    res = {}
    torch.cuda.empty_cache()
    # load or train pretrain models
    drop_suffix = '-dropoutnet' if args.is_dropoutnet else ''
    model_path = os.path.join(args.pretrain_model_path,
                              args.model_name + drop_suffix + '-{}-{}'.format(args.dataset_name, args.seed))
    print(model_path)
    if os.path.exists(model_path):
        model = torch.load(model_path).to(args.device)

        print('load data to device finished')
        dataloaders = get_loaders(args.dataset_name, args.datahub_path, args.device, args.bsz, args.shuffle == 1)
    else:
        print('else load data to device finished')
        model, dataloaders = pretrain(args.dataset_name, args.datahub_path, args.bsz, args.shuffle, args.model_name, \
                                      args.epoch, args.lr, args.weight_decay, args.device, args.save_dir,
                                      args.is_dropoutnet)
        if args.save_pretrain_model:
            torch.save(model, model_path)
    # warmup train and test
    avg_auc_list = []
    for i in range(1):
        model_v = copy.deepcopy(model).to(args.device)
        auc_list = run(model_v, dataloaders, args, args.model_name, args.warmup_model)
        avg_auc_list.append(np.array(auc_list))
    avg_auc_list = list(np.stack(avg_auc_list).mean(axis=0))
    print(avg_auc_list)
