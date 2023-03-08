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
    print("*" * 20, "cvaegan_c_" + model_name, "*" * 20)
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pth')
    train_base = dataloaders['train_base']
    if dataset_name == 'movielens1M':
        # c_feature = 'clk_seq'
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
                         warm_features=dataloaders.user_features,
                         train_loader=train_base,
                         device=device, c_feature=c_feature, c_class_size=c_class_size).to(device)
    warm_model.init_cvaegan()

    def warm_up(dataloader, epochs, iters, logger=False):
        warm_model.train()
        criterion = torch.nn.BCELoss()
        CE_loss = torch.nn.CrossEntropyLoss()
        # cce_loss = tf.keras.losses.CategoricalCrossentropy()

        warm_model.optimizer_cvaegan()

        optimizer_main = torch.optim.Adam(params=[kv[1] for kv in warm_model.named_parameters() if
                                                  kv[1].requires_grad and 'discriminator' not in kv[0]],\
                                          lr=lr, weight_decay=weight_decay)  # except D

        optimizer_d = torch.optim.Adam(params=[p for p in warm_model.discriminator.parameters()] + [p for p in
                                       warm_model.condition_discriminator.parameters()],\
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
                    seq_emb = warm_model.c_feature_emb
                    # main loss = −Ez∼Pz [log D(G(z))].
                    # The discriminator LD = −Ex∼Pr [logD(x)] − Ez∼Pz [log(1 − D(G(z))],(1)
                    pred_adv = warm_model.discriminator(warm_id_emb)
                    loss_G = adv_criterion(pred_adv, real_label)  # make D think it is real
                    main_loss = criterion(target, label.float())
                    main_loss += loss_G


                    # get discriminator matching loss LGD = 1/2 ||Ex∼Pr fC (x) − Ez∼Pz fC (G(z))||2
                    matching_layer = len(warm_model.discriminator) - 1
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()
                    for j in range(matching_layer):
                        real_input = warm_model.discriminator[j](real_input)
                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.discriminator[j](warm_id_emb_input)
                    loss_matching_GD = 0.5 * torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input))

                    # L2_D are the features of an intermediate layer of generator network C ||fC(x) − fC(x 0 )||2
                    loss_L2_D = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get condition discriminator matching loss LGC = 1/2 Ec ||Ex∼Pr fD(x) − Ez∼Pz fD(G(z, c))||2
                    matching_layer = len(warm_model.condition_discriminator) - 1
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()
                    for j in range(matching_layer):
                        real_input = warm_model.condition_discriminator[j](real_input)
                    warm_id_emb_input = warm_id_emb
                    for j in range(matching_layer):
                        warm_id_emb_input = warm_model.condition_discriminator[j](warm_id_emb_input)
                    loss_matching_GC = 0.5 * torch.mean(torch.square(torch.mean(warm_id_emb_input) - torch.mean(real_input)))
                    #                     loss_matching_GC = 0
                    #                     for j in range(18):
                    #                         iclass_idx = torch.equal(class_label, j*torch.ones_like(class_label))
                    #                         loss_matching_GC += torch.square(torch.mean(warm_id_emb_input[iclass_idx]) - torch.mean(real_input[iclass_idx]))

                    # L2_C are the features of an intermediate layer of d classification network D ||fD (x) − fD (x 0 )||2
                    loss_L2_C = 0.5 * torch.mean(torch.square(warm_id_emb_input - real_input))

                    # get l2 reconstruction loss  1/2 (||x − x 0 ||2 + ||fD(x) − fD(x 0 )||2 + ||fC (x) − fC (x 0 )||2) (6)
                    # we add LM pair-wise feature matching loss between x and x' = 1/2 (||x − x 0 ||2
                    target, recon_term, reg_term, warm_id_emb = warm_model(features)
                    real_input = warm_model.origin_item_emb(features[warm_model.item_id_name]).squeeze()
                    LM = 0.5 * (torch.mean(torch.square(warm_id_emb - real_input)))


                    # get Conditional loss LC = −Ex∼Pr [log P(c|x)].The output of each entry represents the posterior probability P(c|x).（3）
                    pred_class = warm_model.condition_discriminator(warm_id_emb).to(device)
                    # class_label = warm_model.condition_discriminator(warm_model.c_feature_emb).to(device)
                    class_label = features[c_feature][:, [0]].squeeze()
                    # class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                    condition_loss = CE_loss(pred_class, class_label)

                    # total loss
                    L2_loss =  loss_L2_D + LM
                    L2_loss += loss_L2_C #效果变差了？
                    loss = main_loss + recon_term + 1e-4 * reg_term + 0.1 * L2_loss \
                           + 0.1 * loss_matching_GD
                    loss = loss + 0.1 * loss_matching_GC
                    loss = loss + 0.1 * condition_loss

                    warm_model.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_main.step()

                    ## train discriminator
                    for inner_iter in range(3):
                        _, _, _, warm_id_emb= warm_model(features)

                        item_ids = features[warm_model.item_id_name]
                        item_id_emb = warm_model.origin_item_emb(item_ids).squeeze()
                        real_item_emb = item_id_emb

                        # adv_loss
                        real_pred = warm_model.discriminator(real_item_emb)
                        fake_pred = warm_model.discriminator(warm_id_emb)  # geneated is fake
                        adv_loss = (adv_criterion(fake_pred, fake_label) + adv_criterion(real_pred, real_label)) / 2

                        # condition loss
                        #                         class_label = torch.floor(features[c_feature]*c_class_size%c_class_size).long().to(device)
                        #                         class_label = torch.zeros(len(pred_class), c_class_size).to(device).scatter_(1, freq, 1).to(device)
                        # class_label = warm_model.discriminator(c_feature_emb).to(device)
                        class_label = features[c_feature][:, [0]].squeeze()
                        pred_class = warm_model.discriminator(warm_id_emb)
                        # condition_loss = CE_loss(pred_class, class_label)
                        # adv_loss += condition_loss

                        optimizer_d.zero_grad()
                        adv_loss.backward(retain_graph=True)
                        optimizer_d.step()

                    a, b, c, d, e, f = a + loss.item(), b + main_loss.item(), c + recon_term.item(), d + reg_term.item() \
                        , e + adv_loss.item(), f + loss_G.item()
                a, b, c, d, e, f = a / iters, b / iters, c / iters, d / iters, e / iters, f / iters
                if logger and (i + 1) % 10 == 0:
                    print("    Iter {}/{}, loss: {:.4f}, main loss: {:.4f}, recon loss: {:.4f}, reg loss: {:.4f}, adv_loss_G: {:.4f}\
                         , loss_D: {:.4f}".format(i + 1, batch_num, a, b, c, d, e, f), end='\r')
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
