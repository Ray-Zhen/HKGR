from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
	t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10,
	                                           get_mAP=True)
	t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
	if retur_indices:
		return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
		        t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]], indices
	else:
		return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("HKGR.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        # return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
        return qfeats, gfeats, qids, gids

    def _compute_embedding_tse(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1))  # flatten
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        # return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
        return qfeats.cuda(), gfeats.cuda(), qids.cuda(), gids.cuda()
    
    def eval(self, model, i2t_metric=False, fold5=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        vq_feats, vg_feats, _, _ = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1)  # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1)  # image features
      

        if not fold5:
            similarity_glob = qfeats @ gfeats.t()
            similarity_loc = vq_feats @ vg_feats.t()
            sims_dict = {
                'GLOB': similarity_glob,
                'LOC': similarity_loc,
                'GLOB+LOC': (similarity_glob + similarity_loc) / 2
            }  

            table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])

            # t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            # t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()
            # table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
            # table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

            # if i2t_metric:
            #     i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            #     i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            #     table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
            # table.float_format = '.4'
            for key in sims_dict.keys():
                sims = sims_dict[key]
                rs = get_metrics(sims, qids, gids, f'{key}-t2i', False)
                table.add_row(rs)
            if i2t_metric:
                for key in sims_dict.keys():
                    sims = sims_dict[key]
                    rs_i2t = get_metrics(sims.t(), gids, qids, f'{key}-i2t', False)
                    table.add_row(rs_i2t)                    

            table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
            table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
            table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
            table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
            table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
            table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
            table.vertical_char = " "
            self.logger.info('\n' + str(table))

            return rs[1] + rs_i2t[1]

        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                similarity = qfeats[i * 5000:(i + 1) * 5000] @ gfeats[i * 1000:(i + 1) * 1000].t()

                t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids[i * 5000:(i + 1) * 5000],
                                                        g_pids=gids[i * 1000:(i + 1) * 1000], max_rank=10, get_mAP=True)
                t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.tolist(), t2i_mAP.tolist(), t2i_mINP.tolist()

                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids[i * 1000:(i + 1) * 1000],
                                                        g_pids=qids[i * 5000:(i + 1) * 5000], max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.tolist(), i2t_mAP.tolist(), i2t_mINP.tolist()

                if True:
                    table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
                    table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP])
                    table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP])
                    table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
                    table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
                    table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
                    table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
                    self.logger.info('\n' + str(table))

                results += [t2i_cmc + i2t_cmc + [t2i_mAP, i2t_mAP]]

            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

            table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
            table.add_row(['t2i', mean_metrics[0], mean_metrics[4], mean_metrics[9], mean_metrics[20]])
            table.add_row(['i2t', mean_metrics[10], mean_metrics[1], mean_metrics[19], mean_metrics[21]])

            table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
            table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
            table.vertical_char = " "
            self.logger.info('\n' + "Mean metrics: ")
            self.logger.info('\n' + str(table))

            return mean_metrics[0]
