import logging
import os

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import grad
import numpy as np

import config
from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.graphsage.graphsage_net import SageNet
from lib_gnn_model.sgc.sgc_net_batch import SGCNet
from lib_gnn_model.gnn_base import GNNBase
from parameter_parser import parameter_parser
from lib_utils import utils


class NodeClassifier(GNNBase):
    def __init__(self, num_feats, num_classes, args, data=None):
        super(NodeClassifier, self).__init__()

        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.target_model = args['target_model']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.model = self.determine_model(num_feats, num_classes).to(self.device)
        self.data = data

    def determine_model(self, num_feats, num_classes):
        self.logger.info('target model: %s' % (self.args['target_model'],))

        if self.target_model == 'SAGE':
            self.lr, self.decay = 0.01, 0.001
            return SageNet(num_feats, 256, num_classes)
        elif self.target_model == 'GAT':
            self.lr, self.decay = 0.01, 0.001
            return GATNet(num_feats, num_classes)
        elif self.target_model == 'GCN':
            self.lr, self.decay = 0.05, 0.0001
            return GCNNet(num_feats, num_classes)
        elif self.target_model == 'GIN':
            self.lr, self.decay = 0.01, 0.0001
            return GINNet(num_feats, num_classes)
        elif self.target_model == 'SGC':
            self.lr, self.decay = 0.05, 0.0001
            return SGCNet(num_feats, num_classes)
        else:
            raise Exception('unsupported target model')

    def train_model(self, unlearn_info=None):
        self.logger.info("training model")
        self.model.train()
        self.model.reset_parameters()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        self._gen_train_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        for epoch in range(self.args['num_epochs']):
            self.logger.info('epoch %s' % (epoch,))

            for batch_size, n_id, adjs in self.train_loader:
                # self.logger.info("batch size: %s"%(batch_size))
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]

                test_node = np.nonzero(self.data.test_mask.cpu().numpy())[0]
                intersect = np.intersect1d(test_node, n_id.numpy())

                optimizer.zero_grad()

                if self.target_model in ['GCN','SGC']:
                    out = self.model(self.data.x[n_id], adjs, self.edge_weight)
                else:
                    out = self.model(self.data.x[n_id], adjs)

                loss = F.nll_loss(out, self.data.y[n_id[:batch_size]])
                loss.backward()
                optimizer.step()

            train_acc, test_acc = self.evaluate_model()
            self.logger.info(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        
        # out1：原始模型在完整数据上的输出。
        # out2：去学习后模型在修改数据上的输出（如移除某些节点、边或特征）。
        grad_all, grad1, grad2 = None, None, None
        if self.args["method"] in ["GIF", "IF"]:
            if self.target_model in ['GCN','SGC']:
                out1 = self.model.forward_once(self.data, self.edge_weight)
                out2 = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

            else:
                out1 = self.model.forward_once(self.data)
                out2 = self.model.forward_once_unlearn(self.data)
            
            if self.args["unlearn_task"] == "edge":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[2]] = True
                mask2 = mask1
            if self.args["unlearn_task"] == "node":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[0]] = True
                mask1[unlearn_info[2]] = True
                mask2 = np.array([False] * out2.shape[0])
                mask2[unlearn_info[2]] = True
            if self.args["unlearn_task"] == "feature":
                mask1 = np.array([False] * out1.shape[0])
                mask1[unlearn_info[1]] = True
                mask1[unlearn_info[2]] = True
                mask2 = mask1

            loss = F.nll_loss(out1[self.data.train_mask], self.data.y[self.data.train_mask], reduction='sum')
            loss1 = F.nll_loss(out1[mask1], self.data.y[mask1], reduction='sum')
            loss2 = F.nll_loss(out2[mask2], self.data.y[mask2], reduction='sum')
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
            grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
            grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        return (grad_all, grad1, grad2)

    def evaluate_unlearn_F1(self, new_parameters):
        idx = 0
        for p in self.model.parameters():
            p.data = new_parameters[idx]
            idx = idx + 1
        if self.target_model in ['GCN','SGC']:
            out = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)

        else:
            out = self.model.forward_once_unlearn(self.data)

        test_f1 = f1_score(
            self.data.y[self.data['test_mask']].cpu().numpy(), 
            out[self.data['test_mask']].argmax(axis=1).cpu().numpy(), 
            average="micro"
        )
        return test_f1

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        if self.target_model in ['GCN','SGC']:
            out = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            out = self.model.inference(self.data.x, self.test_loader, self.device)

        y_true = self.data.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        results = []
        for mask in [self.data.train_mask, self.data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

        return results

    def posterior(self):
        self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()

        self._gen_test_loader()
        if self.target_model in ['GCN','SGC']:
            posteriors = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            posteriors = self.model.inference(self.data.x, self.test_loader, self.device)

        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask], dim=-1)

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        if self.target_model in ['GCN','SGC']:
            logits = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            logits = self.model.inference(self.data.x, self.test_loader, self.device)
        return logits

    def _gen_train_loader(self):
        self.logger.info("generate train loader")
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        self.train_loader = NeighborSampler(
            edge_index, node_idx=self.data.train_mask,
            sizes=[5, 5], num_nodes=self.data.num_nodes,
            batch_size=self.args['batch_size'], shuffle=True,
            num_workers=0)

        if self.target_model in ['GCN','SGC']:
            _, self.edge_weight = gcn_norm(
                self.data.edge_index, 
                edge_weight=None, 
                num_nodes=self.data.x.shape[0],
                add_self_loops=False)

            if self.args["method"] in ["GIF", "IF"]:
                _, self.edge_weight_unlearn = gcn_norm(
                    self.data.edge_index_unlearn, 
                    edge_weight=None, 
                    num_nodes=self.data.x.shape[0],
                    add_self_loops=False)

        self.logger.info("generate train loader finish")

    def _gen_train_unlearn_load(self):
        self.logger.info("generate train unlearn loader")
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        self.train_unlearn_loader = NeighborSampler(
            edge_index, node_idx=None,
            sizes=[-1], num_nodes=self.data.num_nodes,
            # sizes=[5], num_nodes=self.data.num_nodes,
            batch_size=self.data.num_nodes, shuffle=False,
            num_workers=0)

        if self.target_model in ['GCN','SGC']:
            _, self.edge_weight = gcn_norm(
                self.data.edge_index, 
                edge_weight=None, 
                num_nodes=self.data.x.shape[0],
                add_self_loops=False)

        self.logger.info("generate train loader finish")
    
    def _gen_test_loader(self):
        test_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]

        if not self.args['use_test_neighbors']:
            edge_index = utils.filter_edge_index(self.data.edge_index, test_indices, reindex=False)
        else:
            edge_index = self.data.edge_index

        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 3], [3, 1]])

        self.test_loader = NeighborSampler(
            edge_index, node_idx=None,
            sizes=[-1], num_nodes=self.data.num_nodes,
            # sizes=[5], num_nodes=self.data.num_nodes,
            batch_size=self.args['test_batch_size'], shuffle=False,
            num_workers=0)

        if self.target_model in ['GCN','SGC']:
            _, self.edge_weight = gcn_norm(self.data.edge_index, edge_weight=None, num_nodes=self.data.x.shape[0],
                                           add_self_loops=False)


if __name__ == '__main__':
    os.chdir('../')
    args = parameter_parser()

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    train_indices, test_indices = train_test_split(np.arange((data.num_nodes)), test_size=0.2, random_state=100)
    data.train_mask, data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes,
                                                                                                 dtype=torch.bool)
    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True

    graphsage = NodeClassifier(dataset.num_features, dataset.num_classes, args, data)
    graphsage.train_model()
