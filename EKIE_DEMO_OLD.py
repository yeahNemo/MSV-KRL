import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import rdflib
from rdflib.namespace import RDFS
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# 添加MiniLM相关导入
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# --- 辅助函数 ---
def norm(x, pnorm=2, dim=-1):
    """计算张量的范数"""
    return torch.norm(x, p=pnorm, dim=dim)

def normalize_emb(emb):
    """归一化嵌入向量 (L2范数)"""
    return torch.nn.functional.normalize(emb, p=2, dim=-1)

def normalize_radius(radius_emb):
    """确保半径为正值"""
    return torch.clamp(radius_emb, min=1e-5)  # 避免半径为0或负

def generate_minilm_embedding(entity_uri, entity_prop_2_anno_dict, tokenizer, model, device, emb_dim):
    """使用MiniLM为单个实体/关系生成嵌入"""
    try:
        # 尝试获取标注属性
        text = entity_prop_2_anno_dict.get(entity_uri, entity_uri)
    except:
        text = entity_uri
    
    # 编码文本
    inputs_ids = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)
    batch_encoding = inputs_ids['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=batch_encoding, output_hidden_states=True)
        # 使用最后一层的平均池化作为嵌入
        last_hidden_state = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
        # 排除[CLS]和[SEP]标记，对中间的token进行平均池化
        if last_hidden_state.size(0) > 2:
            embedding = torch.mean(last_hidden_state[1:-1], dim=0)  # 排除首尾token
        else:
            embedding = torch.mean(last_hidden_state, dim=0)
        
        # 如果嵌入维度不匹配，进行调整
        if embedding.size(0) != emb_dim:
            if embedding.size(0) > emb_dim:
                embedding = embedding[:emb_dim]  # 截断
            else:
                # 填充
                padding = torch.zeros(emb_dim - embedding.size(0), device=device)
                embedding = torch.cat([embedding, padding], dim=0)
    
    return embedding.cpu().numpy()

class EIKETrain(nn.Module):
    def __init__(self, args, num_instances, num_concepts, num_relations,
                 initial_entity_embeddings=None, initial_relation_embeddings=None,
                 instance_global_indices=None, concept_global_indices=None, device="cpu"):
        super(EIKETrain, self).__init__()
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim  # 假设 emb_dim 在 args 中定义
        self.num_instances = num_instances
        self.num_concepts = num_concepts
        self.num_relations = num_relations

        # --- 嵌入层定义 ---
        # 实例嵌入 (对应 MTL 中的实体嵌入)
        self.instance_vec_ex = nn.Embedding(self.num_instances, self.emb_dim)
        # 概念嵌入 (外部表示和半径)
        self.concept_vec_ex = nn.Embedding(self.num_concepts, self.emb_dim)
        self.concept_r = nn.Embedding(self.num_concepts, 1)  # 概念的半径
        # 关系嵌入
        self.relation_vec = nn.Embedding(self.num_relations, self.emb_dim)

        # --- 初始化嵌入权重 ---
        if initial_entity_embeddings is not None:
            print("使用提供的初始实体嵌入初始化实例嵌入。")
            self.instance_vec_ex.weight.data.copy_(torch.from_numpy(initial_entity_embeddings))
        else:
            print("使用Xavier均匀分布初始化实例嵌入。")
            nn.init.xavier_uniform_(self.instance_vec_ex.weight.data)

        if initial_relation_embeddings is not None:
            print("使用提供的初始关系嵌入初始化关系嵌入。")
            self.relation_vec.weight.data.copy_(torch.from_numpy(initial_relation_embeddings))
        else:
            print("使用Xavier均匀分布初始化关系嵌入。")
            nn.init.xavier_uniform_(self.relation_vec.weight.data)

        nn.init.xavier_uniform_(self.concept_vec_ex.weight.data)
        nn.init.uniform_(self.concept_r.weight.data, a=0.01, b=1.0)  # 半径在 (0.01, 1.0) 之间

        # --- 优化器 ---
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.0
        )
        
        # 损失函数中的超参数
        self.margin = self.args.margin if hasattr(self.args, 'margin') else 1.0
        self.pnorm = self.args.pnorm if hasattr(self.args, 'pnorm') else 2

    def loss_hrt(self, h_pos, r_pos, t_pos, h_neg, r_neg, t_neg):
        """计算 HLR (head, relation, tail) 三元组的损失。"""
        h_pos_emb = self.instance_vec_ex(h_pos)
        r_pos_emb = self.relation_vec(r_pos)
        t_pos_emb = self.instance_vec_ex(t_pos)

        h_neg_emb = self.instance_vec_ex(h_neg)
        r_neg_emb = self.relation_vec(r_neg)
        t_neg_emb = self.instance_vec_ex(t_neg)

        pos_score = norm(h_pos_emb + r_pos_emb - t_pos_emb, pnorm=self.pnorm)
        neg_score = norm(h_neg_emb + r_neg_emb - t_neg_emb, pnorm=self.pnorm)
        
        loss = torch.relu(self.margin + pos_score - neg_score).mean()
        return loss, pos_score, neg_score

    def loss_instance_of(self, inst_pos, concept_pos, inst_neg, concept_neg):
        """计算 InstanceOf 关系的损失。"""
        # 正例: inst_pos 应该在 concept_pos 内
        inst_pos_emb = self.instance_vec_ex(inst_pos)
        concept_pos_emb = self.concept_vec_ex(concept_pos)
        concept_pos_r = self.concept_r(concept_pos).squeeze(-1)
        
        # 反例: inst_neg 不应该在 concept_pos 内
        inst_neg_emb = self.instance_vec_ex(inst_neg)

        dist_pos = norm(inst_pos_emb - concept_pos_emb, pnorm=self.pnorm)
        # 正例损失: 希望 dist_pos <= concept_pos_r
        loss_pos = torch.relu(dist_pos - concept_pos_r + self.margin).mean()

        # 反例损失: 希望 dist_neg > concept_pos_r
        dist_neg = norm(inst_neg_emb - concept_pos_emb, pnorm=self.pnorm)
        loss_neg = torch.relu(concept_pos_r - dist_neg + self.margin).mean()
        
        loss = loss_pos + loss_neg
        return loss

    def loss_subclass_of(self, sub_pos, super_pos, sub_neg, super_neg):
        """计算 SubClassOf 关系的损失。"""
        # 正例: sub_pos 是 super_pos 的子类
        sub_pos_emb = self.concept_vec_ex(sub_pos)
        sub_pos_r = self.concept_r(sub_pos).squeeze(-1)
        super_pos_emb = self.concept_vec_ex(super_pos)
        super_pos_r = self.concept_r(super_pos).squeeze(-1)

        # 反例: sub_neg 不是 super_pos 的子类
        sub_neg_emb = self.concept_vec_ex(sub_neg)
        sub_neg_r = self.concept_r(sub_neg).squeeze(-1)

        # 正例损失: 希望 ||sub_pos_emb - super_pos_emb|| + sub_pos_r <= super_pos_r
        dist_centers_pos = norm(sub_pos_emb - super_pos_emb, pnorm=self.pnorm)
        loss_pos = torch.relu(dist_centers_pos + sub_pos_r - super_pos_r + self.margin).mean()

        # 反例损失: 希望 ||sub_neg_emb - super_pos_emb|| + sub_neg_r > super_pos_r
        dist_centers_neg = norm(sub_neg_emb - super_pos_emb, pnorm=self.pnorm)
        loss_neg = torch.relu(super_pos_r - (dist_centers_neg + sub_neg_r) + self.margin).mean()
        
        loss = loss_pos + loss_neg
        return loss

    def save_trained_embeddings(self, save_dir):
        """保存训练后的实例 (实体) 和关系嵌入向量。"""
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                print(f"创建目录: {save_dir}")
            except OSError as e:
                print(f"创建目录 {save_dir} 失败: {e}")
                return

        entity_embeddings_path = os.path.join(save_dir, "trained_entity_embeddings.npy")
        relation_embeddings_path = os.path.join(save_dir, "trained_relation_embeddings.npy")
        concept_embeddings_path = os.path.join(save_dir, "trained_concept_embeddings.npy")
        concept_radius_path = os.path.join(save_dir, "trained_concept_radius.npy")

        # 获取嵌入，移至CPU，分离计算图，转为numpy
        entity_embeddings = self.instance_vec_ex.weight.data.cpu().detach().numpy()
        relation_embeddings = self.relation_vec.weight.data.cpu().detach().numpy()
        concept_embeddings = self.concept_vec_ex.weight.data.cpu().detach().numpy()
        concept_radius = self.concept_r.weight.data.cpu().detach().numpy()

        try:
            np.save(entity_embeddings_path, entity_embeddings)
            print(f"已保存训练后的实体嵌入至: {entity_embeddings_path}")
            
            np.save(relation_embeddings_path, relation_embeddings)
            print(f"已保存训练后的关系嵌入至: {relation_embeddings_path}")
            
            np.save(concept_embeddings_path, concept_embeddings)
            print(f"已保存训练后的概念嵌入至: {concept_embeddings_path}")
            
            np.save(concept_radius_path, concept_radius)
            print(f"已保存训练后的概念半径至: {concept_radius_path}")
            
            # 保存实体ID和关系ID映射
            if hasattr(self, 'entity2id') and hasattr(self, 'relation2id'):
                with open(os.path.join(save_dir, "entity2id.pkl"), 'wb') as f:
                    pickle.dump(self.entity2id, f)
                with open(os.path.join(save_dir, "relation2id.pkl"), 'wb') as f:
                    pickle.dump(self.relation2id, f)
                print("已保存实体和关系ID映射")
                
            if hasattr(self, 'concept2id'):
                with open(os.path.join(save_dir, "concept2id.pkl"), 'wb') as f:
                    pickle.dump(self.concept2id, f)
                print("已保存概念ID映射")
        except Exception as e:
            print(f"保存嵌入失败: {e}")

    def run_training(self, train_dataloaders, valid_dataloaders=None):
        """执行模型训练。"""
        print(f"开始训练，设备: {self.device}")
        epoch_losses_log = {"train": [], "valid": []}

        for epoch in range(self.args.ekie_nepoch):
            self.train()  # 设置模型为训练模式
            current_epoch_total_loss = 0.0
            num_batches = 0

            # --- 嵌入归一化 (每个epoch开始时) ---
            with torch.no_grad():
                self.instance_vec_ex.weight.data = normalize_emb(self.instance_vec_ex.weight.data)
                self.concept_vec_ex.weight.data = normalize_emb(self.concept_vec_ex.weight.data)
                self.relation_vec.weight.data = normalize_emb(self.relation_vec.weight.data)
                self.concept_r.weight.data = normalize_radius(self.concept_r.weight.data)

            # --- 训练 HLR 三元组 ---
            if self.args.is_train_hrt and 'hrt' in train_dataloaders and train_dataloaders['hrt'] is not None:
                print(f"Epoch {epoch+1}: 训练 HLR 三元组...")
                for batch_data in tqdm(train_dataloaders['hrt'], desc=f"Epoch {epoch+1} HLR"):
                    h_pos, r_pos, t_pos, h_neg, r_neg, t_neg = [d.to(self.device) for d in batch_data]
                    self.optimizer.zero_grad()
                    loss, _, _ = self.loss_hrt(h_pos, r_pos, t_pos, h_neg, r_neg, t_neg)
                    loss.backward()
                    self.optimizer.step()
                    current_epoch_total_loss += loss.item()
                    num_batches += 1
            elif not self.args.is_train_hrt and 'hrt' in train_dataloaders and train_dataloaders['hrt'] is not None:
                print(f"Epoch {epoch+1}: 跳过 HLR 三元组训练 (is_train_hrt=False)")
            
            # --- 训练 InstanceOf 关系 ---
            if 'instance_of' in train_dataloaders and train_dataloaders['instance_of'] is not None:
                print(f"Epoch {epoch+1}: 训练 InstanceOf 关系...")
                for batch_data in tqdm(train_dataloaders['instance_of'], desc=f"Epoch {epoch+1} InstanceOf"):
                    inst_pos, concept_pos, inst_neg, concept_neg = [d.to(self.device) for d in batch_data]
                    self.optimizer.zero_grad()
                    loss = self.loss_instance_of(inst_pos, concept_pos, inst_neg, concept_neg)
                    loss.backward()
                    self.optimizer.step()
                    current_epoch_total_loss += loss.item()
                    num_batches += 1

            # --- 训练 SubClassOf 关系 ---
            if 'subclass_of' in train_dataloaders and train_dataloaders['subclass_of'] is not None:
                print(f"Epoch {epoch+1}: 训练 SubClassOf 关系...")
                for batch_data in tqdm(train_dataloaders['subclass_of'], desc=f"Epoch {epoch+1} SubClassOf"):
                    sub_pos, super_pos, sub_neg, super_neg = [d.to(self.device) for d in batch_data]
                    self.optimizer.zero_grad()
                    loss = self.loss_subclass_of(sub_pos, super_pos, sub_neg, super_neg)
                    loss.backward()
                    self.optimizer.step()
                    current_epoch_total_loss += loss.item()
                    num_batches += 1
            
            avg_epoch_loss = current_epoch_total_loss / num_batches if num_batches > 0 else 0
            epoch_losses_log["train"].append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{self.args.ekie_nepoch}, 训练损失: {avg_epoch_loss:.6f}")

            # --- 验证过程 ---
            if valid_dataloaders and (epoch + 1) % self.args.valid_freq == 0:
                self.eval() 
                current_valid_total_loss = 0.0
                num_valid_batches = 0
                with torch.no_grad():
                    if self.args.is_train_hrt and 'hrt' in valid_dataloaders and valid_dataloaders['hrt'] is not None:
                        for batch_data in valid_dataloaders['hrt']:
                            h_pos, r_pos, t_pos, h_neg, r_neg, t_neg = [d.to(self.device) for d in batch_data]
                            loss, _, _ = self.loss_hrt(h_pos, r_pos, t_pos, h_neg, r_neg, t_neg)
                            current_valid_total_loss += loss.item()
                            num_valid_batches += 1
                    if 'instance_of' in valid_dataloaders and valid_dataloaders['instance_of'] is not None:
                        for batch_data in valid_dataloaders['instance_of']:
                            inst_pos, concept_pos, inst_neg, concept_neg = [d.to(self.device) for d in batch_data]
                            loss = self.loss_instance_of(inst_pos, concept_pos, inst_neg, concept_neg)
                            current_valid_total_loss += loss.item()
                            num_valid_batches += 1
                    if 'subclass_of' in valid_dataloaders and valid_dataloaders['subclass_of'] is not None:
                        for batch_data in valid_dataloaders['subclass_of']:
                            sub_pos, super_pos, sub_neg, super_neg = [d.to(self.device) for d in batch_data]
                            loss = self.loss_subclass_of(sub_pos, super_pos, sub_neg, super_neg)
                            current_valid_total_loss += loss.item()
                            num_valid_batches += 1
                
                avg_valid_loss = current_valid_total_loss / num_valid_batches if num_valid_batches > 0 else 0
                epoch_losses_log["valid"].append(avg_valid_loss)
                print(f"Epoch {epoch + 1}/{self.args.ekie_nepoch}, 验证损失: {avg_valid_loss:.6f}")
            
            # --- 模型保存 (定期) ---
            if hasattr(self.args, 'save_freq') and self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0:
                self.save_model(epoch + 1)
        
        print("训练完成。")
        
        # --- 保存最终模型 ---
        self.save_model("final")

        # --- 保存训练后的嵌入向量 ---
        if hasattr(self.args, 'ekie_embeddings_saved_path') and self.args.ekie_embeddings_saved_path:
            print(f"开始保存训练后的嵌入向量至 {self.args.ekie_embeddings_saved_path}...")
            self.save_trained_embeddings(self.args.ekie_embeddings_saved_path)
        else:
            default_save_dir = "./trained_model_embeddings"
            print(f"警告: args 中未定义 'ekie_embeddings_saved_path' 或路径为空。")
            print(f"将尝试保存嵌入向量至默认路径: {default_save_dir}")
            self.save_trained_embeddings(default_save_dir)
            
        return epoch_losses_log

    def save_model(self, epoch_num_or_name="final"):
        """保存模型状态。"""
        if not hasattr(self.args, 'ekie_models_saved_path'):
            print("错误: args 中未定义 ekie_models_saved_path。模型未保存。")
            return
        
        # 确保保存模型的目录存在
        save_dir = os.path.dirname(self.args.ekie_models_saved_path)
        if save_dir and not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except OSError as e:
                print(f"创建模型保存目录 {save_dir} 失败: {e}")
                return

        save_path = f"{self.args.ekie_models_saved_path}eike_{epoch_num_or_name}.pth"
        try:
            torch.save({
                'epoch': epoch_num_or_name,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'args': self.args 
            }, save_path)
            print(f"模型已保存至 {save_path}")
        except Exception as e:
            print(f"保存模型至 {save_path} 失败: {e}")

    def load_model(self, model_path):
        """加载已保存的模型状态。"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在。")
            return False

        if not torch.cuda.is_available() and "cuda" in str(self.device):
            map_location = torch.device('cpu')
            print(f"CUDA 不可用，模型将加载到 CPU。原始设备为 {self.device}。")
        else:
            map_location = self.device

        try:
            checkpoint = torch.load(model_path, map_location=map_location)
            self.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("优化器状态已加载。")
                except Exception as e:
                    print(f"加载优化器状态时发生错误: {e}。优化器状态未加载。")
            else:
                print("警告: 检查点中未找到优化器状态或模型没有优化器。优化器状态未加载。")

            loaded_epoch = checkpoint.get('epoch', '未知')
            
            print(f"模型从 {model_path} 加载完成。")
            print(f"加载的 Epoch: {loaded_epoch}")
            return True
        except Exception as e:
            print(f"加载模型从 {model_path} 失败: {e}")
            return False

# --- 数据集类 ---
class HRTDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id, num_entities, num_relations, neg_ratio=1):
        """
        HRT三元组数据集
        triples: 三元组列表 [(h, r, t), ...]
        entity2id: 实体到ID的映射
        relation2id: 关系到ID的映射
        num_entities: 实体总数
        num_relations: 关系总数
        neg_ratio: 每个正例对应的负例数量
        """
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_ratio = neg_ratio
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # 转换为ID
        h_id = self.entity2id.get(h, 0)
        r_id = self.relation2id.get(r, 0)
        t_id = self.entity2id.get(t, 0)
        
        # 负采样 (简单策略: 随机替换头或尾)
        if np.random.random() < 0.5:
            # 替换头实体
            h_neg_id = np.random.randint(0, self.num_entities)
            while h_neg_id == h_id:  # 避免采样到相同的实体
                h_neg_id = np.random.randint(0, self.num_entities)
            return torch.tensor(h_id), torch.tensor(r_id), torch.tensor(t_id), torch.tensor(h_neg_id), torch.tensor(r_id), torch.tensor(t_id)
        else:
            # 替换尾实体
            t_neg_id = np.random.randint(0, self.num_entities)
            while t_neg_id == t_id:  # 避免采样到相同的实体
                t_neg_id = np.random.randint(0, self.num_entities)
            return torch.tensor(h_id), torch.tensor(r_id), torch.tensor(t_id), torch.tensor(h_id), torch.tensor(r_id), torch.tensor(t_neg_id)

class SubClassOfDataset(Dataset):
    def __init__(self, triples, concept2id, num_concepts, neg_ratio=1):
        """
        SubClassOf关系数据集
        triples: 三元组列表 [(sub, 'subClassOf', super), ...]
        concept2id: 概念到ID的映射
        num_concepts: 概念总数
        neg_ratio: 每个正例对应的负例数量
        """
        self.triples = triples
        self.concept2id = concept2id
        self.num_concepts = num_concepts
        self.neg_ratio = neg_ratio
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        sub, _, super_c = self.triples[idx]
        
        # 转换为ID
        sub_id = self.concept2id.get(sub, 0)
        super_id = self.concept2id.get(super_c, 0)
        
        # 负采样 (随机替换子类或父类)
        if np.random.random() < 0.5:
            # 替换子类
            sub_neg_id = np.random.randint(0, self.num_concepts)
            while sub_neg_id == sub_id:  # 避免采样到相同的概念
                sub_neg_id = np.random.randint(0, self.num_concepts)
            return torch.tensor(sub_id), torch.tensor(super_id), torch.tensor(sub_neg_id), torch.tensor(super_id)
        else:
            # 替换父类
            super_neg_id = np.random.randint(0, self.num_concepts)
            while super_neg_id == super_id:  # 避免采样到相同的概念
                super_neg_id = np.random.randint(0, self.num_concepts)
            return torch.tensor(sub_id), torch.tensor(super_id), torch.tensor(sub_id), torch.tensor(super_neg_id)

class InstanceOfDataset(Dataset):
    def __init__(self, triples, entity2id, concept2id, num_entities, num_concepts, neg_ratio=1):
        """
        InstanceOf关系数据集
        triples: 三元组列表 [(instance, 'type', concept), ...]
        entity2id: 实体到ID的映射
        concept2id: 概念到ID的映射
        num_entities: 实体总数
        num_concepts: 概念总数
        neg_ratio: 每个正例对应的负例数量
        """
        self.triples = triples
        self.entity2id = entity2id
        self.concept2id = concept2id
        self.num_entities = num_entities
        self.num_concepts = num_concepts
        self.neg_ratio = neg_ratio
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        instance, _, concept = self.triples[idx]
        
        # 转换为ID
        instance_id = self.entity2id.get(instance, 0)
        concept_id = self.concept2id.get(concept, 0)
        
        # 负采样 (随机替换实例或概念)
        if np.random.random() < 0.5:
            # 替换实例
            instance_neg_id = np.random.randint(0, self.num_entities)
            while instance_neg_id == instance_id:  # 避免采样到相同的实体
                instance_neg_id = np.random.randint(0, self.num_entities)
            return torch.tensor(instance_id), torch.tensor(concept_id), torch.tensor(instance_neg_id), torch.tensor(concept_id)
        else:
            # 替换概念
            concept_neg_id = np.random.randint(0, self.num_concepts)
            while concept_neg_id == concept_id:  # 避免采样到相同的概念
                concept_neg_id = np.random.randint(0, self.num_concepts)
            return torch.tensor(instance_id), torch.tensor(concept_id), torch.tensor(instance_id), torch.tensor(concept_neg_id)

# --- 主程序/示例用法 ---
if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="EIKE模型训练")
    parser.add_argument('--config', type=str, default='/home/cqjtu/NLP-Group/YSY/Projects/KGE/NEMO_EXPERIMENT/MSV_KRL_FORK/MSV-KRL/configs/helis_config.json', help='配置文件路径')
    parser.add_argument('--emb_dim', type=int, default=384, help='嵌入维度')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--margin', type=float, default=1.0, help='损失函数边界')
    parser.add_argument('--pnorm', type=int, default=2, help='距离计算的范数 (1为L1, 2为L2)')
    parser.add_argument('--valid_freq', type=int, default=1, help='验证频率')
    parser.add_argument('--save_freq', type=int, default=5, help='模型保存频率(轮数)，0表示禁用中间保存')
    parser.add_argument('--ekie_model_save_path_prefix', type=str, default='./saved_models/eike_model', help='模型保存路径前缀')
    parser.add_argument('--ekie_embedding_save_path', type=str, default='./trained_model_embeddings', help='训练后嵌入保存路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            # 将配置文件中的参数更新到args
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("将使用命令行参数...")
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练的嵌入
    embedding_dict = None
    if args.ekie_using_exsiting_model:
        embedding_path = args.related_file_save_path + args.embeddings_save_file_name
        print(f"尝试加载预训练嵌入: {embedding_path}")
        try:
            with open(embedding_path, 'rb') as f:
                embedding_dict = pickle.load(f)
            print(f"成功加载嵌入字典，包含 {len(embedding_dict)} 个实体/关系")
        except Exception as e:
            print(f"加载嵌入失败: {e}")
            print("将使用随机初始化的嵌入")
            embedding_dict = None
    
    # 加载数据集
    print("加载数据集...")
    
    # 1. 加载实体和关系
    entity_set = set()
    relation_set = set()
    concept_set = set()
    
    # 尝试从文件加载实体和关系
    try:
        with open(args.related_file_save_path + 'nodes.txt', 'r') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    entity_set.add(entity)
        
        with open(args.related_file_save_path + 'edges.txt', 'r') as f:
            for line in f:
                relation = line.strip()
                if relation:
                    relation_set.add(relation)
        
        print(f"从文件加载了 {len(entity_set)} 个实体和 {len(relation_set)} 个关系")
    except Exception as e:
        print(f"从文件加载实体和关系失败: {e}")
        print("将尝试从RDF图中提取实体和关系")
    
    # 2. 加载三元组数据
    # 2.1 加载subClassOf三元组
    subclassof_triples = []
    try:
        subclassof_graph = rdflib.Graph().parse(args.related_file_save_path + 'subgraphs/subclassof.xml')
        for s, p, o in subclassof_graph:
            if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                continue
            
            s_str = str(s).strip()
            p_str = str(p).strip()
            o_str = str(o).strip()
            
            if s_str and o_str:
                subclassof_triples.append([s_str, p_str, o_str])
                concept_set.add(s_str)
                concept_set.add(o_str)
        
        print(f"加载了 {len(subclassof_triples)} 个subClassOf三元组")
    except Exception as e:
        print(f"加载subClassOf三元组失败: {e}")
    
    # 2.2 加载instanceOf三元组
    instanceof_triples = []
    try:
        instanceof_graph = rdflib.Graph().parse(args.related_file_save_path + 'subgraphs/ind_type.xml')
        for s, p, o in instanceof_graph:
            if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                continue
            
            s_str = str(s).strip()
            p_str = str(p).strip()
            o_str = str(o).strip()
            
            if s_str and o_str:
                instanceof_triples.append([s_str, p_str, o_str])
                entity_set.add(s_str)  # 实例
                concept_set.add(o_str)  # 概念
        
        print(f"加载了 {len(instanceof_triples)} 个instanceOf三元组")
    except Exception as e:
        print(f"加载instanceOf三元组失败: {e}")
    
    # 2.3 加载普通关系三元组
    relation_triples = []
    try:
        # 加载所有其他关系三元组
        relation_graph = rdflib.Graph().parse(args.related_file_save_path + 'subgraphs/all_relation.xml')
        for s, p, o in relation_graph:
            if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                continue
            
            s_str = str(s).strip()
            p_str = str(p).strip()
            o_str = str(o).strip()
            
            if s_str and p_str and o_str:
                relation_triples.append([s_str, p_str, o_str])
                entity_set.add(s_str)
                entity_set.add(o_str)
                relation_set.add(p_str)
        
        print(f"加载了 {len(relation_triples)} 个普通关系三元组")
    except Exception as e:
        print(f"加载普通关系三元组失败: {e}")
    
    # 3. 创建ID映射
    # 确保所有概念也在实体集合中
    entity_set.update(concept_set)
    
    entity2id = {entity: idx for idx, entity in enumerate(entity_set)}
    relation2id = {relation: idx for idx, relation in enumerate(relation_set)}
    concept2id = {concept: idx for idx, concept in enumerate(concept_set)}
    
    print(f"实体总数: {len(entity2id)}")
    print(f"关系总数: {len(relation2id)}")
    print(f"概念总数: {len(concept2id)}")
    
    # 4. 准备嵌入初始化
    if embedding_dict is not None:
        # 从预训练嵌入中提取实体和关系嵌入
        entity_embeddings = np.zeros((len(entity2id), args.emb_dim))
        relation_embeddings = np.zeros((len(relation2id), args.emb_dim))
        
        # 填充实体嵌入
        for entity, idx in entity2id.items():
            if entity in embedding_dict:
                entity_embeddings[idx] = embedding_dict[entity]
            else:
                print(f"警告: 实体 '{entity}' 在预训练嵌入中不存在，使用随机初始化")
                entity_embeddings[idx] = np.random.uniform(-0.1, 0.1, args.emb_dim)
        
        # 填充关系嵌入
        for relation, idx in relation2id.items():
            if relation in embedding_dict:
                relation_embeddings[idx] = embedding_dict[relation]
            else:
                print(f"警告: 关系 '{relation}' 在预训练嵌入中不存在，使用随机初始化")
                relation_embeddings[idx] = np.random.uniform(-0.1, 0.1, args.emb_dim)
    else:
        # 使用MiniLM编码器生成嵌入
        print("使用MiniLM编码器生成实体和关系嵌入...")
        
        # 初始化MiniLM模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('/home/cqjtu/LLMs/all-MiniLM-L6-v2')
        minilm_model = AutoModel.from_pretrained('/home/cqjtu/LLMs/all-MiniLM-L6-v2')
        minilm_model.to(device)
        minilm_model.eval()
        for param in minilm_model.parameters():
            param.requires_grad = False
        
        # 加载实体到文本的映射字典
        entity_prop_2_anno_dict = {}
        entity_to_text_path = os.path.join(args.related_file_save_path, 'entity_to_text_dict.pkl')
        if os.path.exists(entity_to_text_path):
            with open(entity_to_text_path, 'rb') as f:
                entity_prop_2_anno_dict = pickle.load(f)
            print(f"加载实体到文本映射字典，包含 {len(entity_prop_2_anno_dict)} 个条目")
        else:
            print("警告: 未找到entity_to_text_dict.pkl文件，将直接使用URI")
        
        # 生成实体嵌入
        entity_embeddings = np.zeros((len(entity2id), args.emb_dim))
        print("生成实体嵌入...")
        for entity, idx in tqdm(entity2id.items(), desc="处理实体"):
            entity_embeddings[idx] = generate_minilm_embedding(
                entity, entity_prop_2_anno_dict, tokenizer, minilm_model, device, args.emb_dim
            )
        
        # 生成关系嵌入
        relation_embeddings = np.zeros((len(relation2id), args.emb_dim))
        print("生成关系嵌入...")
        for relation, idx in tqdm(relation2id.items(), desc="处理关系"):
            relation_embeddings[idx] = generate_minilm_embedding(
                relation, entity_prop_2_anno_dict, tokenizer, minilm_model, device, args.emb_dim
            )
        
        print(f"完成嵌入生成: 实体 {len(entity2id)} 个, 关系 {len(relation2id)} 个")

    # 5. 创建数据集和数据加载器
    # 5.1 划分数据集
    train_ratio = 0.7
    valid_ratio = 0.1
    test_ratio = 0.2
    
    # 划分subClassOf三元组
    if subclassof_triples:
        subclassof_train, temp_data = train_test_split(subclassof_triples, test_size=(1-train_ratio))
        subclassof_valid, subclassof_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))
        print(f"SubClassOf三元组划分: 训练集 {len(subclassof_train)}, 验证集 {len(subclassof_valid)}, 测试集 {len(subclassof_test)}")
    else:
        subclassof_train, subclassof_valid, subclassof_test = [], [], []
    
    # 划分instanceOf三元组
    if instanceof_triples:
        instanceof_train, temp_data = train_test_split(instanceof_triples, test_size=(1-train_ratio))
        instanceof_valid, instanceof_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))
        print(f"InstanceOf三元组划分: 训练集 {len(instanceof_train)}, 验证集 {len(instanceof_valid)}, 测试集 {len(instanceof_test)}")
    else:
        instanceof_train, instanceof_valid, instanceof_test = [], [], []
    
    # 划分普通关系三元组
    if relation_triples:
        relation_train, temp_data = train_test_split(relation_triples, test_size=(1-train_ratio))
        relation_valid, relation_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))
        print(f"普通关系三元组划分: 训练集 {len(relation_train)}, 验证集 {len(relation_valid)}, 测试集 {len(relation_test)}")
    else:
        relation_train, relation_valid, relation_test = [], [], []
    
    # 5.2 创建数据集
    # HRT数据集
    hrt_train_dataset = HRTDataset(relation_train, entity2id, relation2id, len(entity2id), len(relation2id))
    hrt_valid_dataset = HRTDataset(relation_valid, entity2id, relation2id, len(entity2id), len(relation2id))
    
    # SubClassOf数据集
    subclassof_train_dataset = SubClassOfDataset(subclassof_train, concept2id, len(concept2id))
    subclassof_valid_dataset = SubClassOfDataset(subclassof_valid, concept2id, len(concept2id))
    
    # InstanceOf数据集 (需要补充完整)
    class InstanceOfDataset(Dataset):
        def __init__(self, triples, entity2id, concept2id, num_entities, num_concepts, neg_ratio=1):
            """
            InstanceOf关系数据集
            triples: 三元组列表 [(instance, 'type', concept), ...]
            entity2id: 实体到ID的映射
            concept2id: 概念到ID的映射
            num_entities: 实体总数
            num_concepts: 概念总数
            neg_ratio: 每个正例对应的负例数量
            """
            self.triples = triples
            self.entity2id = entity2id
            self.concept2id = concept2id
            self.num_entities = num_entities
            self.num_concepts = num_concepts
            self.neg_ratio = neg_ratio
            
        def __len__(self):
            return len(self.triples)
            
        def __getitem__(self, idx):
            inst, _, concept = self.triples[idx]
            
            # 转换为ID
            inst_id = self.entity2id.get(inst, 0)
            concept_id = self.concept2id.get(concept, 0)
            
            # 负采样 (随机替换实例)
            inst_neg_id = np.random.randint(0, self.num_entities)
            while inst_neg_id == inst_id:  # 避免采样到相同的实体
                inst_neg_id = np.random.randint(0, self.num_entities)
            
            # 负采样 (随机替换概念)
            concept_neg_id = np.random.randint(0, self.num_concepts)
            while concept_neg_id == concept_id:  # 避免采样到相同的概念
                concept_neg_id = np.random.randint(0, self.num_concepts)
            
            return torch.tensor(inst_id), torch.tensor(concept_id), torch.tensor(inst_neg_id), torch.tensor(concept_neg_id)
    
    instanceof_train_dataset = InstanceOfDataset(instanceof_train, entity2id, concept2id, len(entity2id), len(concept2id))
    instanceof_valid_dataset = InstanceOfDataset(instanceof_valid, entity2id, concept2id, len(entity2id), len(concept2id))
    
    # 5.3 创建数据加载器
    batch_size = args.batch_size
    
    hrt_train_loader = DataLoader(hrt_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    hrt_valid_loader = DataLoader(hrt_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    subclassof_train_loader = DataLoader(subclassof_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    subclassof_valid_loader = DataLoader(subclassof_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    instanceof_train_loader = DataLoader(instanceof_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    instanceof_valid_loader = DataLoader(instanceof_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 6. 创建模型
    model = EIKETrain(
        args=args,
        num_instances=len(entity2id),
        num_concepts=len(concept2id),
        num_relations=len(relation2id),
        initial_entity_embeddings=entity_embeddings,
        initial_relation_embeddings=relation_embeddings,
        device=device
    )
    
    # 将ID映射添加到模型中，以便保存
    model.entity2id = entity2id
    model.relation2id = relation2id
    model.concept2id = concept2id
    
    model.to(device)
    
    # 7. 训练模型
    train_dataloaders = {
        'hrt': hrt_train_loader if len(relation_train) > 0 else None,
        'subclass_of': subclassof_train_loader if len(subclassof_train) > 0 else None,
        'instance_of': instanceof_train_loader if len(instanceof_train) > 0 else None
    }
    
    valid_dataloaders = {
        'hrt': hrt_valid_loader if len(relation_valid) > 0 else None,
        'subclass_of': subclassof_valid_loader if len(subclassof_valid) > 0 else None,
        'instance_of': instanceof_valid_loader if len(instanceof_valid) > 0 else None
    }
    
    print("开始训练模型...")
    model.run_training(train_dataloaders, valid_dataloaders)
    
    print("EKIE_DEMO.py 运行完毕。")