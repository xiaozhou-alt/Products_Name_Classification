import os
import re
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
from matplotlib.font_manager import FontProperties, findSystemFonts
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
def setup_chinese_font():
    """设置中文字体，确保中文正常显示，提供多种备选字体，适配Kaggle环境"""
    font_options = [
        {
            "name": "SimHei",
            "system_paths": [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/simhei/SimHei.ttf",
                "/Library/Fonts/SimHei.ttf",  # macOS路径
                "C:/Windows/Fonts/simhei.ttf"  # Windows路径
            ],
            "urls": [
                "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
                "https://gitee.com/chenguanzhou/simhei/raw/master/SimHei.ttf",
                "https://file.bugscaner.com/fonts/SimHei.ttf",
                "https://mirror.tuna.tsinghua.edu.cn/help/fonts/"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "SimHei.ttf")
        },
        {
            "name": "WenQuanYi Micro Hei",
            "system_paths": [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
            ],
            "urls": [
                "https://packages.debian.org/sid/all/fonts-wqy-microhei/download",
                "https://mirrors.ustc.edu.cn/debian/pool/main/f/fonts-wqy-microhei/fonts-wqy-microhei_0.2.0-beta-3_all.deb"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "wqy-microhei.ttc")
        },
        {
            "name": "Heiti TC",
            "system_paths": ["/Library/Fonts/Heiti TC.ttc"],
            "local_path": os.path.join("/kaggle/working/fonts", "Heiti TC.ttc")
        },
        {
            "name": "Noto Sans CJK SC",
            "system_paths": [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
            ],
            "urls": [
                "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/NotoSansCJKsc-Regular.otf"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "NotoSansCJKsc-Regular.otf")
        }
    ]
    
    system_fonts = findSystemFonts()
    
    for font in font_options:
        font_path = None
        
        for path in font["system_paths"]:
            if os.path.exists(path):
                font_path = path
                break
        
        if not font_path:
            for path in system_fonts:
                try:
                    if font["name"].lower() in path.lower() or \
                       any(alt.lower() in path.lower() for alt in font.get("alternatives", [])):
                        font_path = path
                        break
                except:
                    continue
        
        if font_path:
            try:
                font_prop = FontProperties(fname=font_path)
                plt.text(0, 0, "测试", fontproperties=font_prop)
                plt.close()
                
                print(f"成功加载中文字体: {font['name']} ({font_path})")
                
                plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                plt.rcParams['axes.unicode_minus'] = False
                return font_prop
            except:
                print(f"字体 {font['name']} 存在但无法使用，尝试下一种...")
                continue
        
        if 'urls' in font and font["urls"]:
            try:
                font_dir = os.path.dirname(font["local_path"])
                Path(font_dir).mkdir(parents=True, exist_ok=True)
                
                downloaded = False
                for url in font["urls"]:
                    for attempt in range(3):
                        try:
                            print(f"系统中未找到{font['name']}字体，正在从 {url} 下载（尝试 {attempt+1}/3）...")
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            response = requests.get(url, headers=headers, timeout=60)
                            response.raise_for_status()
                            
                            with open(font["local_path"], "wb") as f:
                                f.write(response.content)
                            
                            if os.path.getsize(font["local_path"]) < 1024 * 100:
                                raise Exception("下载的字体文件不完整")
                                
                            downloaded = True
                            break
                        except Exception as e:
                            print(f"从 {url} 下载{font['name']}失败（尝试 {attempt+1}/3）: {str(e)}")
                            if attempt == 2:
                                continue
                            time.sleep(2)
                    if downloaded:
                        break
                
                if not downloaded:
                    raise Exception("所有下载链接都失败了")
                
                if findSystemFonts([font["local_path"]]):
                    font_prop = FontProperties(fname=font["local_path"])
                    print(f"成功下载并加载{font['name']}字体: {font['local_path']}")
                    
                    plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                    plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                    plt.rcParams['axes.unicode_minus'] = False
                    return font_prop
                else:
                    try:
                        font_prop = FontProperties(fname=font["local_path"])
                        plt.text(0, 0, "测试", fontproperties=font_prop)
                        plt.close()
                        print(f"成功加载下载的{font['name']}字体: {font['local_path']}")
                        plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                        plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                        plt.rcParams['axes.unicode_minus'] = False
                        return font_prop
                    except:
                        raise Exception(f"下载的{font['name']}字体无法被系统识别")
                        
            except Exception as e:
                print(f"加载或下载{font['name']}字体失败: {str(e)}")
                continue
    
    try:
        print("尝试使用Kaggle预装的Noto字体...")
        noto_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if os.path.exists(noto_font_path):
            font_prop = FontProperties(fname=noto_font_path)
            plt.rcParams["font.family"] = ["Noto Sans CJK SC", "sans-serif"]
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "sans-serif"]
            plt.rcParams['axes.unicode_minus'] = False
            print("成功使用Noto Sans CJK SC字体")
            return font_prop
    except:
        pass
    
    print("警告: 所有中文字体加载失败，中文可能无法正常显示")
    print("尝试使用系统默认字体...")
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC"]
    plt.rcParams['axes.unicode_minus'] = False
    return FontProperties()


# 测试中文字体显示
def test_chinese_font_display(chinese_font):
    plt.figure(figsize=(8, 4))
    plt.title("中文字体测试 - 商品分类系统", fontproperties=chinese_font)
    plt.xlabel("X轴标签（中文测试）", fontproperties=chinese_font)
    plt.ylabel("Y轴标签（中文测试）", fontproperties=chinese_font)
    plt.text(0.5, 0.5, "测试文字：商品分类 ABC123", 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=12, fontproperties=chinese_font)
    plt.plot([1, 2, 3], [4, 5, 6], label="示例曲线")
    
    plt.legend(
        title="图例（中文）", 
        prop=chinese_font,
        title_fontproperties=chinese_font
    )
    
    test_img_path = os.path.join('/kaggle/working/', 'chinese_font_test.png')
    plt.savefig(test_img_path, dpi=300)
    print(f"中文字体测试图像已保存至: {test_img_path}")
    plt.close()
    
    return test_img_path

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 文本增强函数
def augment_text(text, prob=0.2):
    """对中文文本进行简单的数据增强"""
    if not text or random.random() > prob:
        return text
    
    # 随机插入同义词
    words = list(text)
    if len(words) < 3:
        return text
        
    # 随机选择插入位置
    insert_pos = random.randint(1, len(words)-1)
    
    # 简单同义词替换表（可以扩展）
    synonyms = {
        '好': ['佳', '优', '良'],
        '大': ['巨', '庞', '硕'],
        '小': ['微', '细', '迷你'],
        '高': ['昂', '贵'],
        '低': ['廉', '便宜'],
        '新': ['鲜', '全新'],
        '旧': ['老', '古'],
        '多': ['丰', '众'],
        '少': ['寡', '稀']
    }
    
    # 尝试找到可替换的词
    for i in range(len(words)):
        if words[i] in synonyms:
            # 随机选择一个同义词插入
            synonym = random.choice(synonyms[words[i]])
            words.insert(insert_pos, synonym)
            return ''.join(words)
    
    # 随机交换两个字符
    swap_pos1 = random.randint(0, len(words)-1)
    swap_pos2 = random.randint(0, len(words)-1)
    if swap_pos1 != swap_pos2:
        words[swap_pos1], words[swap_pos2] = words[swap_pos2], words[swap_pos1]
        return ''.join(words)
    
    return text

# 1. 数据加载与处理
def load_data(root_dir):
    data = []
    for main_category in os.listdir(root_dir):
        main_path = os.path.join(root_dir, main_category)
        if not os.path.isdir(main_path):
            continue
            
        for split in ['sample_train.txt', 'sample_test.txt']:
            file_path = os.path.join(main_path, split)
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                        
                    primary_label = parts[0]
                    secondary_part = parts[1]
                    secondary_label = secondary_part.split('@')[-1]
                    product_name = ','.join(parts[2:])
                    
                    data.append({
                        'main_category': main_category,
                        'split': 'train' if split == 'sample_train.txt' else 'test',
                        'primary_label': primary_label,
                        'secondary_label': secondary_label,
                        'product_name': product_name
                    })
    
    return pd.DataFrame(data)

def balance_dataset(df, samples_per_category=100000):
    balanced_dfs = []
    
    for category in df['main_category'].unique():
        category_df = df[df['main_category'] == category].copy()
        
        total_available = len(category_df)
        total_target = min(samples_per_category, total_available)
        print(f"主类别 '{category}' 原始样本数: {total_available}, 将保留: {total_target}")
        
        all_samples = category_df.sample(total_target, random_state=42)
        
        train_size = int(total_target * 0.8)
        test_size = total_target - train_size
        
        if test_size < 1 and total_target >= 1:
            test_size = 1
            train_size = max(0, total_target - test_size)
        
        train_samples = all_samples.iloc[:train_size].copy()
        test_samples = all_samples.iloc[train_size:train_size+test_size].copy()
        
        train_samples['split'] = 'train'
        test_samples['split'] = 'test'
        
        balanced_dfs.append(train_samples)
        balanced_dfs.append(test_samples)
        
        print(f"  训练集: 保留 {len(train_samples)} 条")
        print(f"  测试集: 保留 {len(test_samples)} 条")
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\n所有类别总样本数: {len(balanced_df)}")
    return balanced_df

# 2. 数据统计
def analyze_data(df):
    print("数据集基本统计:")
    print(f"总样本数: {len(df)}")
    
    print("\n按主类别统计:")
    print(df['main_category'].value_counts())
    
    print("\n按训练/测试集统计:")
    print(df['split'].value_counts())
    
    print("\n一级标签数量:")
    print(df['primary_label'].nunique())
    print(df['primary_label'].value_counts().head(10))
    
    print("\n二级标签数量:")
    print(df['secondary_label'].nunique())
    print(df['secondary_label'].value_counts().head(10))
    
    print("\n按主类别和训练/测试集统计:")
    print(pd.crosstab(df['main_category'], df['split']))

# 3. 数据集和数据加载器
class ProductDataset(Dataset):
    def __init__(self, texts, primary_labels, secondary_labels, tokenizer, max_len=128, augment=False):
        self.texts = texts
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 训练时应用数据增强
        if self.augment:
            text = augment_text(text)
        
        primary_label = self.primary_labels[idx]
        secondary_label = self.secondary_labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'primary_label': torch.tensor(primary_label, dtype=torch.long),
            'secondary_label': torch.tensor(secondary_label, dtype=torch.long)
        }

def create_data_loaders(train_df, test_df, tokenizer, batch_size=32, max_len=128, use_weighted_sampler=False):
    train_dataset = ProductDataset(
        texts=train_df['product_name'].values,
        primary_labels=train_df['primary_encoded'].values,
        secondary_labels=train_df['secondary_encoded'].values,
        tokenizer=tokenizer,
        max_len=max_len,
        augment=True  # 训练集使用数据增强
    )
    
    test_dataset = ProductDataset(
        texts=test_df['product_name'].values,
        primary_labels=test_df['primary_encoded'].values,
        secondary_labels=test_df['secondary_encoded'].values,
        tokenizer=tokenizer,
        max_len=max_len,
        augment=False  # 测试集不使用数据增强
    )
    
    # 使用加权采样器处理类别不平衡
    if use_weighted_sampler:
        # 计算一级标签的权重
        class_counts = torch.bincount(torch.tensor(train_df['primary_encoded'].values))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_df['primary_encoded'].values]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader

# 4. 模型定义 - 更复杂的架构
class ProductClassifier(nn.Module):
    """改进的商品分类模型，使用更复杂的结构和注意力机制"""
    def __init__(self, model_name, num_primary_labels, num_secondary_labels, dropout=0.3):
        super(ProductClassifier, self).__init__()
        # 根据模型名称选择合适的预训练模型
        if model_name.startswith('roberta'):
            self.bert = RobertaModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
            
        self.hidden_size = self.bert.config.hidden_size
        
        # 添加额外的全连接层增加模型深度
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        
        # 使用批归一化
        self.bn1 = nn.BatchNorm1d(self.hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size // 4)
        
        # 使用更复杂的dropout策略
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        
        # GELU激活函数比ReLU效果更好
        self.activation = nn.GELU()
        
        # 输出层
        self.fc_primary = nn.Linear(self.hidden_size // 4, num_primary_labels)
        self.fc_secondary = nn.Linear(self.hidden_size // 4, num_secondary_labels)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化自定义层的权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_primary.weight)
        nn.init.xavier_uniform_(self.fc_secondary.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_primary.bias)
        nn.init.zeros_(self.fc_secondary.bias)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True  # 返回注意力权重
        )
        
        # 使用最后一层的[CLS] token输出
        cls_output = outputs.pooler_output
        
        # 通过额外的全连接层
        x = self.dropout1(cls_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.dropout3(x)
        
        # 一级标签预测
        primary_logits = self.fc_primary(x)
        
        # 二级标签预测
        secondary_logits = self.fc_secondary(x)
        
        return primary_logits, secondary_logits, outputs.attentions

# 5. 标签平滑损失函数
class LabelSmoothingLoss(nn.Module):
    """标签平滑损失，减少过拟合风险"""
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 6. 训练函数
def train_epoch(model, data_loader, loss_fn_primary, loss_fn_secondary, optimizer, 
                device, scheduler, n_examples, scaler, accumulation_steps=1):
    model = model.train()
    
    losses = []
    correct_primary = 0
    correct_secondary = 0
    
    optimizer.zero_grad()
    loop = tqdm(data_loader, total=len(data_loader), leave=True)
    
    for step, batch in enumerate(loop):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        primary_labels = batch["primary_label"].to(device)
        secondary_labels = batch["secondary_label"].to(device)
        
        with autocast():
            primary_logits, secondary_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss_primary = loss_fn_primary(primary_logits, primary_labels)
            loss_secondary = loss_fn_secondary(secondary_logits, secondary_labels)
            loss = loss_primary + loss_secondary  # 总损失
            loss = loss / accumulation_steps  # 梯度累积
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # 计算准确率
        primary_preds = torch.argmax(primary_logits, dim=1)
        secondary_preds = torch.argmax(secondary_logits, dim=1)
        
        correct_primary += torch.sum(primary_preds == primary_labels)
        correct_secondary += torch.sum(secondary_preds == secondary_labels)
        
        losses.append(loss.item() * accumulation_steps)
        
        # 更新进度条
        loop.set_description(f"训练中")
        loop.set_postfix(
            loss=np.mean(losses),
            primary_acc=correct_primary.double() / n_examples,
            secondary_acc=correct_secondary.double() / n_examples
        )
    
    return (
        correct_primary.double() / n_examples,
        correct_secondary.double() / n_examples,
        np.mean(losses)
    )

# 7. 评估函数
def eval_model(model, data_loader, loss_fn_primary, loss_fn_secondary, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_primary = 0
    correct_secondary = 0
    
    all_primary_preds = []
    all_secondary_preds = []
    all_primary_labels = []
    all_secondary_labels = []
    all_texts = []
    
    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            primary_labels = batch["primary_label"].to(device)
            secondary_labels = batch["secondary_label"].to(device)
            texts = batch["text"]
            
            primary_logits, secondary_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss_primary = loss_fn_primary(primary_logits, primary_labels)
            loss_secondary = loss_fn_secondary(secondary_logits, secondary_labels)
            loss = loss_primary + loss_secondary
            
            primary_preds = torch.argmax(primary_logits, dim=1)
            secondary_preds = torch.argmax(secondary_logits, dim=1)
            
            correct_primary += torch.sum(primary_preds == primary_labels)
            correct_secondary += torch.sum(secondary_preds == secondary_labels)
            
            losses.append(loss.item())
            
            # 收集所有预测和标签用于混淆矩阵
            all_primary_preds.extend(primary_preds.cpu().numpy())
            all_secondary_preds.extend(secondary_preds.cpu().numpy())
            all_primary_labels.extend(primary_labels.cpu().numpy())
            all_secondary_labels.extend(secondary_labels.cpu().numpy())
            all_texts.extend(texts)
            
            # 更新进度条
            loop.set_description(f"评估中")
            loop.set_postfix(
                loss=np.mean(losses),
                primary_acc=correct_primary.double() / n_examples,
                secondary_acc=correct_secondary.double() / n_examples
            )
    
    return (
        correct_primary.double() / n_examples,
        correct_secondary.double() / n_examples,
        np.mean(losses),
        all_primary_preds, all_primary_labels,
        all_secondary_preds, all_secondary_labels,
        all_texts
    )

# 8. 改进的早停机制
class EarlyStopping:
    """改进的早停机制，监控多个指标"""
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor  # 可以是'val_loss', 'val_primary_acc', 'val_secondary_acc'
        self.best_score = None
        self.early_stop = False
        self.count = 0
        self.best_model_weights = None
        
    def __call__(self, metrics, model):
        # 根据监控指标确定分数
        if self.monitor == 'val_loss':
            score = -metrics['val_loss']  # 损失越小越好
        elif self.monitor == 'val_primary_acc':
            score = metrics['val_primary_acc']  # 准确率越高越好
        elif self.monitor == 'val_secondary_acc':
            score = metrics['val_secondary_acc']  # 准确率越高越好
        else:
            raise ValueError(f"不支持的监控指标: {self.monitor}")
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.count += 1
            print(f"早停计数: {self.count}/{self.patience}")
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = model.state_dict()
            self.count = 0

# 9. 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, title, chinese_font, figsize=(16, 14), 
                          normalize=False, show_values=True, max_labels=15):
    cm = confusion_matrix(y_true, y_pred)
    
    if len(labels) > max_labels:
        label_counts = np.sum(cm, axis=1) + np.sum(cm, axis=0)
        top_indices = np.argsort(label_counts)[-max_labels:]
        cm = cm[np.ix_(top_indices, top_indices)]
        labels = [labels[i] for i in top_indices]
        title += f"（仅显示出现次数最多的{max_labels}个标签）"
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cbar_label = '比例'
    else:
        fmt = 'd'
        cbar_label = '数量'
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    im = sns.heatmap(cm, annot=show_values, fmt=fmt, cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     ax=ax, cbar_kws={'label': cbar_label},
                     annot_kws={"size": 10, "color": "black"})
    
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontproperties=chinese_font, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.title(title, fontproperties=chinese_font, fontsize=16, pad=20)
    plt.xlabel('预测标签', fontproperties=chinese_font, fontsize=14, labelpad=10)
    plt.ylabel('真实标签', fontproperties=chinese_font, fontsize=14, labelpad=10)
    
    plt.xticks(fontproperties=chinese_font, fontsize=8, rotation=45, ha='right')
    plt.yticks(fontproperties=chinese_font, fontsize=8, rotation=0)
    
    plt.subplots_adjust(bottom=0.25, right=0.95)
    
    filename = f'{title.replace(" ", "_").replace("（", "").replace("）", "")}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# 10. 错误分析
def analyze_errors(texts, true_labels, pred_labels, label_names, chinese_font, top_n=20):
    """分析模型预测错误的样本，找出最容易混淆的类别"""
    errors = []
    for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
        if true_label != pred_label:
            errors.append({
                'text': text,
                'true_label': label_names[true_label],
                'pred_label': label_names[pred_label],
                'true_idx': true_label,
                'pred_idx': pred_label
            })
    
    if not errors:
        print("没有错误样本可分析")
        return
    
    error_df = pd.DataFrame(errors)
    print(f"总错误样本数: {len(error_df)}")
    
    # 保存错误样本到文件
    error_df.to_csv('prediction_errors.csv', index=False, encoding='utf-8')
    print("错误样本已保存至 prediction_errors.csv")
    
    # 分析最常见的错误类型
    error_pairs = error_df.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    error_pairs = error_pairs.sort_values('count', ascending=False)
    
    print("\n最常见的错误类型:")
    print(error_pairs.head(10))
    
    # 可视化最常见的错误对
    plt.figure(figsize=(12, 8))
    top_error_pairs = error_pairs.head(top_n)
    pairs = [f"{t} → {p}" for t, p in zip(top_error_pairs['true_label'], top_error_pairs['pred_label'])]
    
    sns.barplot(x='count', y=pairs, data=top_error_pairs)
    plt.title(f'最常见的错误分类对（Top {top_n}）', fontproperties=chinese_font, fontsize=14)
    plt.xlabel('错误数量', fontproperties=chinese_font)
    plt.ylabel('错误分类对', fontproperties=chinese_font)
    plt.yticks(fontproperties=chinese_font, fontsize=8)
    plt.tight_layout()
    plt.savefig('top_error_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()

# 11. 主训练函数
def train_model(data_path, epochs=15, batch_size=32, max_len=128, learning_rate=2e-5, 
                samples_per_category=100000, model_name='hfl/chinese-roberta-wwm-ext',
                use_weighted_sampler=True, label_smoothing=0.1, accumulation_steps=2):
    # 设置中文字体
    print("设置中文字体...")
    chinese_font = setup_chinese_font()
    # 测试中文字体显示
    test_chinese_font_display(chinese_font)
    
    # 加载数据
    print("加载数据中...")
    df = load_data(data_path)
    
    # 平衡数据集
    print(f"平衡数据集 - 每个主类别保留{samples_per_category}条样本...")
    df = balance_dataset(df, samples_per_category=samples_per_category)
    
    # 数据分析
    analyze_data(df)
    
    # 分割训练集和验证集（从训练集中再划分出验证集）
    train_val_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # 从训练集中划分出20%作为验证集
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_val_df['primary_label']  # 保持分层抽样
    )
    
    print(f"\n训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    # 标签编码
    print("编码标签...")
    primary_le = LabelEncoder()
    secondary_le = LabelEncoder()
    
    # 拟合编码器（只使用训练集）
    primary_le.fit(train_df['primary_label'].unique())
    secondary_le.fit(train_df['secondary_label'].unique())
    
    # 编码标签
    train_df['primary_encoded'] = primary_le.transform(train_df['primary_label'])
    train_df['secondary_encoded'] = secondary_le.transform(train_df['secondary_label'])
    
    val_df['primary_encoded'] = primary_le.transform(val_df['primary_label'])
    val_df['secondary_encoded'] = secondary_le.transform(val_df['secondary_label'])
    
    test_df['primary_encoded'] = primary_le.transform(test_df['primary_label'])
    test_df['secondary_encoded'] = secondary_le.transform(test_df['secondary_label'])
    
    # 保存类别信息
    num_primary_labels = len(primary_le.classes_)
    num_secondary_labels = len(secondary_le.classes_)
    
    print(f"一级标签数量: {num_primary_labels}")
    print(f"二级标签数量: {num_secondary_labels}")
    
    # 加载tokenizer
    print(f"加载{model_name}模型和Tokenizer...")
    if model_name.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, _ = create_data_loaders(
        train_df, val_df, tokenizer, batch_size, max_len, use_weighted_sampler
    )
    
    val_loader, test_loader = create_data_loaders(
        val_df, test_df, tokenizer, batch_size, max_len, use_weighted_sampler=False
    )
    
    # 初始化模型
    model = ProductClassifier(
        model_name,
        num_primary_labels,
        num_secondary_labels,
        dropout=0.3
    )
    model = model.to(device)
    
    # 定义损失函数（使用标签平滑）和优化器
    if label_smoothing > 0:
        loss_fn_primary = LabelSmoothingLoss(num_primary_labels, smoothing=label_smoothing).to(device)
        loss_fn_secondary = LabelSmoothingLoss(num_secondary_labels, smoothing=label_smoothing).to(device)
    else:
        loss_fn_primary = nn.CrossEntropyLoss().to(device)
        loss_fn_secondary = nn.CrossEntropyLoss().to(device)
    
    # 使用AdamW优化器，设置权重衰减
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    # 学习率调度器 - 使用余弦退火调度器
    total_steps = len(train_loader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * 0.1)  # 10%的步骤用于预热
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 早停机制 - 监控验证集的平均准确率
    early_stopping = EarlyStopping(
        patience=4, 
        min_delta=0.001,
        monitor='val_secondary_acc'  # 监控二级准确率，因为它相对较低
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练历史记录
    history = {
        'train_primary_acc': [],
        'train_secondary_acc': [],
        'train_loss': [],
        'val_primary_acc': [],
        'val_secondary_acc': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # 开始训练
    print("开始训练...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        
        # 训练
        train_primary_acc, train_secondary_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn_primary,
            loss_fn_secondary,
            optimizer,
            device,
            scheduler,
            len(train_df),
            scaler,
            accumulation_steps=accumulation_steps
        )
        
        print(f"训练集: 一级准确率 {train_primary_acc:.4f}, 二级准确率 {train_secondary_acc:.4f}, 损失 {train_loss:.4f}")
        
        # 在验证集上评估
        val_primary_acc, val_secondary_acc, val_loss, _, _, _, _, _ = eval_model(
            model,
            val_loader,
            loss_fn_primary,
            loss_fn_secondary,
            device,
            len(val_df)
        )
        
        print(f"验证集: 一级准确率 {val_primary_acc:.4f}, 二级准确率 {val_secondary_acc:.4f}, 损失 {val_loss:.4f}")
        
        # 记录历史
        history['train_primary_acc'].append(train_primary_acc.item())
        history['train_secondary_acc'].append(train_secondary_acc.item())
        history['train_loss'].append(train_loss)
        history['val_primary_acc'].append(val_primary_acc.item())
        history['val_secondary_acc'].append(val_secondary_acc.item())
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 早停检查
        metrics = {
            'val_loss': val_loss,
            'val_primary_acc': val_primary_acc.item(),
            'val_secondary_acc': val_secondary_acc.item()
        }
        early_stopping(metrics, model)
        if early_stopping.early_stop:
            print("触发早停机制")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(early_stopping.best_model_weights)
    
    # 最终在测试集上评估
    print("最终在测试集上评估...")
    final_primary_acc, final_secondary_acc, final_loss, \
    primary_preds, primary_labels, secondary_preds, secondary_labels, texts = eval_model(
        model,
        test_loader,
        loss_fn_primary,
        loss_fn_secondary,
        device,
        len(test_df)
    )
    
    print(f"最终测试集: 一级准确率 {final_primary_acc:.4f}, 二级准确率 {final_secondary_acc:.4f}")
    print(f"平均准确率: {(final_primary_acc + final_secondary_acc) / 2:.4f}")
    
    # 生成详细的分类报告
    print("\n一级标签分类报告:")
    print(classification_report(
        primary_labels, 
        primary_preds, 
        target_names=primary_le.classes_,
        digits=4
    ))
    
    print("\n二级标签分类报告:")
    print(classification_report(
        secondary_labels, 
        secondary_preds, 
        target_names=secondary_le.classes_,
        digits=4
    ))
    
    # 错误分析
    print("\n进行错误分析...")
    analyze_errors(
        texts, 
        primary_labels, 
        primary_preds, 
        primary_le.classes_,
        chinese_font
    )
    
    analyze_errors(
        texts, 
        secondary_labels, 
        secondary_preds, 
        secondary_le.classes_,
        chinese_font
    )
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    top_n = 15
    
    # 一级标签混淆矩阵
    plot_confusion_matrix(
        primary_labels,
        primary_preds,
        primary_le.classes_,
        f'一级标签混淆矩阵（Top {top_n}）',
        chinese_font,
        max_labels=top_n,
        show_values=True
    )
    
    # 二级标签混淆矩阵
    plot_confusion_matrix(
        secondary_labels,
        secondary_preds,
        secondary_le.classes_,
        f'二级标签混淆矩阵（Top {top_n}）',
        chinese_font,
        max_labels=top_n,
        show_values=True,
        figsize=(18, 16)
    )
    
    # 绘制训练历史
    print("绘制训练历史...")
    plt.figure(figsize=(16, 10))
    
    # 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_primary_acc'], label='训练集一级准确率')
    plt.plot(history['val_primary_acc'], label='验证集一级准确率')
    plt.title('一级标签准确率曲线', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('准确率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_secondary_acc'], label='训练集二级准确率')
    plt.plot(history['val_secondary_acc'], label='验证集二级准确率')
    plt.title('二级标签准确率曲线', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('准确率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 损失曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_loss'], label='训练集损失')
    plt.plot(history['val_loss'], label='验证集损失')
    plt.title('损失曲线', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('损失', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 学习率曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='学习率')
    plt.title('学习率变化', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('学习率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_history.png')
    plt.close()
    
    # 随机抽取测试样本进行预测展示
    print("随机抽取测试样本进行预测展示...")
    sample_size = 5
    samples = []
    
    for category in df['main_category'].unique():
        category_samples = test_df[test_df['main_category'] == category].sample(sample_size, random_state=42)
        samples.append(category_samples)
    
    sample_df = pd.concat(samples, ignore_index=True)
    
    # 创建样本数据加载器
    sample_dataset = ProductDataset(
        texts=sample_df['product_name'].values,
        primary_labels=sample_df['primary_encoded'].values,
        secondary_labels=sample_df['secondary_encoded'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=sample_size * len(df['main_category'].unique()),
        shuffle=False
    )
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        for batch in sample_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            primary_logits, secondary_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            primary_preds = torch.argmax(primary_logits, dim=1).cpu().numpy()
            secondary_preds = torch.argmax(secondary_logits, dim=1).cpu().numpy()
    
    # 解码预测结果
    sample_df['primary_pred'] = primary_le.inverse_transform(primary_preds)
    sample_df['secondary_pred'] = secondary_le.inverse_transform(secondary_preds)
    
    # 打印结果
    print("\n测试样本预测结果:")
    print("=" * 100)
    for _, row in sample_df.iterrows():
        print(f"商品名称: {row['product_name']}")
        print(f"主类别: {row['main_category']}")
        
        # 一级标签结果
        primary_correct = row['primary_label'] == row['primary_pred']
        primary_color = "\033[92m" if primary_correct else "\033[91m"
        print(f"一级标签 - 真实: {row['primary_label']}, 预测: {primary_color}{row['primary_pred']}\033[0m")
        
        # 二级标签结果
        secondary_correct = row['secondary_label'] == row['secondary_pred']
        secondary_color = "\033[92m" if secondary_correct else "\033[91m"
        print(f"二级标签 - 真实: {row['secondary_label']}, 预测: {secondary_color}{row['secondary_pred']}\033[0m")
        print("-" * 100)
    
    # 保存模型
    print("保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'primary_label_encoder': primary_le,
        'secondary_label_encoder': secondary_le,
        'tokenizer': tokenizer,
        'model_name': model_name
    }, '/kaggle/working/product_classifier_model.pt')
    
    print("训练完成!")
    
    return model, tokenizer, primary_le, secondary_le, history, chinese_font

# 模型集成函数（虽然只训练了一个模型，但保留该函数以便后续扩展）
def ensemble_predict(models, tokenizer, texts, max_len=128, device='cuda'):
    """使用多个模型进行集成预测"""
    all_primary_logits = []
    all_secondary_logits = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            encoding = tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            primary_logits, secondary_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            all_primary_logits.append(primary_logits.cpu().numpy())
            all_secondary_logits.append(secondary_logits.cpu().numpy())
    
    # 平均所有模型的logits
    avg_primary_logits = np.mean(all_primary_logits, axis=0)
    avg_secondary_logits = np.mean(all_secondary_logits, axis=0)
    
    # 获取预测结果
    primary_preds = np.argmax(avg_primary_logits, axis=1)
    secondary_preds = np.argmax(avg_secondary_logits, axis=1)
    
    return primary_preds, secondary_preds

# 主函数
if __name__ == "__main__":
    # 数据根目录路径
    data_directory = "/kaggle/input/product-categories"
    
    # 只训练第一个模型 - 使用RoBERTa
    print("训练模型(RoBERTa)...")
    model, tokenizer, primary_le, secondary_le, history, chinese_font = train_model(
        data_path=data_directory,
        epochs=3,
        batch_size=64,
        max_len=128,
        learning_rate=2e-5,
        samples_per_category=100000,
        model_name='hfl/chinese-roberta-wwm-ext',
        use_weighted_sampler=True,
        label_smoothing=0.1
    )
    
    print("模型训练完成!")