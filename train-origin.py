import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
from matplotlib.font_manager import FontProperties, findSystemFonts, get_font
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """设置中文字体，确保中文正常显示，提供多种备选字体，适配Kaggle环境"""
    # 中文字体选项，包含多种常用中文字体和更多下载源
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
                "https://file.bugscaner.com/fonts/SimHei.ttf",  # 新增源
                "https://mirror.tuna.tsinghua.edu.cn/help/fonts/"  # 新增源
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "SimHei.ttf")  # 修改为Kaggle工作目录
        },
        {
            "name": "WenQuanYi Micro Hei",
            "system_paths": [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
            ],
            "urls": [
                "https://packages.debian.org/sid/all/fonts-wqy-microhei/download",
                "https://mirrors.ustc.edu.cn/debian/pool/main/f/fonts-wqy-microhei/fonts-wqy-microhei_0.2.0-beta-3_all.deb"  # 新增源
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "wqy-microhei.ttc")  # 修改为Kaggle工作目录
        },
        {
            "name": "Heiti TC",  # macOS系统的黑体
            "system_paths": [
                "/Library/Fonts/Heiti TC.ttc"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "Heiti TC.ttc")
        },
        {
            "name": "Noto Sans CJK SC",  # 增加Noto字体作为备选
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
    
    # 检查系统中已安装的中文字体
    system_fonts = findSystemFonts()
    
    # 尝试每种字体
    for font in font_options:
        font_path = None
        
        # 首先检查已知的系统路径
        for path in font["system_paths"]:
            if os.path.exists(path):
                font_path = path
                break
        
        # 如果没找到，搜索所有系统字体
        if not font_path:
            for path in system_fonts:
                try:
                    if font["name"].lower() in path.lower() or \
                       any(alt.lower() in path.lower() for alt in font.get("alternatives", [])):
                        font_path = path
                        break
                except:
                    continue
        
        # 如果找到字体，直接使用
        if font_path:
            try:
                # 验证字体是否可用
                font_prop = FontProperties(fname=font_path)
                # 测试字体是否能正常工作
                plt.text(0, 0, "测试", fontproperties=font_prop)
                plt.close()
                
                print(f"成功加载中文字体: {font['name']} ({font_path})")
                
                # 应用字体设置
                plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                return font_prop
            except:
                print(f"字体 {font['name']} 存在但无法使用，尝试下一种...")
                continue
        
        # 如果系统中没有，尝试下载字体，增加重试机制
        if 'urls' in font and font["urls"]:
            try:
                # 创建字体目录，确保有写入权限
                font_dir = os.path.dirname(font["local_path"])
                Path(font_dir).mkdir(parents=True, exist_ok=True)
                
                # 尝试多个下载链接，每个链接最多重试2次
                downloaded = False
                for url in font["urls"]:
                    for attempt in range(3):  # 最多重试2次
                        try:
                            print(f"系统中未找到{font['name']}字体，正在从 {url} 下载（尝试 {attempt+1}/3）...")
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            # 增加超时时间
                            response = requests.get(url, headers=headers, timeout=60)
                            response.raise_for_status()
                            
                            # 保存字体文件
                            with open(font["local_path"], "wb") as f:
                                f.write(response.content)
                            
                            # 验证文件完整性
                            if os.path.getsize(font["local_path"]) < 1024 * 100:  # 至少100KB
                                raise Exception("下载的字体文件不完整")
                                
                            downloaded = True
                            break
                        except Exception as e:
                            print(f"从 {url} 下载{font['name']}失败（尝试 {attempt+1}/3）: {str(e)}")
                            if attempt == 2:  # 最后一次尝试失败
                                continue
                            time.sleep(2)  # 重试前等待2秒
                    if downloaded:
                        break
                
                if not downloaded:
                    raise Exception("所有下载链接都失败了")
                
                # 确认字体可以被系统识别
                if findSystemFonts([font["local_path"]]):
                    font_prop = FontProperties(fname=font["local_path"])
                    print(f"成功下载并加载{font['name']}字体: {font['local_path']}")
                    
                    # 应用字体设置
                    plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                    plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                    plt.rcParams['axes.unicode_minus'] = False
                    return font_prop
                else:
                    # 尝试直接使用路径，不依赖系统字体缓存
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
    
    # 最后的备选方案：使用Kaggle可能预装的Noto字体
    try:
        print("尝试使用Kaggle预装的Noto字体...")
        # 直接指定Noto字体路径
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
    
    # 如果所有字体都失败，尝试使用默认设置并警告
    print("警告: 所有中文字体加载失败，中文可能无法正常显示")
    print("尝试使用系统默认字体...")
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC"]
    plt.rcParams['axes.unicode_minus'] = False
    return FontProperties()


# 测试中文字体显示
def test_chinese_font_display(chinese_font):
    """测试中文字体是否能正常显示，接收字体属性作为参数"""
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

set_seed()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据加载与处理
def load_data(root_dir):
    """加载所有类别的训练集和测试集数据"""
    data = []
    # 遍历所有主文件夹
    for main_category in os.listdir(root_dir):
        main_path = os.path.join(root_dir, main_category)
        if not os.path.isdir(main_path):
            continue
            
        # 处理训练集和测试集
        for split in ['sample_train.txt', 'sample_test.txt']:
            file_path = os.path.join(main_path, split)
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 解析每行数据
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                        
                    # 提取第一类标签
                    primary_label = parts[0]
                    
                    # 提取第二类标签
                    secondary_part = parts[1]
                    secondary_label = secondary_part.split('@')[-1]
                    
                    # 提取商品名称
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
    """每个主类别文件夹保留指定数量的样本，并按4:1比例划分训练集和测试集"""
    balanced_dfs = []
    
    # 按主类别处理
    for category in df['main_category'].unique():
        # 获取该类别的所有样本
        category_df = df[df['main_category'] == category].copy()
        
        # 计算该类别应保留的总样本数（不超过原始数量）
        total_available = len(category_df)
        total_target = min(samples_per_category, total_available)
        print(f"主类别 '{category}' 原始样本数: {total_available}, 将保留: {total_target}")
        
        # 强制按4:1的比例划分训练集和测试集
        # 先合并所有数据，再重新划分
        all_samples = category_df.sample(total_target, random_state=42)
        
        # 计算划分数量
        train_size = int(total_target * 0.8)  # 80% 作为训练集
        test_size = total_target - train_size  # 20% 作为测试集
        
        # 确保至少有一个测试样本
        if test_size < 1 and total_target >= 1:
            test_size = 1
            train_size = max(0, total_target - test_size)
        
        # 重新划分训练集和测试集
        train_samples = all_samples.iloc[:train_size].copy()
        test_samples = all_samples.iloc[train_size:train_size+test_size].copy()
        
        # 更新split标签
        train_samples['split'] = 'train'
        test_samples['split'] = 'test'
        
        # 添加到结果列表
        balanced_dfs.append(train_samples)
        balanced_dfs.append(test_samples)
        
        print(f"  训练集: 保留 {len(train_samples)} 条")
        print(f"  测试集: 保留 {len(test_samples)} 条")
    
    # 合并所有类别的样本
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\n所有类别总样本数: {len(balanced_df)}")
    return balanced_df

# 2. 数据统计
def analyze_data(df):
    """分析数据集的基本情况"""
    print("数据集基本统计:")
    print(f"总样本数: {len(df)}")
    
    # 按主类别统计
    print("\n按主类别统计:")
    print(df['main_category'].value_counts())
    
    # 按训练/测试集统计
    print("\n按训练/测试集统计:")
    print(df['split'].value_counts())
    
    # 按一级标签统计
    print("\n一级标签数量:")
    print(df['primary_label'].nunique())
    print(df['primary_label'].value_counts().head(10))
    
    # 按二级标签统计
    print("\n二级标签数量:")
    print(df['secondary_label'].nunique())
    print(df['secondary_label'].value_counts().head(10))
    
    # 按主类别和训练/测试集统计
    print("\n按主类别和训练/测试集统计:")
    print(pd.crosstab(df['main_category'], df['split']))

# 3. 数据集和数据加载器
class ProductDataset(Dataset):
    """商品分类数据集"""
    def __init__(self, texts, primary_labels, secondary_labels, tokenizer, max_len=128):
        self.texts = texts
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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

def create_data_loaders(train_df, test_df, tokenizer, batch_size=32, max_len=128):
    """创建训练集和测试集的数据加载器"""
    train_dataset = ProductDataset(
        texts=train_df['product_name'].values,
        primary_labels=train_df['primary_encoded'].values,
        secondary_labels=train_df['secondary_encoded'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = ProductDataset(
        texts=test_df['product_name'].values,
        primary_labels=test_df['primary_encoded'].values,
        secondary_labels=test_df['secondary_encoded'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
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

# 4. 模型定义
class ProductClassifier(nn.Module):
    """商品分类模型，同时预测一级和二级标签"""
    def __init__(self, bert_model_name, num_primary_labels, num_secondary_labels, dropout=0.3):
        super(ProductClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc_primary = nn.Linear(self.bert.config.hidden_size, num_primary_labels)
        self.fc_secondary = nn.Linear(self.bert.config.hidden_size, num_secondary_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的输出
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        
        # 一级标签预测
        primary_logits = self.fc_primary(cls_output)
        
        # 二级标签预测
        secondary_logits = self.fc_secondary(cls_output)
        
        return primary_logits, secondary_logits

# 5. 训练函数
def train_epoch(model, data_loader, loss_fn_primary, loss_fn_secondary, optimizer, device, scheduler, n_examples, scaler):
    """训练一个epoch"""
    model = model.train()
    
    losses = []
    correct_primary = 0
    correct_secondary = 0
    
    loop = tqdm(data_loader, total=len(data_loader), leave=True)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        primary_labels = batch["primary_label"].to(device)
        secondary_labels = batch["secondary_label"].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            primary_logits, secondary_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss_primary = loss_fn_primary(primary_logits, primary_labels)
            loss_secondary = loss_fn_secondary(secondary_logits, secondary_labels)
            loss = loss_primary + loss_secondary  # 总损失
            
        # 反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 计算准确率
        primary_preds = torch.argmax(primary_logits, dim=1)
        secondary_preds = torch.argmax(secondary_logits, dim=1)
        
        correct_primary += torch.sum(primary_preds == primary_labels)
        correct_secondary += torch.sum(secondary_preds == secondary_labels)
        
        losses.append(loss.item())
        
        # 更新进度条
        loop.set_description(f"训练中")
        loop.set_postfix(
            loss=loss.item(),
            primary_acc=correct_primary.double() / n_examples,
            secondary_acc=correct_secondary.double() / n_examples
        )
    
    return (
        correct_primary.double() / n_examples,
        correct_secondary.double() / n_examples,
        np.mean(losses)
    )

# 6. 评估函数
def eval_model(model, data_loader, loss_fn_primary, loss_fn_secondary, device, n_examples):
    """评估模型"""
    model = model.eval()
    
    losses = []
    correct_primary = 0
    correct_secondary = 0
    
    all_primary_preds = []
    all_secondary_preds = []
    all_primary_labels = []
    all_secondary_labels = []
    
    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            primary_labels = batch["primary_label"].to(device)
            secondary_labels = batch["secondary_label"].to(device)
            
            primary_logits, secondary_logits = model(
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
            
            # 更新进度条
            loop.set_description(f"评估中")
            loop.set_postfix(
                loss=loss.item(),
                primary_acc=correct_primary.double() / n_examples,
                secondary_acc=correct_secondary.double() / n_examples
            )
    
    return (
        correct_primary.double() / n_examples,
        correct_secondary.double() / n_examples,
        np.mean(losses),
        all_primary_preds, all_primary_labels,
        all_secondary_preds, all_secondary_labels
    )

# 7. 早停机制
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.early_stop = False
        self.count = 0
        self.best_model_weights = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = model.state_dict()
            self.count = 0

# 8. 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, title, chinese_font, figsize=(16, 14), 
                          normalize=False, show_values=True, max_labels=15):
    """
    绘制优化的混淆矩阵，增强可读性
    修复了数字显示、标签堆叠和数量限制问题
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 如果标签太多，采样显示最常见的标签
    if len(labels) > max_labels:
        # 计算每个标签的总出现次数
        label_counts = np.sum(cm, axis=1) + np.sum(cm, axis=0)
        # 获取最常见的标签索引
        top_indices = np.argsort(label_counts)[-max_labels:]
        # 筛选混淆矩阵
        cm = cm[np.ix_(top_indices, top_indices)]
        # 更新标签列表
        labels = [labels[i] for i in top_indices]
        title += f"（仅显示出现次数最多的{max_labels}个标签）"
    
    # 归一化处理（可选）
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cbar_label = '比例'
    else:
        fmt = 'd'
        cbar_label = '数量'
    
    # 创建图表
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # 绘制热图，使用更合适的颜色映射
    im = sns.heatmap(cm, annot=show_values, fmt=fmt, cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     ax=ax, cbar_kws={'label': cbar_label},
                     annot_kws={"size": 10, "color": "black"})
    
    # 设置颜色条字体
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, fontproperties=chinese_font, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 设置标题和标签字体
    plt.title(title, fontproperties=chinese_font, fontsize=16, pad=20)
    plt.xlabel('预测标签', fontproperties=chinese_font, fontsize=14, labelpad=10)
    plt.ylabel('真实标签', fontproperties=chinese_font, fontsize=14, labelpad=10)
    
    # 设置坐标轴刻度标签，优化旋转角度和对齐方式
    plt.xticks(fontproperties=chinese_font, fontsize=8, rotation=45, ha='right')
    plt.yticks(fontproperties=chinese_font, fontsize=8, rotation=0)
    
    # 调整布局，增加底部边距防止标签被截断
    plt.subplots_adjust(bottom=0.25, right=0.95)
    
    # 保存图片
    filename = f'{title.replace(" ", "_").replace("（", "").replace("）", "")}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


# 9. 主训练函数
def train_model(data_path, epochs=10, batch_size=32, max_len=128, learning_rate=2e-5, samples_per_category=100000):
    # 设置中文字体（仅使用SimHei）
    print("设置SimHei中文字体...")
    chinese_font = setup_chinese_font()
    # 测试中文字体显示
    test_chinese_font_display(chinese_font)
    
    # 加载数据
    print("加载数据中...")
    df = load_data(data_path)
    
    # 平衡数据集 - 每个主类别保留指定数量的样本
    print(f"平衡数据集 - 每个主类别保留{samples_per_category}条样本...")
    df = balance_dataset(df, samples_per_category=samples_per_category)
    
    # 数据分析
    analyze_data(df)
    
    # 分割训练集和测试集
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
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
    
    test_df['primary_encoded'] = primary_le.transform(test_df['primary_label'])
    test_df['secondary_encoded'] = secondary_le.transform(test_df['secondary_label'])
    
    # 保存类别信息
    num_primary_labels = len(primary_le.classes_)
    num_secondary_labels = len(secondary_le.classes_)
    
    print(f"一级标签数量: {num_primary_labels}")
    print(f"二级标签数量: {num_secondary_labels}")
    
    # 加载BERT tokenizer
    print("加载BERT模型和Tokenizer...")
    bert_model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(
        train_df, test_df, tokenizer, batch_size, max_len
    )
    
    # 初始化模型
    model = ProductClassifier(
        bert_model_name,
        num_primary_labels,
        num_secondary_labels
    )
    model = model.to(device)
    
    # 定义损失函数和优化器
    loss_fn_primary = nn.CrossEntropyLoss().to(device)
    loss_fn_secondary = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 训练历史记录
    history = {
        'train_primary_acc': [],
        'train_secondary_acc': [],
        'train_loss': [],
        'val_primary_acc': [],
        'val_secondary_acc': [],
        'val_loss': []
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
            scaler
        )
        
        print(f"训练集: 一级准确率 {train_primary_acc:.4f}, 二级准确率 {train_secondary_acc:.4f}, 损失 {train_loss:.4f}")
        
        # 评估
        val_primary_acc, val_secondary_acc, val_loss, _, _, _, _ = eval_model(
            model,
            test_loader,
            loss_fn_primary,
            loss_fn_secondary,
            device,
            len(test_df)
        )
        
        print(f"验证集: 一级准确率 {val_primary_acc:.4f}, 二级准确率 {val_secondary_acc:.4f}, 损失 {val_loss:.4f}")
        
        # 记录历史
        history['train_primary_acc'].append(train_primary_acc.item())
        history['train_secondary_acc'].append(train_secondary_acc.item())
        history['train_loss'].append(train_loss)
        history['val_primary_acc'].append(val_primary_acc.item())
        history['val_secondary_acc'].append(val_secondary_acc.item())
        history['val_loss'].append(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("触发早停机制")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(early_stopping.best_model_weights)
    
    # 最终评估并获取预测结果
    print("最终评估...")
    final_primary_acc, final_secondary_acc, final_loss, \
    primary_preds, primary_labels, secondary_preds, secondary_labels = eval_model(
        model,
        test_loader,
        loss_fn_primary,
        loss_fn_secondary,
        device,
        len(test_df)
    )
    
    print(f"最终测试集: 一级准确率 {final_primary_acc:.4f}, 二级准确率 {final_secondary_acc:.4f}")
    print(f"平均准确率: {(final_primary_acc + final_secondary_acc) / 2:.4f}")
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    # 为了可视化清晰，只显示前15个最常见的标签
    top_n = 15
    
    # 一级标签混淆矩阵
    plot_confusion_matrix(
        primary_labels,
        primary_preds,
        primary_le.classes_,
        f'一级标签混淆矩阵（Top {top_n}）',
        chinese_font,
        max_labels=top_n,  # 确保只显示top_n个标签
        show_values=True   # 显示数值
    )
    
    # 二级标签混淆矩阵
    plot_confusion_matrix(
        secondary_labels,
        secondary_preds,
        secondary_le.classes_,
        f'二级标签混淆矩阵（Top {top_n}）',
        chinese_font,
        max_labels=top_n,  # 确保只显示top_n个标签
        show_values=True,  # 显示数值
        figsize=(18, 16)   # 更大的图尺寸，适合更多标签
    )
    
    # 绘制训练历史
    print("绘制训练历史...")
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_primary_acc'], label='训练集一级准确率')
    plt.plot(history['train_secondary_acc'], label='训练集二级准确率')
    plt.plot(history['val_primary_acc'], label='验证集一级准确率')
    plt.plot(history['val_secondary_acc'], label='验证集二级准确率')
    plt.title('准确率曲线', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('准确率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练集损失')
    plt.plot(history['val_loss'], label='验证集损失')
    plt.title('损失曲线', fontproperties=chinese_font)
    plt.xlabel('Epoch', fontproperties=chinese_font)
    plt.ylabel('损失', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_history.png')
    plt.close()
    
    # 随机抽取测试样本进行预测展示
    print("随机抽取测试样本进行预测展示...")
    # 每个主类别抽取5个样本
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
            
            primary_logits, secondary_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            primary_preds = torch.argmax(primary_logits, dim=1).cpu().numpy()
            secondary_preds = torch.argmax(secondary_logits, dim=1).cpu().numpy()
    
    # 解码预测结果
    sample_df['primary_pred'] = primary_le.inverse_transform(primary_preds)
    sample_df['secondary_pred'] = secondary_le.inverse_transform(secondary_preds)
    
    # 打印结果（使用ANSI颜色码）
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
        'tokenizer': tokenizer
    }, '/kaggle/working/product_classifier_model.pt')
    
    print("训练完成!")
    
    return model, tokenizer, primary_le, secondary_le, history, chinese_font

# 主函数
if __name__ == "__main__":
    # 请修改为你的数据根目录路径
    data_directory = "/kaggle/input/product-categories"
    
    # 训练模型
    model, tokenizer, primary_le, secondary_le, history, chinese_font = train_model(
        data_path=data_directory,
        epochs=10,
        batch_size=64,
        max_len=128,
        learning_rate=2e-5,
        samples_per_category=50000
    )