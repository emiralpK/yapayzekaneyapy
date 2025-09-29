import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import re
import os
import json
import threading
import time
from datetime import datetime
import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Gelişmiş Tokenizer Sınıfı
class AdvancedTokenizer:
    def __init__(self, max_vocab_size=50000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, "<MASK>": 4}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>", 4: "<MASK>"}
        self.word_freq = Counter()
        self.max_vocab_size = max_vocab_size
        
    def fit(self, texts, min_freq=2):
        # Kelime frekanslarını hesapla
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)
        
        # Kelime sözlüğünü oluştur (en sık kullanılan kelimelerle sınırla)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: -x[1])
        idx = len(self.word2idx)
        
        for word, freq in sorted_words:
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
                # Kelime sınırına ulaştığımızda dur
                if len(self.word2idx) >= self.max_vocab_size:
                    break
    
    def tokenize(self, text):
        # Gelişmiş metin tokenizasyon
        text = text.lower()
        # Noktalama işaretlerini ayır
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # Gereksiz boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    def encode(self, text, max_len=128):
        tokens = self.tokenize(text)
        # <START> token'ı ekle
        encoded = [self.word2idx.get("<START>")]
        # Kelimeleri token ID'lerine dönüştür
        encoded += [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        # <END> token'ı ekle
        encoded.append(self.word2idx.get("<END>"))
        
        # Uzunluğu ayarla
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len-1] + [self.word2idx.get("<END>")]
        
        return encoded
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx["<END>"]:
                break
            if idx == self.word2idx["<PAD>"] or idx == self.word2idx["<START>"]:
                continue
                
            word = self.idx2word.get(idx, "<UNK>")
            words.append(word)
            
        # Noktalama işaretlerini düzelt
        text = " ".join(words)
        text = re.sub(r' ([.,!?;:]) ', r'\1 ', text)
        return text

# Gelişmiş veri seti
class LanguageModelDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=64, stride=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        
        # Daha verimli veri hazırlama
        for text in texts:
            encoded = tokenizer.encode(text, seq_len + 1)
            
            # Kayan pencere yaklaşımıyla daha fazla örnek oluştur
            for i in range(0, max(1, len(encoded) - seq_len), stride):
                x = encoded[i:i+seq_len]
                y = encoded[i+1:i+seq_len+1]
                
                if len(x) == seq_len and len(y) == seq_len:
                    self.data.append((torch.tensor(x), torch.tensor(y)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Gelişmiş Dil Modeli için Attention Modülü
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        
    def forward(self, x):
        # Attention hesapla
        q = self.q(x)  # query
        k = self.k(x)  # key
        v = self.v(x)  # value
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        
        # Maskeleme (gelecek token'ları gösterme)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax ile attention ağırlıklarını hesapla
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum
        output = torch.matmul(attn_weights, v)
        return output

# Quantized Transformer katmanı
class QuantizedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=4, dropout=0.1, bits=8):
        super().__init__()
        self.bits = bits
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        
        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.attention_projection = nn.Linear(num_heads * head_dim, embed_dim)
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Quantization için gereken parametreler
        if bits < 32:
            if bits == 2:
                self.qmin, self.qmax = -2, 1
            elif bits == 4:
                self.qmin, self.qmax = -8, 7
            elif bits == 8:
                self.qmin, self.qmax = -128, 127
                
            self.register_buffer('scale_attn', torch.ones(1))
            self.register_buffer('zero_point_attn', torch.zeros(1))
            self.register_buffer('scale_ff', torch.ones(1))
            self.register_buffer('zero_point_ff', torch.zeros(1))
            
    def quantize(self, x, scale, zero_point):
        if self.bits >= 32:
            return x
        x_q = torch.clamp(torch.round(x / scale + zero_point), self.qmin, self.qmax)
        return (x_q - zero_point) * scale
        
    def forward(self, x):
        # Layer Norm ve Multi-head Attention
        residual = x
        x = self.layer_norm1(x)
        
        # Her attention head'i hesapla ve birleştir
        attn_outputs = [head(x) for head in self.attention_heads]
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = self.attention_projection(attn_output)
        
        # Quantize (eğitim sırasında)
        if self.training and self.bits < 32:
            # Dinamik skala hesapla
            with torch.no_grad():
                self.scale_attn.copy_(torch.max(torch.abs(attn_output)) / (self.qmax - self.qmin) * 2)
                
            attn_output = self.quantize(attn_output, self.scale_attn, self.zero_point_attn)
        
        # Skip connection
        x = residual + self.dropout(attn_output)
        
        # Feed Forward ağ
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        
        # Quantize (eğitim sırasında)
        if self.training and self.bits < 32:
            # Dinamik skala hesapla
            with torch.no_grad():
                self.scale_ff.copy_(torch.max(torch.abs(x)) / (self.qmax - self.qmin) * 2)
                
            x = self.quantize(x, self.scale_ff, self.zero_point_ff)
        
        # Skip connection
        x = residual + self.dropout(x)
        
        return x

# Geliştirilmiş Dil Modeli
class AdvancedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=4, 
                 num_heads=4, dropout=0.1, max_seq_len=512, bits=8):
        super().__init__()
        self.bits = bits
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Token ve pozisyon gömme
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer katmanları
        self.transformer_blocks = nn.ModuleList([
            QuantizedTransformerBlock(embed_dim, hidden_dim, num_heads, dropout, bits)
            for _ in range(num_layers)
        ])
        
        # Son layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Modeli başlat
        self._init_weights()
        
    def _init_weights(self):
        # Xavier başlatma
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Token gömme
        x = self.token_embedding(x)
        
        # Pozisyon gömme ekle
        x = x + self.position_embedding[:, :seq_len, :]
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer blokları
        for block in self.transformer_blocks:
            x = block(x)
        
        # Son layer norm
        x = self.layer_norm(x)
        
        # Token tahminleri
        logits = self.output(x)
        
        return logits
        
    def generate(self, prompt_ids, tokenizer, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval()
        with torch.no_grad():
            # Prompt'u tensor'a çevir
            if not isinstance(prompt_ids, torch.Tensor):
                prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
                
            # Doğru boyuta ve cihaza taşı
            if len(prompt_ids.shape) == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            
            prompt_ids = prompt_ids.to(next(self.parameters()).device)
            input_ids = prompt_ids.clone()
            
            # Token'ları üret
            for _ in range(max_new_tokens):
                # Son 512 token ile çalış (bellek kısıtlaması)
                truncated_ids = input_ids[:, -512:]
                
                # Forward pass
                outputs = self(truncated_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Probs hesapla
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                
                # Token seç
                idx_next = torch.multinomial(top_k_probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, idx_next)
                
                # Kontrol - eğer <END> token'ı geldiyse dur
                if next_token.item() == tokenizer.word2idx["<END>"]:
                    break
                
                # Yeni token'ı ekle
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
            # Prompt'tan sonraki kısmı döndür
            generated_ids = input_ids[0, prompt_ids.size(1):].tolist()
            
            return generated_ids

# AI Trainer Uygulaması
class AdvancedAITrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Gelişmiş Yapay Zeka Eğitimi - RTX Edition")
        self.root.geometry("1400x800")
        self.root.configure(bg="#161B22")
        
        # Değişkenler
        self.model = None
        self.tokenizer = AdvancedTokenizer()
        self.is_training = False
        self.loaded_texts = []
        self.training_history = {'loss': [], 'perplexity': []}
        self.device = self.get_optimal_device()
        self.scaler = GradScaler()
        
        # Stil ayarları
        self.setup_styles()
        
        # Arayüz oluştur
        self.create_widgets()
        
        # GPU bilgisi göster
        self.display_gpu_info()
        
    def get_optimal_device(self):
        """En uygun cihazı belirle"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            # CUDA optimizasyonları
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return device
        else:
            return torch.device("cpu")
        
    def display_gpu_info(self):
        """GPU bilgisini göster"""
        if self.device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                self.log(f"🎮 GPU Algılandı: {gpu_name} ({total_mem:.2f} GB)")
                self.log(f"🚀 CUDA Versiyonu: {torch.version.cuda}")
                self.log(f"📊 cuDNN Versiyonu: {torch.backends.cudnn.version()}")
            except Exception as e:
                self.log(f"⚠️ GPU bilgisi alınırken hata: {str(e)}")
        else:
            self.log("⚠️ GPU bulunamadı! CPU modunda çalışılıyor.")
        
    def setup_styles(self):
        """Arayüz stillerini ayarla"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Modern renk teması
        self.colors = {
            'bg': '#161B22',
            'panel_bg': '#0D1117',
            'text_bg': '#0D1117',
            'fg': '#C9D1D9',
            'accent': '#58A6FF',
            'secondary': '#30363D',
            'success': '#3FB950',
            'danger': '#F85149',
            'warning': '#F7B955'
        }
        
        # TTK Stiller
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        style.configure('TButton', background=self.colors['secondary'], foreground=self.colors['fg'])
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.colors['accent'])
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['fg'])
        
    def create_widgets(self):
        """Arayüz bileşenlerini oluştur"""
        # Ana container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Başlık
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_text = "🧠 Gelişmiş Yapay Zeka Eğitim Platformu" 
        subtitle_text = "RTX Optimizer Edition"
        
        title_label = ttk.Label(title_frame, text=title_text, style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text=subtitle_text, 
                                 font=('Segoe UI', 14, 'italic'),
                                 foreground=self.colors['accent'])
        subtitle_label.pack()
        
        # Sekmeleri oluştur
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_training_tab()    # Eğitim sekmesi
        self.create_chat_tab()        # Sohbet sekmesi
        self.create_stats_tab()       # İstatistik sekmesi
        self.create_settings_tab()    # Ayarlar sekmesi
        
    def create_training_tab(self):
        """Eğitim sekmesini oluştur"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="🎓 Eğitim")
        
        # Sol panel - Veri ve ayarlar
        left_panel = ttk.Frame(train_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # Dosya yükleme bölümü
        file_frame = ttk.LabelFrame(left_panel, text="📁 Veri Yükleme")
        file_frame.pack(fill=tk.X, pady=10, padx=5, ipady=5)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.load_btn = tk.Button(btn_frame, text="📂 Metin Dosyası Seç", 
                                 command=self.load_file,
                                 bg=self.colors['accent'], fg='white',
                                 font=('Segoe UI', 10, 'bold'),
                                 relief=tk.RAISED, padx=5, pady=5)
        self.load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.clear_btn = tk.Button(btn_frame, text="🗑️ Temizle", 
                                  command=self.clear_data,
                                  bg=self.colors['secondary'], fg='white',
                                  font=('Segoe UI', 10),
                                  relief=tk.RAISED, padx=5, pady=5)
        self.clear_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Dosya bilgisi
        self.file_info_var = tk.StringVar(value="Henüz veri yüklenmedi")
        file_info_label = ttk.Label(file_frame, textvariable=self.file_info_var, 
                                   foreground=self.colors['warning'],
                                   font=('Segoe UI', 9))
        file_info_label.pack(padx=10, pady=(0, 10), anchor=tk.W)
        
        # Eğitim ayarları
        model_frame = ttk.LabelFrame(left_panel, text="⚙️ Model Ayarları")
        model_frame.pack(fill=tk.X, pady=10, padx=5, ipady=5)
        
        # Model parametreleri
        params_frame = ttk.Frame(model_frame)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Bit derinliği
        ttk.Label(params_frame, text="Model Hassasiyeti:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.bit_var = tk.StringVar(value="8")
        bit_combo = ttk.Combobox(params_frame, textvariable=self.bit_var, 
                               values=["2", "4", "8", "32"], width=15)
        bit_combo.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(params_frame, text="bit").grid(row=0, column=2, sticky=tk.W)
        
        # Katman sayısı
        ttk.Label(params_frame, text="Transformer Katmanları:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.layers_var = tk.IntVar(value=4)
        layers_combo = ttk.Combobox(params_frame, textvariable=self.layers_var, 
                                   values=[2, 4, 6, 8], width=15)
        layers_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Epoch sayısı
        ttk.Label(params_frame, text="Epoch Sayısı:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.epoch_var = tk.IntVar(value=10)
        epoch_spin = tk.Spinbox(params_frame, from_=1, to=100, 
                               textvariable=self.epoch_var,
                               width=13, bg=self.colors['text_bg'], 
                               fg=self.colors['fg'])
        epoch_spin.grid(row=2, column=1, padx=10, pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.batch_var = tk.IntVar(value=32)
        batch_combo = ttk.Combobox(params_frame, textvariable=self.batch_var, 
                                  values=[8, 16, 32, 64, 128], width=15)
        batch_combo.grid(row=3, column=1, padx=10, pady=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_combo = ttk.Combobox(params_frame, textvariable=self.lr_var, 
                              values=[0.01, 0.005, 0.001, 0.0005, 0.0001], width=15)
        lr_combo.grid(row=4, column=1, padx=10, pady=5)
        
        # Gelişmiş ayarlar
        advanced_frame = ttk.Frame(model_frame)
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Optimizasyon ayarları
        self.amp_var = tk.BooleanVar(value=True)
        amp_check = ttk.Checkbutton(advanced_frame, text="Mixed Precision (FP16) Kullan", 
                                   variable=self.amp_var)
        amp_check.pack(anchor=tk.W)
        
        # Eğitim butonu
        self.train_btn = tk.Button(left_panel, text="🚀 Eğitimi Başlat", 
                                 command=self.start_training,
                                 bg=self.colors['success'], fg='white',
                                 font=('Segoe UI', 12, 'bold'),
                                 relief=tk.RAISED,
                                 cursor='hand2',
                                 height=2,
                                 activebackground=self.colors['success'])
        self.train_btn.pack(fill=tk.X, pady=15, padx=5)
        
        # Sağ panel - Log ve ilerleme
        right_panel = ttk.Frame(train_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # İlerleme çubuğu
        progress_frame = ttk.LabelFrame(right_panel, text="📊 İlerleme")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var,
                                          mode='determinate',
                                          length=400)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_text = tk.StringVar(value="Hazır")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_text)
        progress_label.pack(padx=10, pady=5)
        
        # Grafik alanı
        plot_frame = ttk.LabelFrame(right_panel, text="📈 Eğitim Metrikleri")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figürü
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['panel_bg'])
        self.ax.tick_params(colors=self.colors['fg'])
        self.ax.spines['bottom'].set_color(self.colors['fg'])
        self.ax.spines['top'].set_color(self.colors['fg'])
        self.ax.spines['left'].set_color(self.colors['fg'])
        self.ax.spines['right'].set_color(self.colors['fg'])
        self.ax.set_xlabel('Batch', color=self.colors['fg'])
        self.ax.set_ylabel('Loss', color=self.colors['fg'])
        self.ax.set_title('Eğitim Kaybı', color=self.colors['accent'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log alanı
        log_frame = ttk.LabelFrame(right_panel, text="📝 Eğitim Günlüğü")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                               wrap=tk.WORD,
                                               font=('Consolas', 9),
                                               background=self.colors['text_bg'],
                                               foreground=self.colors['fg'],
                                               height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Başlangıç log mesajı
        self.log("🧠 Gelişmiş Yapay Zeka Eğitim Platformu - Başlatıldı")
        self.log(f"💻 Cihaz: {self.device}")
    
    def create_chat_tab(self):
        """Sohbet sekmesini oluştur"""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="💬 Sohbet")
        
        # Sohbet alanı
        chat_panel = ttk.Frame(chat_frame)
        chat_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Sohbet ayarları
        settings_frame = ttk.LabelFrame(chat_panel, text="⚙️ Sohbet Ayarları")
        settings_frame.pack(fill=tk.X, pady=10)
        
        settings_grid = ttk.Frame(settings_frame)
        settings_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Sıcaklık ayarı
        ttk.Label(settings_grid, text="Sıcaklık:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.temperature_var = tk.DoubleVar(value=0.8)
        temp_slider = ttk.Scale(settings_grid, from_=0.1, to=2.0, 
                             variable=self.temperature_var,
                             length=150,
                             orient=tk.HORIZONTAL)
        temp_slider.grid(row=0, column=1, padx=5)
        temp_value = ttk.Label(settings_grid, text="0.8")
        temp_value.grid(row=0, column=2, padx=5)
        
        # Slider değiştiğinde etiketi güncelle
        def update_temp(event):
            temp_value.config(text=f"{self.temperature_var.get():.1f}")
        temp_slider.bind("<Motion>", update_temp)
        
        # Top-k ayarı
        ttk.Label(settings_grid, text="Top-k:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        self.top_k_var = tk.IntVar(value=40)
        topk_combo = ttk.Combobox(settings_grid, textvariable=self.top_k_var, 
                                values=[10, 20, 40, 50, 100], width=5)
        topk_combo.grid(row=0, column=4, padx=5)
        
        # Max uzunluk ayarı
        ttk.Label(settings_grid, text="Maks Uzunluk:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=10)
        self.max_len_var = tk.IntVar(value=100)
        maxlen_spin = tk.Spinbox(settings_grid, from_=10, to=500, 
                               textvariable=self.max_len_var,
                               width=5, bg=self.colors['text_bg'], fg=self.colors['fg'])
        maxlen_spin.grid(row=1, column=1, padx=5, pady=10)
        
        # Sohbet geçmişi
        chat_display_frame = ttk.LabelFrame(chat_panel, text="💬 Sohbet")
        chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(chat_display_frame,
                                                   wrap=tk.WORD,
                                                   font=('Segoe UI', 11),
                                                   background=self.colors['text_bg'],
                                                   foreground=self.colors['fg'],
                                                   padx=10, pady=10,
                                                   height=15)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mesaj giriş alanı
        message_frame = ttk.Frame(chat_panel)
        message_frame.pack(fill=tk.X, pady=10)
        
        self.message_var = tk.StringVar()
        self.message_entry = ttk.Entry(message_frame, textvariable=self.message_var, font=('Segoe UI', 11))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.message_entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_btn = tk.Button(message_frame, text="📤 Gönder", 
                                command=self.send_message,
                                bg=self.colors['accent'], fg='white',
                                font=('Segoe UI', 10, 'bold'),
                                padx=10, pady=5,
                                relief=tk.RAISED)
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Sohbet geçmişi için metin etiketleri
        self.chat_display.tag_configure('user', foreground='#58A6FF', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('ai', foreground='#79C0FF')
        self.chat_display.tag_configure('system', foreground='#8B949E', font=('Segoe UI', 9, 'italic'))
        
        # Başlangıç mesajı
        self.chat_display.insert(tk.END, "🤖 Sistem: Model eğitildiğinde buradan sohbet edebilirsiniz.\n\n", 'system')
    
    def create_stats_tab(self):
        """İstatistik sekmesini oluştur"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📊 İstatistikler")
        
        # Model bilgileri
        info_panel = ttk.Frame(stats_frame)
        info_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Model istatistikleri
        model_stats_frame = ttk.LabelFrame(info_panel, text="📋 Model Bilgileri")
        model_stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(model_stats_frame,
                                                 wrap=tk.WORD,
                                                 font=('Consolas', 11),
                                                 background=self.colors['text_bg'],
                                                 foreground=self.colors['accent'],
                                                 height=10,
                                                 padx=10, pady=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Kelime hazinesi analizi
        vocab_frame = ttk.LabelFrame(info_panel, text="🔤 Kelime Hazinesi")
        vocab_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.vocab_text = scrolledtext.ScrolledText(vocab_frame,
                                                 wrap=tk.WORD,
                                                 font=('Consolas', 10),
                                                 background=self.colors['text_bg'],
                                                 foreground=self.colors['fg'],
                                                 height=10,
                                                 padx=10, pady=10)
        self.vocab_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model kaydetme/yükleme
        button_frame = ttk.Frame(info_panel)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.save_btn = tk.Button(button_frame, text="💾 Modeli Kaydet", 
                                command=self.save_model,
                                bg=self.colors['accent'], fg='white',
                                font=('Segoe UI', 11),
                                padx=10, pady=5)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = tk.Button(button_frame, text="📂 Model Yükle", 
                                command=self.load_model,
                                bg=self.colors['secondary'], fg='white',
                                font=('Segoe UI', 11),
                                padx=10, pady=5)
        self.load_btn.pack(side=tk.LEFT, padx=5)
    
    def create_settings_tab(self):
        """Ayarlar sekmesini oluştur"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Ayarlar")
        
        # Ayarlar paneli
        settings_panel = ttk.Frame(settings_frame)
        settings_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Sistem ayarları
        system_frame = ttk.LabelFrame(settings_panel, text="🖥️ Sistem Ayarları")
        system_frame.pack(fill=tk.X, pady=10)
        
        system_grid = ttk.Frame(system_frame)
        system_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # GPU/CPU seçimi
        ttk.Label(system_grid, text="İşlem Birimi:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=10)
        
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        device_frame = ttk.Frame(system_grid)
        device_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=10)
        
        ttk.Radiobutton(device_frame, text="GPU (CUDA)", 
                      variable=self.device_var, 
                      value="cuda",
                      state=tk.NORMAL if torch.cuda.is_available() else tk.DISABLED).pack(anchor=tk.W)
        ttk.Radiobutton(device_frame, text="CPU", 
                      variable=self.device_var, 
                      value="cpu").pack(anchor=tk.W)
        
        # Hakkında
        about_frame = ttk.LabelFrame(settings_panel, text="🧪 Hakkında")
        about_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        about_text = """
        🧠 Gelişmiş Yapay Zeka Eğitim Platformu
        
        Bu uygulama, kendi özel yapay zeka dil modelinizi oluşturmanıza ve eğitmenize olanak sağlar.
        
        ✨ Özellikler:
        • Transformer mimarisi ile gelişmiş dil modelleme
        • GPU optimizasyonu (NVIDIA RTX için özel)
        • Farklı model büyüklükleri ve hassasiyetleri (2-bit, 4-bit, 8-bit, 32-bit)
        • Metin üretimi için gelişmiş sampling
        • Eğitim istatistikleri ve görselleştirme
        
        💡 Kullanım:
        1. "Eğitim" sekmesinden metin dosyası yükleyin
        2. Model parametrelerini ayarlayın
        3. Eğitimi başlatın
        4. "Sohbet" sekmesinden modelinizle konuşun
        
        🚀 GPU desteği ile büyük modeller çok daha hızlı eğitilebilir.
        """
        
        about_label = ttk.Label(about_frame, text=about_text, 
                             justify=tk.LEFT, 
                             font=('Segoe UI', 10),
                             wraplength=700)
        about_label.pack(padx=20, pady=20)
    
    def load_file(self):
        """Metin dosyası yükle"""
        file_path = filedialog.askopenfilename(
            title="Eğitim Metni Seç",
            filetypes=[("Metin Dosyaları", "*.txt"), ("Tüm Dosyalar", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Cümlelere böl
            text_chunks = []
            
            # Noktalama işaretleriyle parçalara böl
            segments = re.split(r'(?<=[.!?])\s+', content)
            
            for segment in segments:
                segment = segment.strip()
                if len(segment) > 10:  # Çok kısa cümleleri atla
                    text_chunks.append(segment)
            
            # Eğer çok uzun cümleler varsa bunları da makul uzunluğa böl
            processed_chunks = []
            for chunk in text_chunks:
                if len(chunk) > 1000:  # Çok uzun cümleler
                    sub_chunks = [chunk[i:i+500] for i in range(0, len(chunk), 500)]
                    processed_chunks.extend(sub_chunks)
                else:
                    processed_chunks.append(chunk)
            
            self.loaded_texts = processed_chunks
            
            # Dosya bilgisini göster
            file_size = os.path.getsize(file_path) / 1024  # KB
            file_name = os.path.basename(file_path)
            
            self.file_info_var.set(f"✓ {file_name} ({file_size:.1f} KB)\n"
                                 f"Toplam {len(self.loaded_texts)} metin parçası, "
                                 f"{sum(len(t) for t in self.loaded_texts)} karakter")
            
            self.log(f"📄 Dosya yüklendi: {file_name}")
            self.log(f"📊 {len(self.loaded_texts)} metin parçası bulundu")
            
            # Örnek metin göster
            if self.loaded_texts:
                self.log(f"📝 Örnek metin: {self.loaded_texts[0][:100]}...")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yüklenirken bir hata oluştu: {str(e)}")
            self.log(f"❌ Dosya yükleme hatası: {str(e)}")
    
    def clear_data(self):
        """Yüklenen verileri temizle"""
        if self.loaded_texts:
            self.loaded_texts = []
            self.file_info_var.set("Henüz veri yüklenmedi")
            self.log("🧹 Yüklenen veriler temizlendi")
    
    def log(self, message):
        """Log ekranına mesaj ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Otomatik kaydırma
        
        # UI'ın donmaması için güncelle
        self.root.update_idletasks()
    
    def start_training(self):
        """Eğitimi başlat"""
        if not self.loaded_texts:
            messagebox.showwarning("Uyarı", "Lütfen önce metin dosyası yükleyin!")
            return
        
        if self.is_training:
            messagebox.showinfo("Bilgi", "Eğitim zaten devam ediyor!")
            return
        
        # Eğitimi ayrı bir thread'de başlat
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def train_model(self):
        """Model eğitim işlemi"""
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.progress_text.set("Başlatılıyor...")
        
        try:
            # Eğitim başlangıç zamanı
            start_time = time.time()
            self.log("=" * 50)
            self.log("🚀 Eğitim başlatılıyor...")
            
            # Eğitim parametreleri
            bits = int(self.bit_var.get())
            num_layers = int(self.layers_var.get())
            epochs = int(self.epoch_var.get())
            batch_size = int(self.batch_var.get())
            learning_rate = float(self.lr_var.get())
            use_amp = self.amp_var.get() and self.device.type == "cuda"
            
            self.log(f"⚙️ Parametreler: {bits}-bit, {num_layers} katman, {epochs} epoch, batch={batch_size}, lr={learning_rate}")
            
            # Tokenizer'ı hazırla
            self.log("🔤 Tokenizer eğitiliyor...")
            self.tokenizer.fit(self.loaded_texts, min_freq=2)
            vocab_size = len(self.tokenizer.word2idx)
            self.log(f"📚 Kelime hazinesi: {vocab_size} kelime")
            
            # Dataset ve dataloader oluştur
            self.log("📦 Veri setleri hazırlanıyor...")
            dataset = LanguageModelDataset(self.loaded_texts, self.tokenizer, seq_len=64)
            
            if len(dataset) == 0:
                raise ValueError("Veri seti boş! Daha uzun metinler gerekiyor.")
                
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                   pin_memory=True if self.device.type == "cuda" else False)
            
            # Model oluştur
            self.log("🧠 Model oluşturuluyor...")
            self.model = AdvancedLanguageModel(
                vocab_size=vocab_size, 
                embed_dim=256 if bits >= 8 else 128,
                hidden_dim=512 if bits >= 8 else 256, 
                num_layers=num_layers,
                bits=bits
            ).to(self.device)
            
            # Optimizer ve loss
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/10)
            criterion = nn.CrossEntropyLoss()
            scaler = GradScaler(enabled=use_amp)
            
            # Eğitim istatistikleri
            self.training_history = {'loss': [], 'perplexity': []}
            
            # Toplam adım sayısını hesapla
            total_steps = epochs * len(dataloader)
            current_step = 0
            best_loss = float('inf')
            
            # Eğitim döngüsü
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_start = time.time()
                
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    # Batch'i cihaza taşı
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Optimizer temizle
                    optimizer.zero_grad()
                    
                    # Forward pass (Mixed precision kullanarak)
                    with autocast(enabled=use_amp):
                        outputs = self.model(inputs)
                        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
                        outputs_flat = outputs.view(-1, outputs.size(-1))
                        targets_flat = targets.view(-1)
                        loss = criterion(outputs_flat, targets_flat)
                    
                    # Backward pass
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # İstatistikleri topla
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    perplexity = torch.exp(torch.tensor(loss_value)).item()
                    
                    self.training_history['loss'].append(loss_value)
                    self.training_history['perplexity'].append(perplexity)
                    
                    # İlerleme güncellemesi
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    self.progress_var.set(progress)
                    
                    batch_info = (f"Epoch {epoch+1}/{epochs} - "
                                 f"Batch {batch_idx+1}/{len(dataloader)} - "
                                 f"Loss: {loss_value:.4f} - "
                                 f"PPL: {perplexity:.2f}")
                    self.progress_text.set(batch_info)
                    
                    # Her 10 adımda bir grafiği güncelle
                    if batch_idx % 10 == 0:
                        self.update_training_plot()
                
                # Epoch sonunda özet
                avg_loss = epoch_loss / len(dataloader)
                avg_ppl = np.exp(avg_loss)
                epoch_time = time.time() - epoch_start
                
                # LR scheduler adımı
                scheduler.step()
                
                self.log(f"🔄 Epoch {epoch+1}/{epochs} tamamlandı: "
                       f"Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}, "
                       f"Süre={epoch_time:.1f}s")
                
                # En iyi modeli kaydet
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.log("💾 En iyi model güncellendi!")
            
            # Eğitim tamamlandı
            training_time = time.time() - start_time
            self.log(f"✅ Eğitim tamamlandı! Toplam süre: {training_time:.1f} saniye")
            self.log(f"📊 Final loss: {best_loss:.4f}, Perplexity: {np.exp(best_loss):.2f}")
            
            # Son grafiği güncelle
            self.update_training_plot()
            
            # Model istatistiklerini güncelle
            self.update_model_stats()
            self.update_vocab_stats()
            
            # Chat sekmesine bildirim ekle
            self.chat_display.insert(tk.END, "🤖 Sistem: Model eğitimi tamamlandı! Şimdi sohbet edebilirsiniz.\n\n", 'system')
            
            # Modelin hazır olduğunu belirt
            messagebox.showinfo("Başarılı", "Model eğitimi tamamlandı! Sohbet sekmesinde modelinizi test edebilirsiniz.")
            
        except Exception as e:
            error_msg = f"Eğitim sırasında hata oluştu: {str(e)}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Hata", error_msg)
            import traceback
            self.log(traceback.format_exc())
        
        finally:
            self.is_training = False
            self.train_btn.config(state=tk.NORMAL)
            self.progress_text.set("Hazır")
    
    def update_training_plot(self):
        """Eğitim grafiğini güncelle"""
        if not self.training_history['loss']:
            return
        
        try:
            # Grafiği temizle
            self.ax.clear()
            
            # Loss grafiği
            self.ax.plot(self.training_history['loss'], label='Loss', color='#58A6FF')
            
            # Grafiği biçimlendir
            self.ax.set_facecolor(self.colors['panel_bg'])
            self.ax.tick_params(colors=self.colors['fg'])
            for spine in self.ax.spines.values():
                spine.set_color(self.colors['fg'])
            
            self.ax.set_xlabel('Adım', color=self.colors['fg'])
            self.ax.set_ylabel('Loss', color=self.colors['fg'])
            self.ax.set_title('Eğitim Kaybı', color=self.colors['accent'])
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.ax.legend(loc='upper right')
            
            # Grafiği çiz
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"⚠️ Grafik güncelleme hatası: {str(e)}")
    
    def update_model_stats(self):
        """Model istatistiklerini güncelle"""
        if self.model is None:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Model henüz oluşturulmadı!")
            return
        
        try:
            # Model parametre sayısını hesapla
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Bit derinliğine göre model boyutunu hesapla
            bits = int(self.bit_var.get())
            model_size_bytes = (total_params * bits) / 8
            
            # Boyut formatını düzenle
            if model_size_bytes < 1024 * 1024:
                size_str = f"{model_size_bytes / 1024:.2f} KB"
            else:
                size_str = f"{model_size_bytes / (1024 * 1024):.2f} MB"
            
            # İstatistik metnini oluştur
            stats_text = f"""
╔═════════════════════════════════════════════════════════════╗
║                    MODEL İSTATİSTİKLERİ                     ║
╠═════════════════════════════════════════════════════════════╣
║ 📊 Toplam Parametre Sayısı: {total_params:,}
║ 🔢 Eğitilebilir Parametre: {trainable_params:,}
║ 💾 Model Boyutu: {size_str}
║ 🧠 Model Mimarisi: Transformer ({self.model.embed_dim} boyutlu)
║ 📚 Katman Sayısı: {len(self.model.transformer_blocks)}
║ 🎯 Hassasiyet: {bits}-bit
║ 📝 Kelime Hazinesi: {len(self.tokenizer.word2idx)} kelime
║ 🧮 Gömme Boyutu: {self.model.embed_dim}
║ 💻 Çalışma Cihazı: {self.device}
║ ⏱️ Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}
╚═════════════════════════════════════════════════════════════╝
            """
            
            # Metni güncelle
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"İstatistik hesaplama hatası: {str(e)}")
    
    def update_vocab_stats(self):
        """Kelime hazinesi istatistiklerini güncelle"""
        if not hasattr(self, 'tokenizer') or not self.tokenizer.word2idx:
            self.vocab_text.delete(1.0, tk.END)
            self.vocab_text.insert(tk.END, "Kelime hazinesi henüz oluşturulmadı!")
            return
        
        try:
            # En sık kullanılan kelimeleri bul (özel tokenler hariç)
            special_tokens = {"<PAD>", "<UNK>", "<START>", "<END>", "<MASK>"}
            word_freqs = {word: freq for word, freq in self.tokenizer.word_freq.items() 
                        if word not in special_tokens}
            
            # En sık kullanılan 20 kelimeyi al
            top_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # İstatistik metnini oluştur
            vocab_text = f"📚 Kelime Hazinesi Boyutu: {len(self.tokenizer.word2idx)}\n\n"
            vocab_text += "🔝 En Sık Kullanılan 20 Kelime:\n"
            vocab_text += "═════════════════════════════════\n"
            
            for i, (word, freq) in enumerate(top_words, 1):
                vocab_text += f"{i:2d}. {word:<15} {freq:5d}\n"
            
            # Rastgele 5 kelime örneği
            random_
