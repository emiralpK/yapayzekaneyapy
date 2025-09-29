import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import json
import os
import threading
import time
from datetime import datetime
import re

class CustomTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.word_freq = Counter()
        
    def fit(self, texts, min_freq=2):
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def encode(self, text, max_len=128):
        tokens = self.tokenize(text)
        encoded = [self.word2idx.get(token, 1) for token in tokens]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == 0:
                continue
            if idx == 3:
                break
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        
        for text in texts:
            encoded = tokenizer.encode(text, seq_len)
            if len(encoded) > 1:
                for i in range(len(encoded) - 1):
                    self.data.append((encoded[i], encoded[i + 1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=4):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.ones(out_features))
        self.zero_point = nn.Parameter(torch.zeros(out_features))
        
        if bits == 2:
            self.qmin, self.qmax = -2, 1
        elif bits == 4:
            self.qmin, self.qmax = -8, 7
        else:
            self.qmin, self.qmax = -128, 127
            
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def quantize(self, x):
        x = torch.clamp(x / self.scale.view(-1, 1) + self.zero_point.view(-1, 1), self.qmin, self.qmax)
        return torch.round(x)
    
    def dequantize(self, x):
        return (x - self.zero_point.view(-1, 1)) * self.scale.view(-1, 1)
    
    def forward(self, x):
        w_quant = self.quantize(self.weight)
        w_dequant = self.dequantize(w_quant)
        return torch.matmul(x, w_dequant.t()) + self.bias

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, bits=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if bits in [2, 4]:
                self.layers.append(nn.ModuleDict({
                    'attn': QuantizedLinear(embed_dim, embed_dim, bits),
                    'ff1': QuantizedLinear(embed_dim, hidden_dim, bits),
                    'ff2': QuantizedLinear(hidden_dim, embed_dim, bits),
                    'norm1': nn.LayerNorm(embed_dim),
                    'norm2': nn.LayerNorm(embed_dim)
                }))
            else:
                self.layers.append(nn.ModuleDict({
                    'attn': nn.Linear(embed_dim, embed_dim),
                    'ff1': nn.Linear(embed_dim, hidden_dim),
                    'ff2': nn.Linear(hidden_dim, embed_dim),
                    'norm1': nn.LayerNorm(embed_dim),
                    'norm2': nn.LayerNorm(embed_dim)
                }))
        
        self.output = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            # Self-attention
            attn_out = layer['attn'](layer['norm1'](x))
            x = x + self.dropout(attn_out)
            
            # Feed-forward
            ff_out = layer['ff2'](torch.relu(layer['ff1'](layer['norm2'](x))))
            x = x + self.dropout(ff_out)
        
        return self.output(x)

class AITrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Kendi AI'nı Eğit - Ultra Modern")
        self.root.geometry("1400x800")
        
        # Değişkenler
        self.model = None
        self.tokenizer = CustomTokenizer()
        self.is_training = False
        self.loaded_texts = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stil ayarları
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Renkler
        self.colors = {
            'bg': '#1a1a2e',
            'fg': '#eee',
            'accent': '#00d4ff',
            'secondary': '#0f3460',
            'success': '#00ff88',
            'danger': '#ff0055',
            'warning': '#ffaa00'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.colors['accent'])
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['fg'])
        style.configure('Modern.TButton', font=('Segoe UI', 10, 'bold'))
        
    def create_widgets(self):
        # Ana container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Başlık
        title_frame = tk.Frame(main_container, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, text="🤖 Kendi AI Modelini Oluştur", 
                               font=('Segoe UI', 28, 'bold'), 
                               fg=self.colors['accent'], bg=self.colors['bg'])
        title_label.pack()
        
        gpu_info = "🎮 GPU: " + ("CUDA Aktif (RTX 4070)" if torch.cuda.is_available() else "CPU Modu")
        gpu_label = tk.Label(title_frame, text=gpu_info, 
                             font=('Segoe UI', 10), 
                             fg=self.colors['success'] if torch.cuda.is_available() else self.colors['warning'], 
                             bg=self.colors['bg'])
        gpu_label.pack()
        
        # Notebook (sekmeler)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Eğitim sekmesi
        self.create_training_tab()
        
        # Sohbet sekmesi
        self.create_chat_tab()
        
        # Model bilgi sekmesi
        self.create_info_tab()
        
    def create_training_tab(self):
        train_frame = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.notebook.add(train_frame, text="🎓 Eğitim")
        
        # Sol panel - Ayarlar
        left_panel = tk.Frame(train_frame, bg=self.colors['secondary'], width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # Dosya yükleme
        file_frame = tk.LabelFrame(left_panel, text="📁 Veri Yükleme", 
                                  font=('Segoe UI', 11, 'bold'),
                                  fg=self.colors['fg'], bg=self.colors['secondary'])
        file_frame.pack(fill=tk.X, pady=10)
        
        self.load_btn = tk.Button(file_frame, text="📂 Dosya Seç (.txt)", 
                                 command=self.load_file,
                                 font=('Segoe UI', 10, 'bold'),
                                 bg=self.colors['accent'], fg='white',
                                 cursor='hand2')
        self.load_btn.pack(pady=10, padx=10)
        
        self.file_info = tk.Label(file_frame, text="Dosya yüklenmedi", 
                                 font=('Segoe UI', 9),
                                 fg=self.colors['warning'], bg=self.colors['secondary'])
        self.file_info.pack(pady=5)
        
        # Model ayarları
        model_frame = tk.LabelFrame(left_panel, text="⚙️ Model Ayarları", 
                                   font=('Segoe UI', 11, 'bold'),
                                   fg=self.colors['fg'], bg=self.colors['secondary'])
        model_frame.pack(fill=tk.X, pady=10)
        
        # Bit seçimi
        tk.Label(model_frame, text="Quantization:", 
                font=('Segoe UI', 10),
                fg=self.colors['fg'], bg=self.colors['secondary']).pack(anchor=tk.W, padx=10, pady=5)
        
        self.bit_var = tk.StringVar(value="4")
        bit_frame = tk.Frame(model_frame, bg=self.colors['secondary'])
        bit_frame.pack(pady=5)
        
        for bits in ["2-bit (En Küçük)", "4-bit (Dengeli)", "8-bit (Kaliteli)"]:
            rb = tk.Radiobutton(bit_frame, text=bits, 
                               variable=self.bit_var, 
                               value=bits.split('-')[0],
                               font=('Segoe UI', 9),
                               fg=self.colors['fg'], bg=self.colors['secondary'],
                               selectcolor=self.colors['secondary'])
            rb.pack(anchor=tk.W)
        
        # Epoch sayısı
        tk.Label(model_frame, text="Epoch Sayısı:", 
                font=('Segoe UI', 10),
                fg=self.colors['fg'], bg=self.colors['secondary']).pack(anchor=tk.W, padx=10, pady=5)
        
        self.epoch_var = tk.IntVar(value=10)
        self.epoch_scale = tk.Scale(model_frame, from_=1, to=50, 
                                   orient=tk.HORIZONTAL,
                                   variable=self.epoch_var,
                                   bg=self.colors['secondary'], fg=self.colors['fg'],
                                   highlightthickness=0)
        self.epoch_scale.pack(fill=tk.X, padx=10)
        
        # Eğitim butonu
        self.train_btn = tk.Button(left_panel, text="🚀 Eğitimi Başlat", 
                                  command=self.start_training,
                                  font=('Segoe UI', 12, 'bold'),
                                  bg=self.colors['success'], fg='white',
                                  height=2, cursor='hand2')
        self.train_btn.pack(fill=tk.X, pady=20, padx=10)
        
        # Sağ panel - Log ve ilerleme
        right_panel = tk.Frame(train_frame, bg=self.colors['secondary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # İlerleme çubuğu
        progress_frame = tk.LabelFrame(right_panel, text="📊 İlerleme", 
                                     font=('Segoe UI', 11, 'bold'),
                                     fg=self.colors['fg'], bg=self.colors['secondary'])
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=400)
        self.progress_bar.pack(pady=10, padx=10, fill=tk.X)
        
        self.progress_label = tk.Label(progress_frame, text="Hazır", 
                                      font=('Segoe UI', 10),
                                      fg=self.colors['fg'], bg=self.colors['secondary'])
        self.progress_label.pack()
        
        # Log alanı
        log_frame = tk.LabelFrame(right_panel, text="📝 Eğitim Logları", 
                                font=('Segoe UI', 11, 'bold'),
                                fg=self.colors['fg'], bg=self.colors['secondary'])
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                 font=('Consolas', 9),
                                                 bg='#0a0a0a', fg=self.colors['success'],
                                                 height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_chat_tab(self):
        chat_frame = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.notebook.add(chat_frame, text="💬 Sohbet")
        
        # Sohbet alanı
        chat_container = tk.Frame(chat_frame, bg=self.colors['secondary'])
        chat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Sohbet geçmişi
        self.chat_display = scrolledtext.ScrolledText(chat_container,
                                                     font=('Segoe UI', 11),
                                                     bg='#0a0a0a', fg=self.colors['fg'],
                                                     wrap=tk.WORD,
                                                     height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Mesaj giriş alanı
        input_frame = tk.Frame(chat_container, bg=self.colors['secondary'])
        input_frame.pack(fill=tk.X, pady=10)
        
        self.message_entry = tk.Entry(input_frame,
                                     font=('Segoe UI', 11),
                                     bg=self.colors['bg'], fg=self.colors['fg'],
                                     insertbackground=self.colors['accent'])
        self.message_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.message_entry.bind('<Return>', lambda e: self.send_message())
        
        self.send_btn = tk.Button(input_frame, text="📤 Gönder",
                                command=self.send_message,
                                font=('Segoe UI', 10, 'bold'),
                                bg=self.colors['accent'], fg='white',
                                cursor='hand2')
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
    def create_info_tab(self):
        info_frame = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.notebook.add(info_frame, text="📊 Model Bilgileri")
        
        # Model istatistikleri
        stats_container = tk.Frame(info_frame, bg=self.colors['secondary'])
        stats_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.stats_text = tk.Text(stats_container,
                                 font=('Consolas', 10),
                                 bg='#0a0a0a', fg=self.colors['accent'],
                                 height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Butonlar
        btn_frame = tk.Frame(stats_container, bg=self.colors['secondary'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="💾 Modeli Kaydet",
                 command=self.save_model,
                 font=('Segoe UI', 10, 'bold'),
                 bg=self.colors['success'], fg='white',
                 cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="📂 Model Yükle",
                 command=self.load_model,
                 font=('Segoe UI', 10, 'bold'),
                 bg=self.colors['accent'], fg='white',
                 cursor='hand2').pack(side=tk.LEFT, padx=5)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Eğitim Dosyası Seç",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Metni cümlelere böl
                sentences = re.split(r'[.!?]+', content)
                self.loaded_texts = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.file_info.config(text=f"✅ Yüklendi: {os.path.basename(file_path)}\n"
                                          f"Boyut: {file_size:.1f} KB | "
                                          f"Cümle: {len(self.loaded_texts)}",
                                     fg=self.colors['success'])
                
                self.log("Dosya başarıyla yüklendi!")
                self.log(f"Toplam {len(self.loaded_texts)} cümle bulundu.")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}")
    
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    
    def start_training(self):
        if not self.loaded_texts:
            messagebox.showwarning("Uyarı", "Lütfen önce bir dosya yükleyin!")
            return
        
        if self.is_training:
            messagebox.showinfo("Bilgi", "Eğitim zaten devam ediyor!")
            return
        
        # Eğitimi ayrı thread'de başlat
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self):
        self.is_training = True
        self.train_btn.config(state='disabled')
        
        try:
            self.log("=" * 50)
            self.log("🚀 Eğitim başlıyor...")
            self.log(f"Cihaz: {self.device}")
            self.log(f"Bit: {self.bit_var.get()}")
            self.log(f"Epoch: {self.epoch_var.get()}")
            
            # Tokenizer'ı eğit
            self.log("📝 Tokenizer hazırlanıyor...")
            self.tokenizer.fit(self.loaded_texts)
            vocab_size = len(self.tokenizer.word2idx)
            self.log(f"Kelime hazinesi boyutu: {vocab_size}")
            
            # Dataset oluştur
            dataset = TextDataset(self.loaded_texts, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Model oluştur
            self.log("🤖 Model oluşturuluyor...")
            bits = int(self.bit_var.get())
            self.model = MiniTransformer(vocab_size, bits=bits).to(self.device)
            
            # Optimizer ve loss
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Eğitim döngüsü
            total_steps = self.epoch_var.get() * len(dataloader)
            current_step = 0
            
            self.model.train()
            for epoch in range(self.epoch_var.get()):
                epoch_loss = 0
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs = inputs.unsqueeze(1).to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    current_step += 1
                    
                    # İlerleme güncelle
                    progress = (current_step / total_steps) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"Epoch {epoch+1}/{self.epoch_var.get()} - "
                                                   f"Batch {batch_idx+1}/{len(dataloader)}")
                
                avg_loss = epoch_loss / len(dataloader)
                self.log(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            self.log("✅ Eğitim tamamlandı!")
            self.update_model_stats()
            
            messagebox.showinfo("Başarılı", "Model eğitimi tamamlandı! Artık sohbet edebilirsiniz.")
            
        except Exception as e:
            self.log(f"❌ Hata: {str(e)}")
            messagebox.showerror("Hata", f"Eğitim sırasında hata: {str(e)}")
        
        finally:
            self.is_training = False
            self.train_btn.config(state='normal')
            self.progress_var.set(0)
            self.progress_label.config(text="Hazır")
    
    def send_message(self):
        if not self.model:
            messagebox.showwarning("Uyarı", "Önce bir model eğitmeniz gerekiyor!")
            return
        
        user_message = self.message_entry.get().strip()
        if not user_message:
            return
        
        # Kullanıcı mesajını göster
        self.chat_display.insert(tk.END, f"👤 Sen: {user_message}\n", 'user')
        self.message_entry.delete(0, tk.END)
        
        # Model cevabı
        try:
            response = self.generate_response(user_message)
            self.chat_display.insert(tk.END, f"🤖 AI: {response}\n\n", 'ai')
            self.chat_display.see(tk.END)
        except Exception as e:
            self.chat_display.insert(tk.END, f"❌ Hata: {str(e)}\n\n", 'error')
    
    def generate_response(self, prompt, max_length=50):
        self.model.eval()
        with torch.no_grad():
            # Prompt'u encode et
            input_ids = torch.tensor([self.tokenizer.encode(prompt, 20)]).to(self.device)
            
            generated = []
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token = torch.argmax(outputs[0, -1, :]).item()
                
                if next_token == 3:  # END token
                    break
                
                generated.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
            
            response = self.tokenizer.decode(generated)
            return response if response else "Hmm, düşünüyorum..."
    
    def update_model_stats(self):
        if not self.model:
            return
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size = total_params * 4 / (1024 * 1024)  # MB
        
        stats_info = f"""
╔════════════════════════════════════════╗
║         MODEL İSTATİSTİKLERİ          ║
╠════════════════════════════════════════╣
║ 📊 Toplam Parametre: {total_params:,}
║ 🎯 Eğitilebilir: {trainable_params:,}
║ 💾 Model Boyutu: ~{model_size:.2f} MB
║ 🔢 Quantization: {self.bit_var.get()}-bit
║ 📚 Kelime Sayısı: {len(self.tokenizer.word2idx)}
║ 🖥️ Cihaz: {self.device}
║ ⏰ Son Eğitim: {datetime.now().strftime('%H:%M:%S')}
╚════════════════════════════════════════╝
        """
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_info)
    
    def save_model(self):
        if not self.model:
            messagebox.showwarning("Uyarı", "Kaydedilecek model yok!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth")]
        )
        
        if file_path:
            torch.save({
                'model_state': self.model.state_dict(),
                'tokenizer': self.tokenizer.word2idx,
                'config': {
                    'vocab_size': len(self.tokenizer.word2idx),
                    'bits': int(self.bit_var.get())
                }
            }, file_path)
            
            messagebox.showinfo("Başarılı", "Model kaydedildi!")
            self.log(f"Model kaydedildi: {file_path}")
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth")]
        )
        
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                
                # Tokenizer'ı yükle
                self.tokenizer.word2idx = checkpoint['tokenizer']
                self.tokenizer.idx2word = {v: k for k, v in checkpoint['tokenizer'].items()}
                
                # Model'i oluştur ve yükle
                config = checkpoint['config']
                self.model = MiniTransformer(config['vocab_size'], bits=config['bits']).to(self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                
                self.bit_var.set(str(config['bits']))
                
                messagebox.showinfo("Başarılı", "Model yüklendi!")
                self.log(f"Model yüklendi: {file_path}")
                self.update_model_stats()
                
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya yüklenemedi: {str(e)}")

def main():
    root = tk.Tk()
    app = AITrainerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
