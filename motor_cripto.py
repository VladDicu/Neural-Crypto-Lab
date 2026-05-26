import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import difflib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import tempfile
import os

BLOCK_SIZE = 64

# ==========================================
# 1. ARHITECTURI NEURALE REFĂCUTE
# ==========================================
class AgentCriptografic(nn.Module):
    def __init__(self):
        super().__init__()
        # Separare pentru a putea extrage spațiul latent (256 dimensiuni)
        self.encoder = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(256, 64), nn.Tanh())
        
    def forward(self, text, cheie): 
        return self.decoder(self.encoder(torch.cat((text, cheie), dim=1)))
        
    def obtine_spatiu_latent(self, text, cheie):
        return self.encoder(torch.cat((text, cheie), dim=1))

class ScriptKiddieEve(nn.Module):
    def __init__(self):
        super().__init__()
        self.retea = nn.Sequential(nn.Linear(128, 8), nn.ReLU(), nn.Linear(8, 64), nn.Tanh())
    def forward(self, text, cheie): return self.retea(torch.cat((text, cheie), dim=1))

class SuperEve(nn.Module):
    def __init__(self):
        super().__init__()
        self.retea = nn.Sequential(
            nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, 64), nn.Tanh()
        )
    def forward(self, text, cheie): return self.retea(torch.cat((text, cheie), dim=1))

# ==========================================
# 2. FUNCȚII UTILITARE & REȚEA TCP
# ==========================================
def codare_tmr(biti): return [b for b in biti for _ in range(3)]
def decodare_tmr(biti): return [1 if sum(biti[i:i+3]) >= 2 else 0 for i in range(0, len(biti), 3)]

def prepara_date(text):
    biti = [int(b) for byte in text.encode('utf-8') for b in format(byte, '08b')]
    semnal = [1.0 if b == 1 else -1.0 for b in codare_tmr(biti)]
    if len(semnal) % BLOCK_SIZE != 0: semnal += [-1.0] * (BLOCK_SIZE - len(semnal) % BLOCK_SIZE)
    blocuri = [torch.tensor([semnal[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]], dtype=torch.float32) for i in range(max(1, len(semnal) // BLOCK_SIZE))]
    return blocuri, len(biti), semnal

def reconstruieste(blocuri, lungime):
    biti_curati = decodare_tmr([(b.item() > 0) for bloc in blocuri for b in bloc[0]])[:lungime]
    sir = "".join(str(b) for b in biti_curati)
    bytes_out = bytearray(int(sir[i:i+8], 2) for i in range(0, len(sir), 8) if len(sir[i:i+8])==8)
    return bytes_out.decode('utf-8', errors='replace')

def flux_cbc(agent, blocuri, cheie, vi, mod='enc'):
    rez = []; c_ant = vi
    for b in blocuri:
        if mod == 'enc':
            out = agent(-(b * c_ant), cheie)
            rez.append(out); c_ant = out.sign()
        else:
            rez.append(-(agent(b, cheie).sign() * c_ant.sign()))
            c_ant = b.sign()
    return rez

def simulare_retea_tcp(blocuri_cifru):
    log_retea = []
    src_ip = "192.168.1.10 (Str. Ankara)"
    dst_ip = "10.0.0.50 (Str. Vasile Lascăr)"
    for i, bloc in enumerate(blocuri_cifru):
        latenta = random.uniform(12.5, 55.0)
        log_retea.append(f"[TCP ACK] Pachet {i:03d} | Src: {src_ip} -> Dst: {dst_ip} | Latență: {latenta:.1f} ms")
    return blocuri_cifru, "\n".join(log_retea)

# ==========================================
# 3. MOTOR SIMULARE ATAC (FILA 1)
# ==========================================
def simuleaza_laborator(mesaj, tip_eve, tip_scenariu, epoci):
    if not mesaj: return "Aștept date...", "", "", None
    
    alice, bob = AgentCriptografic(), AgentCriptografic()
    opt_ab = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.005)
    loss_fn_ab = nn.MSELoss()
    
    for _ in range(int(epoci)):
        msg_rnd, key_rnd = torch.randn(128, 64).sign(), torch.randn(128, 64).sign()
        opt_ab.zero_grad()
        loss_ab = loss_fn_ab(bob(alice(msg_rnd, key_rnd), key_rnd), msg_rnd)
        loss_ab.backward(); opt_ab.step()

    if tip_eve == "Atacator de Rând (Bot/Script Kiddie)":
        eve = ScriptKiddieEve(); lr_eve, loss_eve_fn = 0.0001, nn.MSELoss() 
    elif tip_eve == "Super-Eve (Deep Learning)":
        eve = SuperEve(); lr_eve, loss_eve_fn = 0.002, nn.MSELoss()
    else: 
        eve = AgentCriptografic(); lr_eve, loss_eve_fn = 0.008, nn.L1Loss() 

    opt_eve = optim.Adam(eve.parameters(), lr=lr_eve)
    istoric_loss_eve = []

    for _ in range(int(epoci)):
        msg_rnd, key_rnd = torch.randn(128, 64).sign(), torch.randn(128, 64).sign()
        cifru_interceptat = alice(msg_rnd, key_rnd).detach()
        opt_eve.zero_grad()
        loss_eve = loss_eve_fn(eve(cifru_interceptat, torch.zeros_like(key_rnd)), msg_rnd)
        loss_eve.backward(); opt_eve.step()
        istoric_loss_eve.append(loss_eve.item())

    blocuri, lungime, _ = prepara_date(mesaj)
    cheie, vi = torch.randn(1, 64).sign(), torch.randn(1, 64).sign()
    cheie_falsa_eve = torch.zeros(1, 64)
    
    cifru_real = flux_cbc(alice, blocuri, cheie, vi, 'enc')
    cifru_tranzitat, log_tcp = simulare_retea_tcp(cifru_real)
    
    text_brut_bob = ""
    rezultat_eve = ""

    with torch.no_grad():
        if tip_scenariu == "Scenariul 1: Compromitere Totală (Man-in-the-Middle)":
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_tranzitat, cheie_falsa_eve, vi, 'dec'), lungime)
            cifru_falsificat = flux_cbc(eve, blocuri, cheie_falsa_eve, vi, 'enc')
            text_brut_bob = reconstruieste(flux_cbc(bob, cifru_falsificat, cheie, vi, 'dec'), lungime)
        elif tip_scenariu == "Scenariul 2: Interceptare Pasivă (Eavesdropping)":
            text_brut_bob = reconstruieste(flux_cbc(bob, cifru_tranzitat, cheie, vi, 'dec'), lungime)
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_tranzitat, cheie_falsa_eve, vi, 'dec'), lungime)
        elif tip_scenariu == "Scenariul 3: Furt + Bruiaj Canal (Jamming)":
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_tranzitat, cheie_falsa_eve, vi, 'dec'), lungime)
            cifru_bruiat = [c + (torch.randn_like(c) * 1.5) for c in cifru_tranzitat]
            text_brut_bob = reconstruieste(flux_cbc(bob, cifru_bruiat, cheie, vi, 'dec'), lungime)

    grad_similitudine = difflib.SequenceMatcher(None, mesaj, text_brut_bob).ratio()
    if grad_similitudine > 0.85: 
        rezultat_bob = f"🔒 [INTEGRITATE VERIFICATĂ] Semnătură Neurală Validă ({grad_similitudine*100:.1f}%)\n--> {text_brut_bob}"
    else:
        rezultat_bob = f"🚨 [ALERTĂ CRITICĂ] Integritate eșuată ({grad_similitudine*100:.1f}%).\nConexiune RESPINSĂ automat."

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(istoric_loss_eve, color='red', label="Curba Erorii lui Eve")
    ax.set_title(f"Analiza Atacului: {tip_eve}")
    ax.set_ylabel("Loss"); ax.set_xlabel("Epoci")
    ax.legend(); fig.tight_layout()

    return log_tcp, rezultat_bob, rezultat_eve, fig

# ==========================================
# 4. MOTOR VIZUALIZARE CONEXIUNE (FILA 2)
# ==========================================
def genereaza_animatie_radar(semnal_orig, semnal_criptat, semnal_decriptat):
    limit = min(128, len(semnal_orig))
    if limit % 2 != 0: limit -= 1
    
    orig_pts = np.array(semnal_orig[:limit]).reshape(-1, 2)
    cript_pts = np.array(semnal_criptat[:limit]).reshape(-1, 2)
    dec_pts = np.array(semnal_decriptat[:limit]).reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
    ax.set_facecolor('black')
    scat = ax.scatter([], [], c='cyan', s=30, alpha=0.8, edgecolors='white')

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title("Constelația Datelor pe Traseul Ankara -> Lascăr", color='white')
    ax.axis('off')

    def update(frame):
        if frame < 15:
            t = frame / 15.0
            scat.set_offsets((1-t)*orig_pts + t*cript_pts); scat.set_color('cyan')
        elif frame < 30:
            scat.set_offsets(cript_pts + np.random.normal(0, 0.05, cript_pts.shape)); scat.set_color('red')
        else:
            t = (frame - 30) / 15.0
            scat.set_offsets((1-t)*cript_pts + t*dec_pts); scat.set_color('lime')
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=45, interval=80, blit=True)
    temp_path = tempfile.mktemp(suffix='.gif')
    ani.save(temp_path, writer='pillow', fps=15)
    plt.close(fig)
    return temp_path

def analizeaza_conexiune_vizual(mesaj, epoci):
    if not mesaj: return None, None
    
    alice, bob = AgentCriptografic(), AgentCriptografic()
    opt = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.005)
    loss_fn = nn.MSELoss()
    
    alice.train(); bob.train()
    for _ in range(int(epoci)):
        msg_rnd, key_rnd = torch.randn(128, BLOCK_SIZE).sign(), torch.randn(128, BLOCK_SIZE).sign()
        opt.zero_grad()
        loss = loss_fn(bob(alice(msg_rnd, key_rnd), key_rnd), msg_rnd)
        loss.backward(); opt.step()

    blocuri, lungime, semnal_brut = prepara_date(mesaj)
    tensor_mesaj = torch.cat(blocuri, dim=0)
    
    chei_dinamice = []
    cheie_baza = torch.randn(1, BLOCK_SIZE).sign()
    for i in range(len(blocuri)):
        nonce = torch.tensor([1.0 if b == '1' else -1.0 for b in format(i, f'0{BLOCK_SIZE}b')], dtype=torch.float32)
        chei_dinamice.append(-(cheie_baza * nonce))
    tensor_chei = torch.cat(chei_dinamice, dim=0)

    alice.eval(); bob.eval()
    with torch.no_grad():
        tensor_cifru = alice(tensor_mesaj, tensor_chei)
        tensor_decriptat = bob(tensor_cifru, tensor_chei)
        
        # PCA 3D
        vectori_latenti = alice.obtine_spatiu_latent(tensor_mesaj, tensor_chei).numpy()
        zgomot_text = torch.randn(150, BLOCK_SIZE).sign()
        zgomot_chei = torch.randn(150, BLOCK_SIZE).sign()
        vectori_zgomot = alice.obtine_spatiu_latent(zgomot_text, zgomot_chei).numpy()

    toate_datele = np.vstack([vectori_latenti, vectori_zgomot])
    pca = PCA(n_components=3)
    comp_3d = pca.fit_transform(toate_datele)

    df = pd.DataFrame(comp_3d, columns=['X', 'Y', 'Z'])
    df['Tip Date'] = ['Informație Legitimă (Spre Lascăr)'] * len(vectori_latenti) + ['Zgomot Criptografic'] * len(vectori_zgomot)
    df['Dimensiune'] = [6] * len(vectori_latenti) + [2] * len(vectori_zgomot)

    fig_3d = px.scatter_3d(
        df, x='X', y='Y', z='Z', color='Tip Date', size='Dimensiune', opacity=0.8,
        color_discrete_sequence=['#00ff00', '#444444'],
        title="Proiecția PCA: Spațiul Latent al Conexiunii"
    )
    fig_3d.update_layout(scene=dict(bgcolor="black"), paper_bgcolor="black", font_color="white", margin=dict(l=0, r=0, b=0, t=40))

    # Generare Radar
    valori_criptate_brute = tensor_cifru.flatten().tolist()
    valori_decriptate_brute = tensor_decriptat.flatten().tolist()
    gif_path = genereaza_animatie_radar(semnal_brut, valori_criptate_brute, valori_decriptate_brute)

    return fig_3d, gif_path
