import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt

BLOCK_SIZE = 64

class AgentCriptografic(nn.Module):
    def __init__(self):
        super().__init__()
        self.retea = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 64), nn.Tanh())
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

# --- Încărcarea Modelelor din Colab ---
def incarca_agenti():
    alice, bob, eve = AgentCriptografic(), AgentCriptografic(), SuperEve()
    stare_ab = torch.load('models/alice_bob_v1.pt', map_location='cpu', weights_only=True)
    alice.load_state_dict(stare_ab['alice_state'])
    bob.load_state_dict(stare_ab['bob_state'])
    eve.load_state_dict(torch.load('models/super_eve_v1.pt', map_location='cpu', weights_only=True))
    alice.eval(); bob.eval(); eve.eval()
    return alice, bob, eve

ALICE, BOB, EVE = incarca_agenti()

# --- Funcții Utilitare ---
def codare_tmr(biti): return [b for b in biti for _ in range(3)]
def decodare_tmr(biti): return [1 if sum(biti[i:i+3]) >= 2 else 0 for i in range(0, len(biti), 3)]

def prepara_date(text):
    biti = [int(b) for byte in text.encode('utf-8') for b in format(byte, '08b')]
    semnal = [1.0 if b == 1 else -1.0 for b in codare_tmr(biti)]
    if len(semnal) % BLOCK_SIZE != 0: semnal += [-1.0] * (BLOCK_SIZE - len(semnal) % BLOCK_SIZE)
    blocuri = [torch.tensor([semnal[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]], dtype=torch.float32) for i in range(max(1, len(semnal) // BLOCK_SIZE))]
    return blocuri, len(biti)

def reconstruieste(blocuri, lungime):
    biti_curati = decodare_tmr([(b.item() > 0) for bloc in blocuri for b in bloc[0]])[:lungime]
    sir = "".join(str(b) for b in biti_curati)
    bytes_out = bytearray(int(sir[i:i+8], 2) for i in range(0, len(sir), 8) if len(sir[i:i+8])==8)
    return bytes_out.decode('utf-8', errors='replace')

def flux_cbc(agent, blocuri, cheie, vi, mod='enc'):
    rez = []; c_ant = vi
    with torch.no_grad():
        for b in blocuri:
            if mod == 'enc':
                out = agent(-(b * c_ant), cheie)
                rez.append(out); c_ant = out.sign()
            else:
                rez.append(-(agent(b, cheie).sign() * c_ant.sign()))
                c_ant = b.sign()
    return rez

# --- Scenarii (Fără antrenament, doar execuție) ---
def run_zero_knowledge(text):
    if not text: return "Aștept date..."
    blocuri, lungime = prepara_date(text)
    cheie, vi, cheie_eve = torch.randn(1, 64).sign(), torch.randn(1, 64).sign(), torch.zeros(1, 64)
    cifru = flux_cbc(ALICE, blocuri, cheie, vi, 'enc')
    eve_standard = AgentCriptografic() # Inițializată la întâmplare
    return reconstruieste(flux_cbc(eve_standard, cifru, cheie_eve, vi, 'dec'), lungime)

def run_super_eve(text):
    if not text: return "Aștept date..."
    blocuri, lungime = prepara_date(text)
    cheie, vi = torch.randn(1, 64).sign(), torch.randn(1, 64).sign()
    cifru = flux_cbc(ALICE, blocuri, cheie, vi, 'enc')
    return reconstruieste(flux_cbc(EVE, cifru, torch.zeros(1, 64), vi, 'dec'), lungime)

# ==========================================
# INTERFAȚA WEB
# ==========================================
with gr.Blocks(theme=gr.themes.Monochrome()) as interfata:
    gr.Markdown("## 🛡️ Criptografie Neurală (Accelerat via modele .pt)")
    
    with gr.Tab("1. Atac Standard (Eșec)"):
        in_s1 = gr.Textbox(label="Mesaj Alice")
        btn_s1 = gr.Button("Interceptare Zero-Knowledge")
        out_s1 = gr.Textbox(label="Rezultat Eve (Zgomot)")
        btn_s1.click(run_zero_knowledge, inputs=in_s1, outputs=out_s1)

    with gr.Tab("2. Super-Eve (Extragere Parțială)"):
        in_s2 = gr.Textbox(label="Mesaj Alice")
        btn_s2 = gr.Button("Aplică Modelul Pre-antrenat (Super-Eve)")
        out_s2 = gr.Textbox(label="Text Recuperat de Eve")
        btn_s2.click(run_super_eve, inputs=in_s2, outputs=out_s2)

if __name__ == "__main__":
    interfata.launch()