import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
import matplotlib.pyplot as plt

BLOCK_SIZE = 64

# ==========================================
# 1. ARHITECTURI NEURALE
# ==========================================
class AgentCriptografic(nn.Module):
    # Putere Moderată (Alice, Bob, Eve Standard/Clonă)
    def __init__(self):
        super().__init__()
        self.retea = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 64), nn.Tanh())
    def forward(self, text, cheie): return self.retea(torch.cat((text, cheie), dim=1))

class SuperEve(nn.Module):
    # Putere Masivă (Deep Learning cu 3 straturi)
    def __init__(self):
        super().__init__()
        self.retea = nn.Sequential(
            nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, 64), nn.Tanh()
        )
    def forward(self, text, cheie): return self.retea(torch.cat((text, cheie), dim=1))

# ==========================================
# 2. FUNCȚII UTILITARE (TMR & CBC)
# ==========================================
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
    for b in blocuri:
        if mod == 'enc':
            out = agent(-(b * c_ant), cheie)
            rez.append(out); c_ant = out.sign()
        else:
            rez.append(-(agent(b, cheie).sign() * c_ant.sign()))
            c_ant = b.sign()
    return rez

# ==========================================
# 3. MOTORUL DE SIMULARE (ANTRENAMENT RAPID & ATAC)
# ==========================================
def simuleaza_laborator(mesaj, tip_eve, tip_scenariu, epoci):
    if not mesaj: return "Aștept date...", "", None
    
    # 3.1. Inițializare și Antrenament Conexiune Securizată (Alice & Bob)
    alice, bob = AgentCriptografic(), AgentCriptografic()
    opt_ab = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.005)
    loss_fn_ab = nn.MSELoss()
    
    for _ in range(int(epoci)):
        msg_rnd, key_rnd = torch.randn(128, 64).sign(), torch.randn(128, 64).sign()
        opt_ab.zero_grad()
        loss_ab = loss_fn_ab(bob(alice(msg_rnd, key_rnd), key_rnd), msg_rnd)
        loss_ab.backward(); opt_ab.step()

    # 3.2. Pregătire Profil Eve
    if tip_eve == "Putere Moderată (Standard)":
        eve = AgentCriptografic()
        lr_eve, loss_eve_fn = 0.005, nn.MSELoss()
    elif tip_eve == "Super-Eve (Deep Learning)":
        eve = SuperEve()
        lr_eve, loss_eve_fn = 0.002, nn.MSELoss()
    else: # Clonare (Knowledge Distillation)
        eve = AgentCriptografic()
        lr_eve, loss_eve_fn = 0.008, nn.L1Loss() # Distilarea folosește L1 pentru clonare exactă

    opt_eve = optim.Adam(eve.parameters(), lr=lr_eve)
    istoric_loss_eve = []

    # 3.3. Antrenament Adversarial pentru Eve
    for _ in range(int(epoci)):
        msg_rnd, key_rnd = torch.randn(128, 64).sign(), torch.randn(128, 64).sign()
        cifru_interceptat = alice(msg_rnd, key_rnd).detach()
        
        opt_eve.zero_grad()
        # Dacă e clonă, Eve atacă prin L1 Loss, altfel atacă textul prin MSE
        loss_eve = loss_eve_fn(eve(cifru_interceptat, torch.zeros_like(key_rnd)), msg_rnd)
        loss_eve.backward(); opt_eve.step()
        istoric_loss_eve.append(loss_eve.item())

    # 3.4. Execuția Scenariului de Atac asupra Datelor Reale
    blocuri, lungime = prepara_date(mesaj)
    cheie, vi = torch.randn(1, 64).sign(), torch.randn(1, 64).sign()
    cheie_falsa_eve = torch.zeros(1, 64)
    
    cifru_real = flux_cbc(alice, blocuri, cheie, vi, 'enc')
    
    rezultat_bob = ""
    rezultat_eve = ""

    with torch.no_grad():
        if tip_scenariu == "Scenariul 1: Compromitere Totală (Man-in-the-Middle)":
            # Eve prinde cifrul, încearcă să-l citească, apoi trimite propriul ei cifru corupt către Bob
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_real, cheie_falsa_eve, vi, 'dec'), lungime)
            cifru_falsificat = flux_cbc(eve, blocuri, cheie_falsa_eve, vi, 'enc')
            rezultat_bob = reconstruieste(flux_cbc(bob, cifru_falsificat, cheie, vi, 'dec'), lungime)
            
        elif tip_scenariu == "Scenariul 2: Interceptare Pasivă (Eavesdropping)":
            # Conexiunea rămâne intactă. Bob citește perfect. Eve ascultă "pe fir".
            rezultat_bob = reconstruieste(flux_cbc(bob, cifru_real, cheie, vi, 'dec'), lungime)
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_real, cheie_falsa_eve, vi, 'dec'), lungime)
            
        elif tip_scenariu == "Scenariul 3: Furt + Bruiaj Canal (Jamming)":
            # Eve citește cifrul real, dar aplică zgomot sever pe cablu, stricând semnalul spre Bob
            rezultat_eve = reconstruieste(flux_cbc(eve, cifru_real, cheie_falsa_eve, vi, 'dec'), lungime)
            cifru_bruiat = [c + (torch.randn_like(c) * 1.5) for c in cifru_real] # 150% zgomot adăugat
            rezultat_bob = reconstruieste(flux_cbc(bob, cifru_bruiat, cheie, vi, 'dec'), lungime)

    # 3.5. Generare Grafic Telemetrie
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(istoric_loss_eve, color='red', label="Curba Erorii lui Eve")
    ax.set_title(f"Analiza Atacului: {tip_eve}")
    ax.set_ylabel("Loss"); ax.set_xlabel("Epoci")
    ax.legend(); fig.tight_layout()

    return rezultat_bob, rezultat_eve, fig

# ==========================================
# 4. INTERFAȚA WEB (GRADIO)
# ==========================================
with gr.Blocks(theme=gr.themes.Monochrome()) as interfata:
    gr.Markdown("# 🛡️ Centrul de Comandă: Criptografie vs. Securitate Cibernetică")
    gr.Markdown("Alege profilul atacatorului și tactica folosită pentru a testa reziliența rețelei Alice-Bob.")

    with gr.Row():
        with gr.Column():
            in_mesaj = gr.Textbox(label="Mesaj Secret (Alice -> Bob)", value="Acesta este un test clasificat.")
            
            tip_eve = gr.Radio(
                ["Putere Moderată (Standard)", "Super-Eve (Deep Learning)", "Clonă (Knowledge Distillation)"], 
                value="Super-Eve (Deep Learning)", label="🤖 1. Selectează Arhitectura lui Eve"
            )
            
            tip_scenariu = gr.Radio(
                ["Scenariul 1: Compromitere Totală (Man-in-the-Middle)", 
                 "Scenariul 2: Interceptare Pasivă (Eavesdropping)", 
                 "Scenariul 3: Furt + Bruiaj Canal (Jamming)"],
                value="Scenariul 2: Interceptare Pasivă (Eavesdropping)", label="⚔️ 2. Selectează Vectorul de Atac"
            )
            
            epoci_slider = gr.Slider(50, 400, 150, step=50, label="Epoci (Timp antrenament & infiltrare)")
            btn_run = gr.Button("Lansează Simularea", variant="primary")

        with gr.Column():
            out_bob = gr.Textbox(label="✅ Recepție BOB (Receptor Legitim)", lines=3)
            out_eve = gr.Textbox(label="🚨 Date furate de EVE (Atacator)", lines=3)
            grafic = gr.Plot(label="Telemetria Atacului")

    btn_run.click(simuleaza_laborator, [in_mesaj, tip_eve, tip_scenariu, epoci_slider], [out_bob, out_eve, grafic])

if __name__ == "__main__":
    interfata.launch()
