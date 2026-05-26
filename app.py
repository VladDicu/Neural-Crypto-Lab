import gradio as gr
from motor_cripto import simuleaza_laborator

with gr.Blocks(theme=gr.themes.Monochrome()) as interfata:
    gr.Markdown("# 🛡️ Sistem Multi-Nivel de Securitate Cibernetică și Criptografie Neurală")

    with gr.Row():
        with gr.Column():
            in_mesaj = gr.Textbox(label="Mesaj Secret (Payload TCP)", value="Date confidentiale transmise prin canal securizat.")
            
            tip_eve = gr.Radio(
                ["Atacator de Rând (Bot/Script Kiddie)", "Super-Eve (Deep Learning)", "Clonă (Knowledge Distillation)"], 
                value="Atacator de Rând (Bot/Script Kiddie)", label="🤖 Arhitectură Atacator"
            )
            
            tip_scenariu = gr.Radio(
                ["Scenariul 1: Compromitere Totală (Man-in-the-Middle)", 
                 "Scenariul 2: Interceptare Pasivă (Eavesdropping)", 
                 "Scenariul 3: Furt + Bruiaj Canal (Jamming)"],
                value="Scenariul 2: Interceptare Pasivă (Eavesdropping)", label="⚔️ Vector de Atac"
            )
            
            epoci_slider = gr.Slider(50, 400, 150, step=50, label="Epoci (Timp antrenament)")
            btn_run = gr.Button("Inițiază Transmisia", variant="primary")

        with gr.Column():
            out_retea = gr.Textbox(label="📡 Log Trafic Rețea (Latență TCP)", lines=6)
            out_bob = gr.Textbox(label="✅ Scutul de Integritate BOB (Receptor)", lines=4)
            out_eve = gr.Textbox(label="🚨 Date Extrase de EVE", lines=3)
            grafic = gr.Plot(label="Telemetrie Atac")

    btn_run.click(simuleaza_laborator, [in_mesaj, tip_eve, tip_scenariu, epoci_slider], [out_retea, out_bob, out_eve, grafic])

if __name__ == "__main__":
    interfata.launch()
