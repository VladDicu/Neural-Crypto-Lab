import gradio as gr
from motor_cripto import simuleaza_laborator, analizeaza_conexiune_vizual, compara_sisteme

with gr.Blocks() as interfata:
    gr.Markdown("# 🛡️ Dashboard Central: Securitate Cibernetică & Analiză Neurală")

    with gr.Tabs():
        # --- FILA 1: SIMULATORUL DE ATAC ---
        with gr.TabItem("⚔️ Simulator de Atac (Rețea TCP)"):
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

        # --- FILA 2: VIZUALIZAREA CONEXIUNII ---
        with gr.TabItem("🌌 Analiză Vizuală: Conexiunea Alice-Bob"):
            gr.Markdown("Această secțiune izolează traficul legitim și analizează modul în care rețeaua codifică informația în spațiul latent, ignorând prezența atacatorului.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    in_mesaj_viz = gr.Textbox(lines=4, label="Mesaj pentru Analiză", value="Flux de date de la Str. Ankara către Str. Vasile Lascăr.")
                    epoci_viz = gr.Slider(100, 800, 300, step=50, label="Epoci de antrenament")
                    btn_viz = gr.Button("Generează Diagrame", variant="primary")
                    
                    radar_video = gr.Image(label="Constelația Datelor (Fază I/Q)")
                    
                with gr.Column(scale=2):
                    grafic_pca = gr.Plot(label="Spațiul Latent 3D (Click & Drag pentru a roti cubul)")

            btn_viz.click(analizeaza_conexiune_vizual, [in_mesaj_viz, epoci_viz], [grafic_pca, radar_video])
            # --- FILA 3: COMPARAȚIE CLASIC vs. AI ---
        with gr.TabItem("⚖️ Evoluția Criptografiei: Clasic vs. AI"):
            gr.Markdown("Test de penetrare: Un bot de nivel mediu atacă un algoritm clasic (matematic) comparativ cu criptografia generată de AI.")
            
            with gr.Row():
                with gr.Column():
                    in_mesaj_comp = gr.Textbox(label="Mesaj Test", value="Sistemele rigide sunt vulnerabile la forta bruta.")
                    btn_comp = gr.Button("Lansează Atacul (Bot Script Kiddie)", variant="primary")
                    cifru_clasic_out = gr.Textbox(label="Cifru Clasic Interceptat (Binar/Hex ascuns)")
                    
                with gr.Column():
                    rez_clasic_out = gr.Textbox(label="🛑 Vulnerabilitate Clasic (Matematic)", lines=4)
                    rez_neural_out = gr.Textbox(label="🛡️ Reziliență Neurală (AI)", lines=4)
                    
            btn_comp.click(compara_sisteme, inputs=[in_mesaj_comp], outputs=[cifru_clasic_out, rez_clasic_out, rez_neural_out])

if __name__ == "__main__":
    interfata.launch(theme=gr.themes.Monochrome())
