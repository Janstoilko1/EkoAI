import os
import re
 
# ─── NASTAVLJIVI SPREMENLJIVKI ────────────────────────────────────────────────
VHODNA_POT = r"C:\Users\Jan\OneDrive - Univerza v Mariboru\Desktop\EkoAI\Audio_logs\steklo"   # mapa, kjer so datoteke
OSNOVNO_IME = "steklenikozarcek"       # osnova novega imena (brez številke)
# ─────────────────────────────────────────────────────────────────────────────
 
 
def preimenuj_datoteke(pot: str, ime: str) -> None:
    """
    Preimenuje vse datoteke v mapi `pot` po vzorcu `ime + counter`.
    Primer: dokument1.txt, dokument2.txt, dokument3.txt ...
 
    Mape so preskočene — preimenujem samo datoteke.
    """
    if not os.path.isdir(pot):
        print(f"[NAPAKA] Mapa ne obstaja: {pot}")
        return
 
    # Zberemo samo datoteke (ne map), razvrščene po imenu
    datoteke = sorted(
        vnos for vnos in os.listdir(pot)
        if os.path.isfile(os.path.join(pot, vnos))
    )
 
    if not datoteke:
        print("[INFO] V mapi ni nobene datoteke.")
        return
 
    print(f"Najdeno {len(datoteke)} datotek(e) v: {pot}\n")
 
    for counter, staro_ime in enumerate(datoteke, start=1):
        _, koncnica = os.path.splitext(staro_ime)
        novo_ime = f"{ime}{counter}{koncnica}"
 
        stara_pot = os.path.join(pot, staro_ime)
        nova_pot  = os.path.join(pot, novo_ime)
 
        if stara_pot == nova_pot:
            print(f"  [PRESKOČENO] {staro_ime} (ime se ne spremeni)")
            continue
 
        if os.path.exists(nova_pot):
            print(f"  [PRESKOČENO] {staro_ime} → {novo_ime}  (cilj že obstaja!)")
            continue
 
        os.rename(stara_pot, nova_pot)
        print(f"  {staro_ime}  →  {novo_ime}")
 
    print("\nPreimenovanje končano.")