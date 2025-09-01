#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def parse_dat_main(filepath):
    """Estrae (times, mems) del processo principale (righe 'MEM') e normalizza t0=0."""
    times, mems = [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "MEM":
                # formato atteso: MEM <mem_mib> <time_s>
                try:
                    mem = float(parts[1])
                    t = float(parts[2])
                except (IndexError, ValueError):
                    continue
                times.append(t)
                mems.append(mem)
    if len(times) == 0:
        return np.array([]), np.array([])
    times = np.asarray(times, dtype=float)
    mems = np.asarray(mems, dtype=float)
    # normalizza i tempi a partire da zero per ogni serie
    t0 = times.min()
    times = times - t0
    # assicura ordinamento crescente (per sicurezza)
    order = np.argsort(times)
    return times[order], mems[order]

def collect_dat_files(folder, pattern="*.dat", recursive=False):
    folder = os.path.abspath(folder)
    if recursive:
        glob_pat = os.path.join(folder, "**", pattern)
        files = glob.glob(glob_pat, recursive=True)
    else:
        glob_pat = os.path.join(folder, pattern)
        files = glob.glob(glob_pat)
    # Ordina per nome per riproducibilità
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser(
        description="Plot multiplo di file mprof .dat (solo main) da una cartella, con curva media."
    )
    ap.add_argument("folder", help="Cartella contenente i file .dat")
    ap.add_argument("-t", "--title", default="Memory Profile",
                    help="Titolo del grafico")
    ap.add_argument("-s", "--save", default=None,
                    help="Nome file per salvare il grafico (es. plot.png)")
    ap.add_argument("--pattern", default="*.dat",
                    help="Pattern dei file (default: *.dat)")
    ap.add_argument("--recursive", action="store_true",
                    help="Cerca i .dat in modo ricorsivo nella cartella")
    ap.add_argument("--grid-points", type=int, default=1000,
                    help="Punti della griglia temporale per la curva media (default: 1000)")
    ap.add_argument("--labels", default=None,
                    help="Etichette personalizzate, separate da virgola (devono coincidere col numero di file trovati)")
    args = ap.parse_args()

    files = collect_dat_files(args.folder, pattern=args.pattern, recursive=args.recursive)
    if not files:
        print(f"[ERR] Nessun file che combacia con '{args.pattern}' in: {args.folder}")
        return

    # Etichette: se non date, usa basename dei file
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(files):
            print(f"[ERR] Numero di etichette ({len(labels)}) diverso dal numero di file ({len(files)}).")
            return
    else:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in files]

    series = []
    for fpath, label in zip(files, labels):
        t, m = parse_dat_main(fpath)
        if len(t) < 2:
            print(f"[WARN] Dati insufficienti (MEM) in '{fpath}' — saltato.")
            continue
        series.append({"t": t, "m": m, "label": label})

    if not series:
        print("[ERR] Nessuna serie valida da plottare.")
        return

    # Costruisci griglia comune nel range condiviso [0, min(max_t_i)]
    max_times = [s["t"][-1] for s in series]
    t_end = float(min(max_times))
    if t_end <= 0:
        print("[ERR] Intervallo temporale condiviso non valido per la media.")
        return
    t_grid = np.linspace(0.0, t_end, num=max(50, args.grid_points))

    # Interpola ogni serie su t_grid e calcola la media
    interp_vals = []
    for s in series:
        interp = np.interp(t_grid, s["t"], s["m"])
        interp_vals.append(interp)
    mean_curve = np.mean(np.vstack(interp_vals), axis=0)

    # Plot
    plt.figure(figsize=(10, 6))

    for s in series:
        peak_idx = int(np.argmax(s["m"]))
        peak_val = s["m"][peak_idx]
        plt.plot(
            s["t"], s["m"],
            linestyle="-", linewidth=0.8, alpha=0.9,
            label=f"{s['label']} (peak {peak_val:.1f} MiB)"
        )

    # Curva media in evidenza (tratteggiata)
    plt.plot(t_grid, mean_curve, linestyle="--", linewidth=2.5, alpha=0.95, label="mean (common grid)")

    plt.xlabel("Time (s)")
    plt.ylabel("Memory used (MiB)")
    plt.title(args.title)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"[OK] Plot salvato in {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
