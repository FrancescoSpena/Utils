import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Plot memory_profiler .dat file con figli")
    parser.add_argument("file", type=str, help="File .dat generato da mprof")
    parser.add_argument("-t", "--title", type=str, default="Memory Profile",
                        help="Titolo del grafico")
    parser.add_argument("-s", "--save", type=str, default=None,
                        help="Nome file per salvare il grafico (es. plot.png)")
    args = parser.parse_args()

    # dizionario {child_id: (time[], mem[])}
    data = defaultdict(lambda: ([], []))

    with open(args.file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "MEM":
                # processo principale (id = 0)
                mem = float(parts[1])
                t = float(parts[2])
                data[0][0].append(t)
                data[0][1].append(mem)
            elif parts[0] == "CHLD":
                cid = int(parts[1])
                mem = float(parts[2])
                t = float(parts[3])
                data[cid][0].append(t)
                data[cid][1].append(mem)

    # normalizza tempi (partono da 0)
    t0 = min(min(times) for times, _ in data.values())
    for cid in data:
        times, mems = data[cid]
        data[cid] = ([t - t0 for t in times], mems)

    plt.figure(figsize=(14, 6)) 

    # plottiamo solo il processo principale (cid == 0)
    times, mems = data[0]
    peak_idx = max(range(len(mems)), key=lambda i: mems[i])
    peak_val = mems[peak_idx]

    plt.plot(times, mems, linestyle=":", linewidth=1.8, alpha=0.9, label=f"main (peak {peak_val:.1f} MiB)")
    plt.scatter(times[peak_idx], peak_val, s=20, zorder=5, color="red")

    plt.xlabel("Time (s)")
    plt.ylabel("Memory used (MiB)")
    plt.title(args.title)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()



    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"[OK] Plot salvato in {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
