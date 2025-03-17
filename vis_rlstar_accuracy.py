import os
import glob
import re



delta = ["02", "01"]

def extract_accuracy(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            match = re.search(r'accuracy:([01]\.[0-9]+)', content)
            if match:
                return float(match.group(1))
    except FileNotFoundError:
        pass
    return None

f = open(f"results/acc_zip2_rlstar.tex","w")
f.write("""
\\begin{tikzpicture}
    \\begin{axis}[
        width=5.5cm,
        height=5cm,
        xlabel={Iteration of RL-STaR},
        legend pos=outer north east,
        grid=major,
        grid style={dashed,gray!30},
        xmin=1, xmax=10,
        ymin=0, ymax=1.1,
       title={LLMs},
    ] """)
for i,p in enumerate(delta):
    f.write(""" \\addplot[c%s, thick] table[row sep=\\\\]{\n"""%(i+1))
    f.write("""        x y \\\\""")
    result_old = 0
    for i in range(1, 31):
        log_filenames = sorted(glob.glob(f"results/rlstar_zip2_3_t_05_{p}_*/eval_test_{i}.log"))
        log_filename = None
        if len(log_filenames) > 0:
             log_filename = log_filenames[-1]
        else:
            if result_old == 1.0:
                f.write(f"{i} {result_old} \\\\\n")
        if log_filename is not None:
            result = extract_accuracy(log_filename)
            if result:
                f.write(f"{i} {result} \\\\\n")
                print(f"{i} {result} \\\\")
                result_old = result
    f.write("};\n")
f.write(""" \\end{axis}
\\end{tikzpicture}
""")
