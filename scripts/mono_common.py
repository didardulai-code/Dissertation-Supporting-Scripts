import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLACK="#000000"; DARK="#333333"; MED="#8a8a8a"; GREY="#9a9a9a"; BAND="#dcdcdc"
def style():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
     "font.size":8.6,"axes.linewidth":0.6,"axes.unicode_minus":False,"figure.facecolor":"white",
     "axes.facecolor":"white","axes.edgecolor":"#9a9a9a","xtick.color":"#4a4a4a","ytick.color":"#4a4a4a",
     "axes.axisbelow":True,"pdf.fonttype":42})
