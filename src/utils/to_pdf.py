import pickle
import os
import glob
from dotmap import DotMap
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import requests
import uuid
from weasyprint import HTML

def draw_matrix(data, norm = False):
    if norm:
        v_bounds = {'vmin' : -1, 'vmax': 1}
        normed_text = ''
    else:
        v_bounds = {}
        normed_text = '_normed'

    n_plots = len(data['trans_m'])
    rows = 2
    cols = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols)
    fig.set_figheight(rows * 5)
    fig.set_figwidth(cols * 5)
    if n_plots % 2 == 1:
        axes[-1, -1].axis('off')

    out_file = str(uuid.uuid4())+ normed_text + '.png'

    for i, (key, record) in enumerate(data['trans_m'].items()):
        row = i // cols
        col = i % cols
        w = record['w']
        w = w.reshape(w.shape[:2])
        # b = record['b']
        # b = b.reshape(b.shape[:1])
        # b = b[None, :]
        im = axes[row, col].imshow(w, cmap='gray', **v_bounds)
        #fig.colorbar(im, ax=axes[i])
        axes[row, col].set_title(key)
        fig.colorbar(im, ax=axes[row, col])

    plt.savefig(out_file);
    plt.close()
    return out_file

def draw_lr_curve_from_file(model_path, title=''):
    if '.pt' not in model_path:
        return ''
    root, _ = os.path.split(model_path)
    in_file = os.path.join(root, 'training_log.csv')
    df = pd.read_csv(in_file)
    return draw_lr_curve(df, title)


def draw_lr_curve(df, title=''):
    if not len(df):
        return ''
    out_file = str(uuid.uuid4())+'.png'
    plt.figure();
    styles = {
        'train_accuracy' : 'r-',
        'val_accuracy' : 'r--',
        'train_loss' : 'b-',
        'val_loss' : 'b--',
    }
    df.plot(xlabel="epoch", ylabel="acc / loss", style=styles);
    plt.title(title)
    plt.savefig(out_file);
    plt.close()

    return out_file


def to_pdf(data, out_file, now, template='templates/pdf.html', css='templates/style.css'):

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template)
    data['plots'] = {}

    if not data['params']['flatten']:
        data['plots']['matrix'] = draw_matrix(data)
        data['plots']['matrix_normed'] = draw_matrix(data, norm = True)

    data['plots']['front_model'] = draw_lr_curve_from_file(data['params']['front_model'], title='1. Model learning\'s curve')
    data['plots']['end_model'] = draw_lr_curve_from_file(data['params']['end_model'], title='2. Model learning\'s curve')
    data['plots']['frank'] = draw_lr_curve(data['trans_fit'], title='Frankenstein\'s learning curve')
    data['time'] = str(now)#.strftime("%Y/%m/%d, %H:%M:%S")

    html_out = template.render(data)

    HTML(string=html_out, base_url="").write_pdf(out_file, stylesheets=[css])
    
    for _, filepath in data['plots'].items():
        if filepath != '':
            os.remove(filepath)
