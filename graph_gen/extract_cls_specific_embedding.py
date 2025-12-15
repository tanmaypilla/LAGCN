import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import yaml

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time

joints = [
    'base of the spine',
    'middle of the spine',
    'neck',
    'head',
    'left shoulder',
    'left elbow',
    'left wrist',
    'left hand',
    'right shoulder',
    'right elbow',
    'right wrist',
    'right hand',
    'left hip',
    'left knee',
    'left ankle',
    'left foot',
    'right hip',
    'right knee',
    'right ankle',
    'right foot',
    'spine',
    'tip of the left hand',
    'left thumb',
    'tip of the right hand',
    'right thumb',
]

cls_info = yaml.load(open(os.path.join(os.path.dirname(__file__), 'dataset_classinfo.yaml'), 'r'), Loader=yaml.FullLoader)

class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs )    

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', 
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)

def plot_mat(mat):
    plt.imshow(mat.cpu().numpy())
    plt.colorbar()
    plt.xticks(range(len(joints)), joints, rotation=90)
    plt.yticks(range(len(joints)), joints)
    plt.show()

def get_matrix_sim(feats, temperature=1):
    a = [feats[i]['pooler_output'] for i in joints]
    a = torch.cat(a, dim=0)
    a /= a.norm(dim=1, keepdim=True)
    mat = a @ a.T
    mat *= temperature
    # mat = mat.softmax(0)
    return mat

def get_matrix_knn(feats, k):
    pass

def get_matrix_radius(feats, radius):
    pass

def get_adaptive_minimum_radius_matrix(feats, max_point):
    pass

def get_adaptive_maximum_radius_matrix(feats, min_point):
    pass

def main():
    device = 'cuda'

    tic = time.time()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
    model = model.to(device)

    print('load model cost %.6fs' % (time.time() - tic))

    # 经过tokenizer，句子前后分别被加上[CLS]和[SEP]token
    inputs = tokenizer("Hello world!", return_tensors="pt").to(device)

    input_embeddings = model.embeddings(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'])

    # last_hidden_state 为每一个token再最后一层的特征，有token个向量；pooler_output为经过全连接层的[CLS]token的特征，只有一个向量，可以用于下游任务
    # 预训练模型中有pooler的相关参数，pooler是预训练时使用“是否为下一句”这个任务训练的
    outputs = model(**inputs)

    token_hidden_output = outputs['last_hidden_state']
    pooler_output = outputs['pooler_output']

    def extract_string_features(string):
        inputs = tokenizer(string, return_tensors="pt").to(device)

        input_embeddings = model.embeddings(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'])

        outputs = model(**inputs)

        last_hidden_state = outputs['last_hidden_state']
        pooler_output = outputs['pooler_output']

        return {'last_hidden_state': last_hidden_state, 'pooler_output': pooler_output, 'embeddings': input_embeddings}

    def extract_simple_feature():
        with torch.no_grad():
            features = {}
            for k in joints:
                features[k] = extract_string_features(k)
        return features

    def extract_prompt_feature(prefix='', postfix=''):
        with torch.no_grad():
            features = {}
            for k in joints:
                input_str = k
                if prefix != '':
                    input_str = prefix + ' ' + input_str
                if postfix != '':
                    input_str = input_str + ' ' + postfix
                features[k] = extract_string_features(input_str)
        return features

    def extract_prompt_feature_and_save_sim_mat(prefix='', postfix=''):
        feat = extract_prompt_feature(prefix, postfix)
        mat = get_matrix_sim(feat)
        save_fname = os.path.join('matrix', prefix.replace(' ', '_') + '-' + postfix.replace(' ', '_') + '.npy')
        np.save(save_fname, mat.cpu().numpy())

    def plot_cls_specific_sim_mat(prefix='', postfix=''):
        all_mat = []
        for c in tqdm(cls_info['NTU60']):
            mat = get_matrix_sim(extract_prompt_feature(prefix + ' ' + c, postfix), temperature=100)
            all_mat.append(mat.cpu().numpy())

        fig, ax = plt.subplots()

        img = ax.imshow(all_mat[0])

        ax.set_xticks(range(len(joints)), joints, rotation=90)
        ax.set_yticks(range(len(joints)), joints)

        def update(idx):
            img.set_data(all_mat[idx])
            ax.set_title(cls_info['NTU60'][idx])
            return img, 

        anim = Player(fig, update, maxi=len(all_mat) - 1)
        plt.show()

    def plot_cls_specific_sim_mat_v2(prompt_str='', dataset='NTU120', with_last_punctuation=False, **kwargs):
        if not with_last_punctuation:
            prompt_str = prompt_str[:-1]
        all_mat = []
        all_feats = {}
        for c in cls_info[dataset]:
            synthesis_string = prompt_str.replace('[C]', c)
            feats = {}
            with torch.no_grad():
                for j in joints:
                    feats[j] = extract_string_features(synthesis_string.replace('[J]', j))
                mat = get_matrix_sim(feats, temperature=kwargs.get('temperature', 200))
                all_feats[c] = feats
            all_mat.append(mat.cpu().numpy())

        arr = np.stack(all_mat)
        arr = arr.reshape(arr.shape[0], -1)
        arr = torch.Tensor(arr)
        arr /= torch.norm(arr, dim=1, keepdim=True)
        diversity = ((arr @ arr.T).sum() - arr.shape[0]) / (arr.shape[0] ** 2 - arr.shape[0])

        print('prompt: "%s"' % prompt_str,'diversity %.4f' % (1 - diversity))

        if kwargs.get('vis', False):
            fig, ax = plt.subplots()

            img = ax.imshow(all_mat[0])

            ax.set_xticks(range(len(joints)), joints, rotation=90)
            ax.set_yticks(range(len(joints)), joints)

            def update(idx):
                img.set_data(all_mat[idx])
                ax.set_title(cls_info[dataset][idx])
                return img, 

            anim = Player(fig, update, maxi=len(all_mat) - 1)
            plt.show()

        feat_mats = []
        for k in all_feats:
            cls_feat_mat = []
            for c in joints:
                cls_feat_mat.append(all_feats[k][c]['pooler_output'])
            cls_feat_mat = torch.cat(cls_feat_mat, dim=0)
            feat_mats.append(cls_feat_mat)
        feat_mats = torch.stack(feat_mats).cpu().numpy()

        if kwargs.get('save', True):
            base_path = os.path.join(os.path.dirname(__file__), 'cls_matrix', dataset)
            os.makedirs(base_path, exist_ok=True)
            file_prefix = prompt_str.replace(' ', '_')
            if with_last_punctuation:
                file_prefix = file_prefix[:-1] + '-with-punctuation'
            np.save(os.path.join(base_path, file_prefix + '.npy'), np.stack(all_mat))
            np.save(os.path.join(base_path, file_prefix + '_feat.npy'), feat_mats)
        
    # tic = time.time()
    
    # simple_feature = extract_simple_feature()
    # prompt_feature1 = extract_prompt_feature(postfix='of human body')
    # prompt_feature2 = extract_prompt_feature(postfix='of human skeleton')
    # prompt_feature3 = extract_prompt_feature(postfix='for skeleton based action recognition')
    # prompt_feature4 = extract_prompt_feature(prefix='recognize human action use')
    # prompt_feature5 = extract_prompt_feature(prefix='recognize human action use', postfix='of human body')
    
    # mat = get_matrix_sim(simple_feature)
    # print('computation cost: %.6fs' % (time.time() - tic))

    # plot_mat(mat)

    # when [C] [J] of human body.
    # plot_cls_specific_sim_mat(prefix='when', postfix='of human body')
    plot_cls_specific_sim_mat_v2('when [C] [J] of human body.')
    # when [C] what will [J] act like?
    plot_cls_specific_sim_mat_v2('when [C] what will [J] act like?')
    # what will [J] act like when [C]?
    plot_cls_specific_sim_mat_v2('what will [J] act like when [C]?')
    # When a person is [C], [J] is in motion.
    plot_cls_specific_sim_mat_v2('when a person is [C], [J] is in motion.')
    # What happens to [J] when a person is [C].
    plot_cls_specific_sim_mat_v2('what happens to [J] when a person is [C].')
    # [J] function in [C]
    plot_cls_specific_sim_mat_v2('[J] function in [C].')
    # [J] is function in which action? In [C]
    # plot_cls_specific_sim_mat_v2('[J] is function in which action? In [C]')

if __name__ == '__main__':
    main()
