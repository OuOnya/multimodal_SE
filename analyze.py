import os
import librosa

from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from multiprocessing import Pool


from config import args
from preprocessing import *
from model.model import *


def analyze(model, model_name, processes=None, use_griffin=False):
    sr = args.sample_rate

    print('Caching data ...')
    cache_clean_data(elec_preprocessor=(1, 124), is_training=False,
                     force_update=False, device=args.device)
    print('Cached!')

    try:
        os.makedirs(f'Enhanced{os.sep}{model_name}{os.sep}')
    except:
        pass

    with Pool(processes) as p, \
            tqdm(args.test_noise_type) as noise_bar, \
            tqdm(total=len(args.test_SNR_type)) as SNR_bar, \
            tqdm(total=70) as test_bar:
        for noise_type in noise_bar:
            noise_bar.set_description(f'{noise_type}')
            result_file = f'Enhanced{os.sep}{model_name}{os.sep}{noise_type}.txt'
            open(result_file, 'w')

            SNR_bar.reset()
            for SNR_type in args.test_SNR_type:
                SNR_bar.set_description(f'{SNR_type}')
                noise_dir = f'{noise_type}{os.sep}a1{os.sep}{SNR_type}'
                total_sample = 0
                folder_result = []

                test_bar.reset()
                for sample in range(1, 71):
                    test_bar.set_description(f'{to_TMHINT_name(sample)}.wav')

                    Sx, phasex, elec, clean = load_data(
                        noise_dir, sample, is_training=False)
                    noisy = torch.Tensor([Sx.T]).to(args.device)
                    # noisy = None

                    with torch.no_grad():
                        _, _, Se = model(noisy, elec)
                    Se = Se[0].cpu().detach().numpy().T

                    if not use_griffin:
                        enhanced = spec2wav(Se, phasex)
                    else:
                        enhanced = librosa.core.griffinlim(10**(Se / 2),
                                                           n_iter=5,
                                                           hop_length=args.hop_length,
                                                           win_length=args.win_length,
                                                           window=args.window)

                    folder_result.append([
                        p.apply_async(pesq, (sr, clean, enhanced, 'wb')),
                        p.apply_async(stoi, (clean, enhanced, sr, False)),
                        p.apply_async(stoi, (clean, enhanced, sr, True)),
                    ])

                    total_sample += 1
                    test_bar.refresh()

                if total_sample:
                    results = [0] * 3
                    for single_result in folder_result:
                        for i, result in enumerate(single_result):
                            results[i] += result.get()
                        noise_bar.refresh()
                        SNR_bar.refresh()
                        test_bar.update()

                    results = [_ / total_sample for _ in results]

                    with open(result_file, 'a') as writer:
                        writer.write(f'SNR: {SNR_type}\n')
                        writer.write(f'PESQ:  {results[0]}\n')
                        writer.write(f'STOI:  {results[1]}\n')
                        writer.write(f'ESTOI: {results[2]}\n')
                        writer.write('\n')

                SNR_bar.update()
            noise_bar.update()


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def show_analyze(compare_dict, save_folder=None, metrics=['PESQ', 'STOI', 'ESTOI'], figsize=(20, 6)):
    if save_folder:
        save_dir = f'Enhanced{os.sep}{save_folder}{os.sep}'
        try:
            os.makedirs(save_dir)
        except:
            pass

    label_name = list(compare_dict.keys())

    compare_size = len(compare_dict)
    total_test_SNR_type = len(args.test_SNR_type)
    for noise_type in args.test_noise_type:
        compare_models = [
            f'Enhanced{os.sep}{model_name}{os.sep}{noise_type}.txt'
            for model_name in compare_dict.values()
        ]

        results = {
            metric: np.zeros((compare_size, total_test_SNR_type))
            for metric in metrics
        }

        for i, file in enumerate(compare_models):
            for line in open(file, 'r'):
                if 'SNR' in line:
                    index = args.test_SNR_type.index(line.split()[-1])

                for metric in results.keys():
                    if line.startswith(metric):
                        results[metric][i][index] = float(line.split(' ')[-1])
                        break

        x = np.arange(total_test_SNR_type)  # the label locations
        width = 1.0 / (compare_size + 1)  # the width of the bars
        bottom = [1, 0.4, 0.1]
        for j, (metric, result) in enumerate(results.items()):
            fig, ax = plt.subplots(figsize=figsize)

            for i, compared in enumerate(result):
                offset = (1 - compare_size) / 2 + i
                offset *= width
                rects = ax.bar(
                    x + offset, compared - bottom[j], width, label=label_name[i], bottom=bottom[j])
                # autolabel(ax, rects)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(f'{metric} Scores')
            ax.set_title(f'{noise_type.title()} {metric}')
            ax.set_xticks(x)
            ax.set_xticklabels(args.test_SNR_type)
            ax.legend()
            plt.tight_layout(pad=0.5)

            if save_folder:
                plt.savefig(f'{save_dir}{ax.get_title()}.png')

            plt.show()


if __name__ == '__main__':
    # ===== Analyze Model Performance =====
    model_name = 'S_CNN+E V1 (Epoch 40)'
    model = S_CNN().load_model(args.SCNN_checkpoint_path,
                               f'{model_name}.pt', args.device)
    model.to(args.device)

    analyze(model, model_name, 8, use_griffin=False)

    # ===== Show and Compare the Performance of Selected Models =====
    compare_dict = {
        'Noisy': 'Noisy',
        # 'A':                      'FCN_A (Epoch 40) (no eval)',
        # 'S':                      'S_CNN (Epoch 40)',
        # 'S norm':                 'S_CNN (norm) (Epoch 40)',
        # 'S+E (original)':         'S_CNN+E V1 (Epoch 40)',
        # 'S+E norm':               'S_CNN E (norm) (Epoch 40)',

        # 'S+E ReLU':               'S_CNN+E ReLU (Epoch 40)',
        # 'S+E ReLU LF':            'S_CNN+E ReLU LF (Epoch 40)',
        # 'S+E ReLU 5 LF':          'S_CNN+E ReLU 5 LF (Epoch 40)',
        # 'S+E LSTM ReLU':          'S_CNN+E LSTM ReLU (Epoch 40)',
        # 'S+E LSTM':               'S_CNN+E LSTM (Epoch 40)',
        # 'S+E LSTM 5 LF':          'S_CNN+E LSTM 5 LF (Epoch 40)',
        # 'S+E LSTM LF':            'S_CNN+E LSTM LF (Epoch 40)',
        # 'S/E ReLU':               'S_CNN or E ReLU (Epoch 40)',
        # 'S/E LSTM ReLU':          'S_CNN or E LSTM ReLU (Epoch 40)',
        # 'S/E LSTM':               'S_CNN or E LSTM (Epoch 40)',

        # 'E r':                    'S_CNN E (Epoch 300)',
        # 'E ReLU r':               'S_CNN E ReLU (Epoch 300) (random phase)',
        # 'E LSTM ReLU r':          'S_CNN E LSTM ReLU (Epoch 300) (random phase)',
        # 'E LSTM r':               'S_CNN E LSTM (Epoch 300)',

        # 'E ReLU':                 'S_CNN E ReLU (Epoch 300)',
        # 'E LSTM ReLU':            'S_CNN E LSTM ReLU (Epoch 300)',

        # 'S preE 10(10)':          'S_CNN LSTM PreE 10 shot (Epoch 10)',

        # 'S+E EF cat':             'S_CNN+E EF cat (Epoch 40)',
        'S+E EF cat 1L': 'S_CNN+E EF cat Linear (1 loss) (Epoch 40)',
        # 'S+E EF mean':            'S_CNN+E EF mean (Epoch 40)',
        # 'S+E CNN EF cat':         'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (Epoch 40)',
        # 'S+E CNN EF cat 40':      'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (Epoch 40 True)',
        # 'S+E CNN EF cat 1L':      'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (1 loss) (Epoch 40)',
        # 'S+E CNN EF cat 1L 40':   'S_CNN+E CNN_16_32_64 stride_1_3 EF cat (1 loss) (Epoch 40 True)',
        # 'S+E CNN EF mean':        'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (Epoch 40)',
        # 'S+E CNN EF mean 40':     'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (Epoch 40 True)',
        # 'S+E CNN EF mean 1L':     'S_CNN+E CNN_16_32_64 stride_1_3 EF mean (1 loss) (Epoch 40)',


        # 'S+E LF cat 1L':          'S_CNN+E LF cat (1 loss) (Epoch 40)',
        'S+E LF cat L 1L': 'S_CNN+E LF cat Linear (1 loss) (Epoch 40)',
        # 'S+E LF mean 1L':         'S_CNN+E LF mean (1 loss) (Epoch 40)',
        # 'S+E LF mean L 1L':       'S_CNN+E LF mean Linear (1 loss) (Epoch 40)',
        # 'S+E LF mask 1L':         'S_CNN+E LF mask (1 loss) (Epoch 40)',
        # 'S+E LF cat':             'S_CNN+E LF cat CNN (Epoch 40)',
        # 'S+E CNN np LF cat':      'S_CNN+E CNN_16_32_64 nopad LF cat CNN (Epoch 40)',
        # 'S+E CNN np LF cat 40':   'S_CNN+E CNN_16_32_64 nopad LF cat CNN (Epoch 40 True)',
        # 'S+E CNN LF cat':         'S_CNN+E CNN_16_32_64 stride_1_3 LF cat CNN (Epoch 40)',
        # 'S+E CNN LF cat 40':      'S_CNN+E CNN_16_32_64 stride_1_3 LF cat CNN (Epoch 40 True)',
        # 'S+E CNN LF cat 1L':      'S_CNN+E CNN_16_32_64 stride_1_3 LF cat (1 loss) (Epoch 40)',
        # 'S+E CNN LF cat 1L 40':   'S_CNN+E CNN_16_32_64 stride_1_3 LF cat (1 loss) (Epoch 40 True)',
        # 'S+E CNN LF cat L 1L':    'S_CNN+E CNN_16_32_64 stride_1_3 LF cat Linear (1 loss) (Epoch 40)',
        # 'S+E CNN LF mean':        'S_CNN+E CNN_16_32_64 stride_1_3 LF mean CNN (Epoch 40)',
        # 'S+E CNN LF mean 1L':     'S_CNN+E CNN_16_32_64 stride_1_3 LF mean (1 loss) (Epoch 40)',
        # 'S+E CNN LF mean L 1L':   'S_CNN+E CNN_16_32_64 stride_1_3 LF mean Linear (1 loss) (Epoch 40)',
        # 'S+E FCNN LF cat':        'S_CNN+E S_CNN LF cat CNN (Epoch 40)',
        # 'S+E FCNN LF cat 40':     'S_CNN+E S_CNN LF cat CNN (Epoch 40 True)',
    }

    show_analyze(
        compare_dict=compare_dict,
        # save_folder='Test',
        # metrics=['PESQ'],
        # figsize=(20, 6),
    )
